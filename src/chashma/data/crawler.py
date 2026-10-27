from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from playwright.async_api import Browser, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from chashma.data.schema import Element, PageCapture, Viewport
from chashma.geometry import Box

log = logging.getLogger(__name__)

EXTRACT_JS = (Path(__file__).parent / "extract.js").read_text()

DEFAULT_VIEWPORTS = [
    Viewport(name="desktop", width=1280, height=800),
    Viewport(name="laptop", width=1440, height=900),
    Viewport(name="mobile", width=390, height=844, is_mobile=True),
]

ROBOTS_AGENT = "chashma"
BLOCKED_TITLES = ("just a moment", "access denied", "attention required", "captcha", "403", "404")
SKIP_EXTENSIONS = (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".xml")


@dataclass
class CrawlConfig:
    out_dir: Path
    viewports: list[Viewport] = field(default_factory=lambda: list(DEFAULT_VIEWPORTS))
    color_schemes: tuple[str, ...] = ("light", "dark")
    screens: int = 2
    variants_per_page: int = 3
    links_per_site: int = 3
    concurrency: int = 4
    nav_timeout_s: float = 20.0
    settle_ms: int = 1500
    per_domain_delay_s: float = 2.0
    respect_robots: bool = True
    min_elements: int = 5
    seed: int = 0


@dataclass
class CrawlStats:
    pages: int = 0
    captures: int = 0
    skipped_robots: int = 0
    skipped_blocked: int = 0
    skipped_few_elements: int = 0
    errors: int = 0


def domain_of(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.")


def capture_id(url: str, viewport: str, scheme: str, screen: int) -> str:
    return hashlib.sha1(f"{url}|{viewport}|{scheme}|{screen}".encode()).hexdigest()[:16]


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient):
        self._client = client
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._lock = asyncio.Lock()

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with self._lock:
            if origin not in self._parsers:
                self._parsers[origin] = await self._fetch(origin)
        parser = self._parsers[origin]
        return parser is None or parser.can_fetch(ROBOTS_AGENT, url)

    async def _fetch(self, origin: str) -> RobotFileParser | None:
        try:
            resp = await self._client.get(f"{origin}/robots.txt", follow_redirects=True)
        except httpx.HTTPError:
            return None
        if resp.status_code >= 500:
            parser = RobotFileParser()
            parser.disallow_all = True
            return parser
        if resp.status_code >= 400:
            return None
        parser = RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser


class DomainThrottle:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, domain: str) -> None:
        lock = self._locks.setdefault(domain, asyncio.Lock())
        async with lock:
            elapsed = time.monotonic() - self._last.get(domain, 0.0)
            if elapsed < self.delay_s:
                await asyncio.sleep(self.delay_s - elapsed)
            self._last[domain] = time.monotonic()


def _element_key(raw: dict) -> tuple:
    x, y, w, h = raw["rect"]
    return raw["tag"], raw["name"], round(x / 2), round(y / 2), round(w / 2), round(h / 2)


def stable_elements(before: list[dict], after: list[dict]) -> list[dict]:
    after_keys = {_element_key(e) for e in after}
    return [e for e in before if _element_key(e) in after_keys]


def dedupe_elements(elements: list[Element], iou_threshold: float = 0.9) -> list[Element]:
    kept: list[Element] = []
    for el in elements:
        if all(el.box.iou(k.box) < iou_threshold for k in kept):
            kept.append(el)
    return kept


class Crawler:
    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.stats = CrawlStats()
        self._rng = random.Random(cfg.seed)
        self._throttle = DomainThrottle(cfg.per_domain_delay_s)
        self._seen: set[str] = set()
        self._variants = list(
            itertools.product(cfg.viewports, cfg.color_schemes, range(cfg.screens))
        )

    def _sample_variants(self) -> list[tuple[Viewport, str, int]]:
        first, rest = self._variants[0], self._variants[1:]
        k = max(0, min(self.cfg.variants_per_page - 1, len(rest)))
        return [first, *self._rng.sample(rest, k)]

    async def run(self, urls: Iterable[str]) -> CrawlStats:
        queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        for url in urls:
            self._enqueue(queue, url, depth=0)

        async with (
            async_playwright() as pw,
            httpx.AsyncClient(timeout=10, headers={"User-Agent": ROBOTS_AGENT}) as client,
        ):
            browser = await pw.chromium.launch()
            robots = RobotsCache(client)
            workers = [
                asyncio.create_task(self._worker(queue, browser, robots))
                for _ in range(self.cfg.concurrency)
            ]
            await queue.join()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await browser.close()
        return self.stats

    def _enqueue(self, queue: asyncio.Queue, url: str, depth: int) -> None:
        url = urldefrag(url.strip())[0]
        if url and url not in self._seen:
            self._seen.add(url)
            queue.put_nowait((url, depth))

    async def _worker(self, queue: asyncio.Queue, browser: Browser, robots: RobotsCache) -> None:
        while True:
            url, depth = await queue.get()
            try:
                if self.cfg.respect_robots and not await robots.allowed(url):
                    self.stats.skipped_robots += 1
                    continue
                links = await self._capture_page(browser, url)
                if depth == 0 and links:
                    k = min(self.cfg.links_per_site, len(links))
                    for link in self._rng.sample(links, k):
                        self._enqueue(queue, link, depth=1)
            except Exception as e:
                self.stats.errors += 1
                log.warning("failed %s: %s", url, e)
            finally:
                queue.task_done()

    async def _capture_page(self, browser: Browser, url: str) -> list[str]:
        self.stats.pages += 1
        links: list[str] = []
        for i, (viewport, scheme, screen) in enumerate(self._sample_variants()):
            cid = capture_id(url, viewport.name, scheme, screen)
            domain = domain_of(url)
            out = self.cfg.out_dir / domain
            if (out / f"{cid}.json").exists():
                continue
            await self._throttle.wait(domain)
            context = await browser.new_context(
                viewport={"width": viewport.width, "height": viewport.height},
                device_scale_factor=viewport.device_scale_factor,
                is_mobile=viewport.is_mobile,
                has_touch=viewport.is_mobile,
                color_scheme=scheme,
                locale="en-US",
            )
            try:
                page = await context.new_page()
                if not await self._load(page, url, screen, viewport):
                    return links
                if i == 0:
                    links = await self._same_site_links(page, url)
                capture = await self._snapshot(page, url, cid, viewport, scheme, out)
                if capture is not None:
                    self.stats.captures += 1
            except PlaywrightError as e:
                self.stats.errors += 1
                log.debug("variant failed %s %s: %s", url, cid, e)
            finally:
                await context.close()
        return links

    async def _load(self, page: Page, url: str, screen: int, viewport: Viewport) -> bool:
        resp = await page.goto(
            url, wait_until="domcontentloaded", timeout=self.cfg.nav_timeout_s * 1000
        )
        if resp is None or resp.status >= 400:
            self.stats.skipped_blocked += 1
            return False
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except PlaywrightError:
            pass
        if screen > 0:
            await page.evaluate("(y) => window.scrollTo(0, y)", screen * viewport.height * 0.9)
        await page.wait_for_timeout(self.cfg.settle_ms)
        title = (await page.title()).lower()
        if any(b in title for b in BLOCKED_TITLES):
            self.stats.skipped_blocked += 1
            return False
        return True

    async def _same_site_links(self, page: Page, url: str) -> list[str]:
        hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        domain = domain_of(url)
        out = []
        for href in set(hrefs):
            href = urldefrag(href)[0]
            p = urlparse(href)
            if (
                p.scheme in ("http", "https")
                and domain_of(href) == domain
                and href.rstrip("/") != url.rstrip("/")
                and not p.path.lower().endswith(SKIP_EXTENSIONS)
            ):
                out.append(href)
        return sorted(out)

    async def _snapshot(
        self, page: Page, url: str, cid: str, viewport: Viewport, scheme: str, out: Path
    ) -> PageCapture | None:
        before = await page.evaluate(EXTRACT_JS, {})
        png = await page.screenshot(animations="disabled", caret="hide")
        after = await page.evaluate(EXTRACT_JS, {})

        vw, vh = before["viewport"]
        elements = [
            Element(
                idx=i,
                tag=raw["tag"],
                role=raw["role"],
                name=raw["name"] or "",
                text=raw["text"] or "",
                type=raw["type"],
                href=raw["href"],
                placeholder=raw["placeholder"],
                is_icon=raw["is_icon"],
                bbox=Box.from_pixels(*raw["rect"], width=vw, height=vh).as_list(),
            )
            for i, raw in enumerate(stable_elements(before["elements"], after["elements"]))
        ]
        elements = dedupe_elements(elements)
        if len(elements) < self.cfg.min_elements:
            self.stats.skipped_few_elements += 1
            return None
        for i, el in enumerate(elements):
            el.idx = i

        out.mkdir(parents=True, exist_ok=True)
        (out / f"{cid}.png").write_bytes(png)
        width, height = _png_size(png)
        capture = PageCapture(
            id=cid,
            url=url,
            final_url=page.url,
            domain=domain_of(url),
            title=before["title"][:200],
            lang=before["lang"],
            viewport=viewport,
            color_scheme=scheme,  # type: ignore[arg-type]
            scroll_y=before["scroll_y"],
            screenshot=f"{cid}.png",
            image_size=(width, height),
            elements=elements,
            captured_at=datetime.now(UTC),
        )
        (out / f"{cid}.json").write_text(capture.model_dump_json(indent=1))
        return capture


def _png_size(png: bytes) -> tuple[int, int]:
    # png header, w and h are at bytes 16..24
    return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def load_seeds(path: Path, limit: int | None = None) -> list[str]:
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line and not line.startswith("http"):
            line = line.split(",", 1)[1].strip()
        urls.append(line if line.startswith("http") else f"https://{line}")
        if limit and len(urls) >= limit:
            break
    return urls


def iter_captures(root: Path) -> Iterable[PageCapture]:
    for path in sorted(root.glob("*/*.json")):
        if path.name.endswith(".instr.json"):
            continue
        yield PageCapture.model_validate(json.loads(path.read_text()))
