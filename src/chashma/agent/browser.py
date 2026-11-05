from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.async_api import BrowserContext, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from chashma.data.marks import encode_image, load_font
from chashma.llm import VisionLLM, parse_json
from chashma.model.grounder import Grounder, Prediction

log = logging.getLogger(__name__)

PLANNER_PROMPT = """\
You control a web browser to complete a task.

Task: {task}
Current URL: {url}
Steps so far:
{history}

Look at the screenshot and choose the single next action. Reply with JSON only, one of:
{{"action": "click", "target": "<element to click>", "reason": "..."}}
{{"action": "type", "target": "<field>", "text": "<text>", "submit": true, "reason": "..."}}
{{"action": "scroll", "direction": "down" | "up", "reason": "..."}}
{{"action": "press", "key": "Enter" | "Escape" | "Tab", "reason": "..."}}
{{"action": "navigate", "url": "<absolute url>", "reason": "..."}}
{{"action": "back", "reason": "..."}}
{{"action": "done", "answer": "<result for the user>", "reason": "..."}}

Describe targets by visible text, purpose, or appearance and position, the way you \
would tell a person where to click. Never give coordinates. If the last step failed, \
try a different description or a different approach."""

ACTIONS = {"click", "type", "scroll", "press", "navigate", "back", "done"}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(ACTIONS)},
        "target": {"type": "string"},
        "text": {"type": "string"},
        "submit": {"type": "boolean"},
        "direction": {"type": "string", "enum": ["up", "down"]},
        "key": {"type": "string"},
        "url": {"type": "string"},
        "answer": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}


@dataclass
class Step:
    index: int
    action: dict
    url: str
    point: tuple[int, int] | None = None
    grounder_raw: str = ""
    grounder_ms: float = 0.0
    error: str | None = None
    screenshot: str = ""

    def describe(self) -> str:
        a = self.action
        kind = a.get("action", "?")
        what = a.get("target") or a.get("url") or a.get("key") or a.get("direction") or ""
        line = f"{self.index}. {kind} {what!r}"
        if kind == "type":
            line += f" text={a.get('text', '')!r}"
        return line + (f" -> FAILED: {self.error}" if self.error else " -> ok")


@dataclass
class AgentResult:
    task: str
    start_url: str
    finished: bool
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    planner_input_tokens: int = 0
    planner_output_tokens: int = 0


class BrowserAgent:
    def __init__(
        self,
        planner: VisionLLM,
        grounder: Grounder,
        out_dir: Path,
        max_steps: int = 15,
        viewport: tuple[int, int] = (1280, 800),
        headless: bool = True,
        planner_max_side: int = 1024,
    ):
        self.planner = planner
        self.grounder = grounder
        self.out_dir = out_dir
        self.max_steps = max_steps
        self.viewport = viewport
        self.headless = headless
        self.planner_max_side = planner_max_side
        self._page: Page | None = None

    async def run(self, task: str, start_url: str) -> AgentResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        result = AgentResult(task=task, start_url=start_url, finished=False, answer=None)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={"width": self.viewport[0], "height": self.viewport[1]},
                locale="en-US",
            )
            self._follow_new_tabs(context)
            self._page = await context.new_page()
            await self._page.goto(start_url, wait_until="domcontentloaded")
            try:
                for i in range(1, self.max_steps + 1):
                    step = await self._step(i, task, result.steps)
                    result.steps.append(step)
                    if step.action.get("action") == "done" and not step.error:
                        result.finished = True
                        result.answer = step.action.get("answer")
                        break
            finally:
                await browser.close()

        result.planner_input_tokens = self.planner.usage.input_tokens
        result.planner_output_tokens = self.planner.usage.output_tokens
        self._save(result)
        return result

    def _follow_new_tabs(self, context: BrowserContext) -> None:
        def on_page(page: Page) -> None:
            self._page = page

        context.on("page", on_page)

    async def _settle(self) -> None:
        with contextlib.suppress(PlaywrightError):
            await self._page.wait_for_load_state("domcontentloaded", timeout=10_000)
        await self._page.wait_for_timeout(800)

    async def _step(self, i: int, task: str, history: list[Step]) -> Step:
        await self._settle()
        png = await self._page.screenshot()
        image = Image.open(io.BytesIO(png)).convert("RGB")
        step = Step(index=i, action={}, url=self._page.url)

        prompt = PLANNER_PROMPT.format(
            task=task,
            url=self._page.url,
            history="\n".join(s.describe() for s in history[-10:]) or "(none yet)",
        )
        try:
            reply = await self.planner.ask(
                encode_image(image, self.planner_max_side), prompt, schema=ACTION_SCHEMA
            )
            step.action = parse_json(reply.text)
            if step.action.get("action") not in ACTIONS:
                raise ValueError(f"unknown action {step.action.get('action')!r}")
        except ValueError as e:
            step.error = f"bad planner reply: {e}"
            self._snapshot(step, image)
            return step

        try:
            await self._execute(step, image)
        except PlaywrightError as e:
            step.error = str(e).splitlines()[0][:200]
        self._snapshot(step, image)
        log.info(step.describe())
        return step

    async def _ground(self, image: Image.Image, target: str) -> Prediction:
        if hasattr(self.grounder, "apredict"):
            return await self.grounder.apredict(image, target)
        return await asyncio.to_thread(self.grounder.predict, image, target)

    async def _execute(self, step: Step, image: Image.Image) -> None:
        a = step.action
        kind = a["action"]
        page = self._page

        if kind in ("click", "type"):
            target = str(a.get("target", "")).strip()
            if not target:
                step.error = "no target given"
                return
            pred = await self._ground(image, target)
            step.grounder_raw, step.grounder_ms = pred.raw, pred.latency_ms
            if pred.point is None:
                step.error = "grounder could not locate the target"
                return
            x, y = pred.point.to_pixels(*image.size)
            step.point = (x, y)
            await page.mouse.click(x, y)
            if kind == "type":
                await page.keyboard.press("ControlOrMeta+A")
                await page.keyboard.type(str(a.get("text", "")), delay=15)
                if a.get("submit"):
                    await page.keyboard.press("Enter")
        elif kind == "scroll":
            dy = self.viewport[1] * 0.8 * (-1 if a.get("direction") == "up" else 1)
            await page.mouse.wheel(0, dy)
        elif kind == "press":
            await page.keyboard.press(str(a.get("key", "Enter")))
        elif kind == "navigate":
            await page.goto(str(a["url"]), wait_until="domcontentloaded")
        elif kind == "back":
            await page.go_back(wait_until="domcontentloaded")

    def _snapshot(self, step: Step, image: Image.Image) -> None:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        if step.point:
            x, y = step.point
            for r, color in ((14, "#ff1744"), (4, "#ff1744")):
                draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        caption = step.describe()
        draw.rectangle([0, 0, img.width, 28], fill="#000000")
        draw.text((8, 5), caption[:150], fill="#ffffff", font=load_font(15))
        step.screenshot = f"step_{step.index:02d}.png"
        img.save(self.out_dir / step.screenshot)

    def _save(self, result: AgentResult) -> None:
        (self.out_dir / "trajectory.json").write_text(json.dumps(asdict(result), indent=2))
        frames = [Image.open(self.out_dir / s.screenshot) for s in result.steps if s.screenshot]
        if frames:
            frames = [f.resize((960, round(960 * f.height / f.width))) for f in frames]
            frames[0].save(
                self.out_dir / "trajectory.gif",
                save_all=True,
                append_images=frames[1:],
                duration=1500,
                loop=0,
            )
