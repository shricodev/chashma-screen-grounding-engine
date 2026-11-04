from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.async_api")

from chashma.data.crawler import EXTRACT_JS  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "page.html"


async def test_extract_finds_only_clickable_visible_elements():
    try:
        async with playwright.async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            await page.goto(FIXTURE.as_uri())
            result = await page.evaluate(EXTRACT_JS, {})
            await browser.close()
    except playwright.Error as e:
        pytest.skip(f"chromium not available: {e}")

    names = {e["name"] for e in result["elements"]}
    assert {"Pricing", "Search", "you@example.com", "Close"} <= names
    # hidden, invisible, disabled, covered by the modal, below the fold, aria-hidden
    for bad in ("Hidden", "Invisible", "Disabled", "Covered by modal", "Far below", "Aria hidden"):
        assert bad not in names

    search = next(e for e in result["elements"] if e["name"] == "Search")
    assert search["is_icon"] and search["role"] == "button"
