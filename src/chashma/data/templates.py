from __future__ import annotations

import random
from collections import Counter

from chashma.data.schema import Element

ROLE_NOUNS = {
    "link": "link",
    "button": "button",
    "textbox": "field",
    "searchbox": "search box",
    "combobox": "dropdown",
    "checkbox": "checkbox",
    "radio": "option",
    "tab": "tab",
    "menuitem": "menu item",
    "switch": "toggle",
}

CLICK_VERBS = ["click", "click on", "press", "select", "open"]
TAP_VERBS = ["tap", "tap on", "press"]


def _label(el: Element) -> str:
    return (el.name or el.placeholder or "").strip()


def template_instruction(el: Element, rng: random.Random, mobile: bool = False) -> str | None:
    label = _label(el)
    if not label or len(label) > 60:
        return None
    noun = ROLE_NOUNS.get(el.role, "")
    if el.role in ("textbox", "searchbox"):
        options = [f"click the {label} {noun}", f"focus the {label} field", f"type into {label}"]
    else:
        verb = rng.choice(TAP_VERBS if mobile else CLICK_VERBS)
        if el.is_icon:
            options = [f"{verb} the {label} icon", f"{verb} the {label} button"]
        else:
            options = [f"{verb} {label}", f'{verb} "{label}"']
            if noun:
                options.append(f"{verb} the {label} {noun}")
    return rng.choice(options)


def template_instructions(
    elements: list[Element], rng: random.Random, mobile: bool = False, k: int | None = None
) -> list[tuple[Element, str]]:
    counts = Counter(_label(e).lower() for e in elements)
    unique = [e for e in elements if _label(e) and counts[_label(e).lower()] == 1]
    if k is not None:
        unique = rng.sample(unique, min(k, len(unique)))
    out = []
    for el in unique:
        if (text := template_instruction(el, rng, mobile)) is not None:
            out.append((el, text))
    return out
