import random

import pytest

from chashma.data.annotate import clean_instruction, parse_annotation
from chashma.data.build import split_for
from chashma.data.crawler import dedupe_elements, stable_elements
from chashma.data.schema import Element
from chashma.data.templates import template_instructions
from chashma.eval.harness import EvalExample, score, summarize
from chashma.geometry import Box, Point
from chashma.model.format import encode_point, parse_point
from chashma.model.grounder import Prediction


def el(idx, bbox, name="x", role="button", **kw):
    return Element(idx=idx, tag="button", role=role, name=name, bbox=bbox, **kw)


def test_box_from_pixels_clips_to_screen():
    b = Box.from_pixels(-10, 790, 100, 50, width=1280, height=800)
    assert b.x1 == 0 and b.y2 == 1.0
    assert b.contains(b.center)


def test_point_roundtrip_lands_in_same_cell():
    p = Point(0.4237, 0.9999)
    assert encode_point(p) == "<point>423,999</point>"
    back = parse_point(encode_point(p))
    assert abs(back.x - p.x) < 1e-3 and abs(back.y - p.y) < 1e-3


@pytest.mark.parametrize("text", ["", "no idea", "<point>1200,5</point>"])
def test_parse_point_rejects_garbage(text):
    assert parse_point(text) is None


def test_parse_point_tolerates_missing_tags():
    assert parse_point("sure: 500, 250") == Point(0.5005, 0.2505)


def test_overlay_leaks_are_dropped():
    assert clean_instruction("click the box labeled 3") is None
    assert clean_instruction("press the item marked in red") is None
    assert clean_instruction("  open   pricing ") == "open pricing"


def test_parse_annotation_maps_marks_to_elements():
    targets = [el(7, [0, 0, 0.1, 0.1]), el(9, [0.5, 0.5, 0.6, 0.6])]
    reply = {"marks": [
        {"id": 1, "direct": "click Login", "functional": "sign in", "visual": "box 1"},
        {"id": 2, "skip": True},
        {"id": 5, "direct": "out of range"},
    ]}
    instructions, skipped = parse_annotation(reply, targets)
    assert [(i.element_idx, i.style) for i in instructions] == [(7, "direct"), (7, "functional")]
    assert skipped == [9]


def test_templates_skip_ambiguous_labels():
    els = [el(0, [0, 0, .1, .1], "Buy"), el(1, [.2, .2, .3, .3], "Buy"),
           el(2, [.5, .5, .6, .6], "Cart")]
    out = template_instructions(els, random.Random(0))
    assert [e.idx for e, _ in out] == [2]


def test_stable_elements_drop_moving_ones():
    a = {"tag": "a", "name": "slide", "rect": [0, 0, 100, 20]}
    b = {"tag": "a", "name": "home", "rect": [0, 50, 100, 20]}
    moved = {**a, "rect": [300, 0, 100, 20]}
    assert stable_elements([a, b], [moved, b]) == [b]


def test_dedupe_nested_duplicates():
    els = [el(0, [0, 0, .2, .1]), el(1, [0, 0, .2, .1]), el(2, [.5, .5, .6, .6])]
    assert [e.idx for e in dedupe_elements(els)] == [0, 2]


def test_split_is_per_domain_and_deterministic():
    assert split_for("example.com") == split_for("example.com")
    splits = {split_for(f"site{i}.com") for i in range(500)}
    assert splits == {"train", "val", "test"}


def test_scoring_and_breakdown():
    ex = EvalExample("a", "click", Box(0.1, 0.1, 0.2, 0.2), lambda: None, {"type": "icon"})
    ex2 = EvalExample("b", "click", Box(0.5, 0.5, 0.6, 0.6), lambda: None, {"type": "text"})
    records = [
        score(ex, Prediction(Point(0.15, 0.15), "", 10)),
        score(ex2, Prediction(None, "??", 20)),
    ]
    report = summarize("m", "d", records)
    assert report.accuracy == 0.5 and report.parse_rate == 0.5
    assert report.breakdown["type"]["icon"]["accuracy"] == 1.0


def test_screenspot_platform_groups():
    from chashma.eval.screenspot import resolve_platforms

    assert resolve_platforms(["web"]) == {"tool", "shop", "gitlab", "forum"}
    assert resolve_platforms(["ios", "desktop"]) == {"ios", "macos", "windows"}
    assert resolve_platforms(None) is None
    with pytest.raises(ValueError):
        resolve_platforms(["webb"])


def test_agent_step_log_line():
    pytest.importorskip("playwright")
    from chashma.agent.browser import Step

    ok = Step(1, {"action": "type", "target": "search", "text": "asyncio"}, "u")
    bad = Step(2, {"action": "click", "target": "Login"}, "u", error="not found")
    assert ok.describe() == "1. type 'search' text='asyncio' -> ok"
    assert bad.describe() == "2. click 'Login' -> FAILED: not found"


def test_publish_prepares_card_and_skips_training_state(tmp_path):
    from chashma.publish import publish

    ckpt = tmp_path / "ckpt"
    (ckpt / "checkpoint-200").mkdir(parents=True)
    for name in ("model.safetensors", "config.json", "training_args.bin"):
        (ckpt / name).write_text("x")
    (ckpt / "chashma.json").write_text('{"base_model": "org/base", "train_samples": 1234}')
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "r.json").write_text(
        '{"model": "ckpt", "dataset": "chashma-test", "n": 10, "accuracy": 0.5,'
        ' "latency_p50_ms": 40.0}'
    )

    out = publish(ckpt, "me/chashma-test", reports, out_root=tmp_path / "dist")
    files = {p.name for p in out.iterdir()}
    assert files == {"model.safetensors", "config.json", "chashma.json", "README.md"}
    card = (out / "README.md").read_text()
    assert "base_model: org/base" in card and "1,234" in card and "50.0%" in card
