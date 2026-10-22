from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from chashma.geometry import Box


class Viewport(BaseModel):
    name: str
    width: int
    height: int
    device_scale_factor: float = 1.0
    is_mobile: bool = False


class Element(BaseModel):
    idx: int
    tag: str
    role: str
    name: str = ""
    text: str = ""
    type: str | None = None
    href: str | None = None
    placeholder: str | None = None
    is_icon: bool = False
    bbox: list[float] = Field(min_length=4, max_length=4)

    @property
    def box(self) -> Box:
        return Box(*self.bbox)


class PageCapture(BaseModel):
    id: str
    url: str
    final_url: str
    domain: str
    title: str = ""
    lang: str | None = None
    viewport: Viewport
    color_scheme: Literal["light", "dark"] = "light"
    scroll_y: float = 0
    screenshot: str
    image_size: tuple[int, int]
    elements: list[Element]
    captured_at: datetime


InstructionStyle = Literal["direct", "functional", "visual", "template"]


class GroundingSample(BaseModel):
    id: str
    image: str
    image_size: tuple[int, int]
    instruction: str
    bbox: list[float] = Field(min_length=4, max_length=4)
    style: InstructionStyle
    source: str
    capture_id: str
    domain: str
    url: str
    viewport: str
    color_scheme: Literal["light", "dark"]
    element_role: str
    is_icon: bool
    split: Literal["train", "val", "test"] = "train"

    @property
    def box(self) -> Box:
        return Box(*self.bbox)
