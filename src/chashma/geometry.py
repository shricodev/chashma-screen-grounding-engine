from __future__ import annotations

from dataclasses import dataclass


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int]:
        return round(self.x * width), round(self.y * height)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError(f"malformed box: {self}")

    @classmethod
    def from_pixels(cls, x: float, y: float, w: float, h: float, width: int, height: int) -> Box:
        return cls(
            _clamp(x / width),
            _clamp(y / height),
            _clamp((x + w) / width),
            _clamp((y + h) / height),
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, p: Point) -> bool:
        return self.x1 <= p.x <= self.x2 and self.y1 <= p.y <= self.y2

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )

    def as_list(self) -> list[float]:
        return [round(v, 5) for v in (self.x1, self.y1, self.x2, self.y2)]
