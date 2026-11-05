from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__all__ = ["Prediction", "load_grounder"]

if TYPE_CHECKING:
    from chashma.model.grounder import Prediction, load_grounder


def __getattr__(name: str):
    if name in __all__:
        from chashma.model import grounder

        return getattr(grounder, name)
    raise AttributeError(f"module 'chashma' has no attribute {name!r}")
