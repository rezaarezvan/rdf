from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FigureSpec:
    name: str
    fn: Callable
    animation: Optional[str] = None
    kw: dict[str, Any] = field(default_factory=dict)


_REGISTRY: list[FigureSpec] = []


def figure(name: Optional[str] = None, *, animation: Optional[str] = None, **kw):
    """Register a plot function for later building. Does not wrap fn."""

    def deco(fn: Callable) -> Callable:
        _REGISTRY.append(
            FigureSpec(name=name or fn.__name__, fn=fn, animation=animation, kw=kw)
        )
        return fn

    return deco


def registered() -> list[FigureSpec]:
    return list(_REGISTRY)


def reset() -> None:
    _REGISTRY.clear()


def build(rdf=None) -> None:
    """Render every registered figure."""
    from rdf import RDF

    rdf = rdf or RDF()
    for spec in _REGISTRY:
        if spec.animation:
            rdf.create_animated(
                spec.name, spec.fn, animation=spec.animation, **spec.kw
            )
        else:
            rdf.create(spec.name, spec.fn, **spec.kw)
