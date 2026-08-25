from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FigureSpec:
    name: str
    fn: Callable
    animation: Optional[str] = None
    height: Optional[float] = None
    kw: dict[str, Any] = field(default_factory=dict)


_REGISTRY: list[FigureSpec] = []


def figure(
    name: Optional[str] = None,
    *,
    animation: Optional[str] = None,
    height: Optional[float] = None,
    **kw,
):
    """
    Register a plot function for later building.
    Does not wrap fn.
    Width is fixed (RDF.width); `height` in inches is the one dimension a
    figure may vary. Equal-aspect diagrams need it or they collapse to a
    narrow strip of ink on the default golden-ratio canvas.
    """

    def deco(fn: Callable) -> Callable:
        _REGISTRY.append(
            FigureSpec(
                name=name or fn.__name__,
                fn=fn,
                animation=animation,
                height=height,
                kw=kw,
            )
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
                spec.name,
                spec.fn,
                animation=spec.animation,
                height=spec.height,
                **spec.kw,
            )
        else:
            rdf.create(spec.name, spec.fn, height=spec.height, **spec.kw)
