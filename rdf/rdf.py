from __future__ import annotations

import os
import io
import sys
import inspect
import warnings
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional

from rdf.theme import ColorTheme
from rdf.svg import process as process_svg
from rdf.animate import AnimationType, animate

MPLSTYLE = Path(__file__).parent / "academic.mplstyle"


def _resolve_name(name: Optional[str], save_name: Optional[str], method: str) -> str:
    if save_name is not None:
        warnings.warn(
            f"{method}(save_name=...) is deprecated; use name=...",
            DeprecationWarning,
            stacklevel=3,
        )
        if name is None:
            return save_name
    assert name is not None, f"{method}() missing required argument: name"
    return name


def _dispatch(
    plot_func: Callable[..., None],
    fig: plt.Figure,
    ax,
    color_map: dict,
) -> None:
    """Pass only the kwargs the plot function actually declares."""
    params = inspect.signature(plot_func).parameters
    available = {"fig": fig, "ax": ax, "color_map": color_map}
    kwargs = {k: v for k, v in available.items() if k in params}
    plot_func(**kwargs)


class RDF:
    """Generate themed, tagged, and optionally animated SVGs."""

    def __init__(
        self,
        theme: Optional[ColorTheme] = None,
        *,
        save_root: str | Path = "result",
        style: Optional[Path] = None,
        name: Optional[str] = None,
    ) -> None:
        self.theme = theme or ColorTheme.default()
        stem = name or Path(sys.argv[0]).stem
        self.save_root = Path(save_root) / stem
        self.style = style or MPLSTYLE
        assert self.style.exists(), f"style file not found: {self.style}"

    def _render(
        self, plot_func: Callable[..., None], *, is_3d: bool = False, **kw
    ) -> str:
        with plt.style.context(str(self.style)):
            fig = plt.figure()
            params = inspect.signature(plot_func).parameters
            ax = (
                fig.add_subplot(111, projection="3d" if is_3d else None)
                if "ax" in params
                else None
            )
            color_map = kw.pop("color_map", None) or {**self.theme.base, **self.theme.light}
            _dispatch(plot_func, fig, ax, color_map)
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
            plt.close(fig)
            return buf.getvalue()

    def create(
        self,
        name: Optional[str] = None,
        plot_func: Optional[Callable[..., None]] = None,
        *,
        is_3d: bool = False,
        save_name: Optional[str] = None,
        **kw,
    ) -> str:
        """Render a static themed SVG."""
        name = _resolve_name(name, save_name, "create")
        assert plot_func is not None, "create() missing required argument: plot_func"
        os.makedirs(self.save_root, exist_ok=True)
        svg = process_svg(self._render(plot_func, is_3d=is_3d, **kw), self.theme)
        out = self.save_root / f"{name}.svg"
        out.write_text(svg, "utf-8")
        print(f"saved: {out}")
        return svg

    def create_animated(
        self,
        name: Optional[str] = None,
        plot_func: Optional[Callable[..., None]] = None,
        *,
        animation: str = "draw",
        duration: float = 2.0,
        delay: float = 0.0,
        loop: bool = True,
        is_3d: bool = False,
        save_name: Optional[str] = None,
        **kw,
    ) -> str:
        """Render an animated themed SVG."""
        name = _resolve_name(name, save_name, "create_animated")
        assert plot_func is not None, "create_animated() missing required argument: plot_func"
        static = self.create(f"{name}_static", plot_func, is_3d=is_3d, **kw)
        anim_type = {
            "draw": AnimationType.DRAW,
            "fade": AnimationType.FADE,
            "pulse": AnimationType.PULSE,
        }.get(animation.lower(), AnimationType.NONE)
        animated = animate(static, anim_type, duration=duration, delay=delay, loop=loop)
        out = self.save_root / f"{name}.svg"
        out.write_text(animated, "utf-8")
        print(f"saved: {out}")
        return animated

    # Aliases for backwards compatibility
    create_themed_plot = create
    create_animated_plot = create_animated
