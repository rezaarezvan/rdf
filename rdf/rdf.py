from __future__ import annotations

import os
import io
import sys
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional

from rdf.theme import ColorTheme
from rdf.svg import process as process_svg
from rdf.animate import AnimationType, animate

MPLSTYLE = Path(__file__).parent / "academic.mplstyle"


class RDF:
    """Generate themed, tagged, and optionally animated SVGs."""

    def __init__(
        self,
        theme: Optional[ColorTheme] = None,
        *,
        save_root: str | Path = "result",
        style: Optional[Path] = None,
    ) -> None:
        self.theme = theme or ColorTheme.default()
        self.save_root = Path(save_root) / Path(sys.argv[0]).stem
        self.style = style or MPLSTYLE
        assert self.style.exists(), f"style file not found: {self.style}"

    def _render(
        self, plot_func: Callable[..., None], *, is_3d: bool = False, **kw
    ) -> str:
        with plt.style.context(str(self.style)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d" if is_3d else None)
            kw.setdefault("color_map", {**self.theme.base, **self.theme.light})
            plot_func(ax=ax, **kw)
            # Force axis colors to black (will be themed by CSS)
            ax.tick_params(colors="#000")
            for attr in ["xaxis", "yaxis"]:
                getattr(ax, attr).label.set_color("#000")
            ax.title.set_color("#000")
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
            plt.close(fig)
            return buf.getvalue()

    def create(
        self, name: str, plot_func: Callable[..., None], *, is_3d: bool = False, **kw
    ) -> str:
        """Render a static themed SVG."""
        os.makedirs(self.save_root, exist_ok=True)
        svg = process_svg(self._render(plot_func, is_3d=is_3d, **kw), self.theme)
        out = self.save_root / f"{name}.svg"
        out.write_text(svg, "utf-8")
        print(f"saved: {out}")
        return svg

    def create_animated(
        self,
        name: str,
        plot_func: Callable[..., None],
        *,
        animation: str = "draw",
        duration: float = 2.0,
        delay: float = 0.0,
        loop: bool = True,
        is_3d: bool = False,
        **kw,
    ) -> str:
        """Render an animated themed SVG."""
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
