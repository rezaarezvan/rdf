from __future__ import annotations

import io
import os
import sys
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Optional
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .themes import ColorTheme
from .css import build_style
from . import svg_tools, tagger
from .animator import AnimationStyle, add_animation


class RDF:
    """
    Generate themed, tagged, and optionally animated SVGs.
    """

    def __init__(
        self, theme: Optional[ColorTheme] = None, *, save_root: str | Path = "result"
    ) -> None:
        self.theme = theme or ColorTheme.default()
        self.save_root = Path(save_root) / Path(sys.argv[0]).stem

    def _render_static_svg(
        self, plot_func: Callable[..., None], *, is_3d: bool, **plot_kwargs
    ) -> str:
        fig = plt.figure(facecolor="none")
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
        ax.set_facecolor("none")
        plot_kwargs.setdefault("color_map", self.theme.base)
        plot_func(ax=ax, **plot_kwargs)
        ax.tick_params(colors="#000")
        ax.xaxis.label.set_color("#000")
        ax.yaxis.label.set_color("#000")
        ax.title.set_color("#000")
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
        plt.close(fig)
        return buf.getvalue()

    def _theme_and_tag(self, raw_svg: str) -> str:
        svg = svg_tools.inject_css(raw_svg, f"<style>{build_style(self.theme)}</style>")
        svg = tagger.tag_svg(svg, self.theme)
        return svg

    def create_themed_plot(
        self,
        save_name: str,
        plot_func: Callable[..., None],
        *,
        is_3d: bool = False,
        **plot_kwargs,
    ) -> str:
        """
        Render a **static** SVG with theme CSS and tagging applied.
        """
        os.makedirs(self.save_root, exist_ok=True)
        raw_svg = self._render_static_svg(plot_func, is_3d=is_3d, **plot_kwargs)
        svg = self._theme_and_tag(raw_svg)
        out = self.save_root / f"{save_name}.svg"
        out.write_text(svg, "utf-8")
        print(f"SVG saved to {out}")
        return svg

    def create_animated_plot(
        self,
        save_name: str,
        plot_func: Callable[..., None],
        *,
        animation_type: str = "draw",
        animation_duration: float = 2.0,
        animation_delay: float = 0.0,
        loop: bool = True,
        is_3d: bool = False,
        **plot_kwargs,
    ) -> str:
        """
        Same as create_themed_plot but with animation CSS/JS injected.
        """
        static_svg = self.create_themed_plot(
            f"{save_name}_static", plot_func, is_3d=is_3d, **plot_kwargs
        )
        style_map = {
            "draw": AnimationStyle.DRAW,
            "fade": AnimationStyle.FADE,
            "pulse": AnimationStyle.PULSE,
        }
        style = style_map.get(animation_type.lower(), AnimationStyle.NONE)
        animated = add_animation(
            static_svg,
            style=style,
            duration=animation_duration,
            delay=animation_delay,
            loop=loop,
        )
        out = self.save_root / f"{save_name}.svg"
        out.write_text(animated, "utf-8")
        print(f"Animated SVG saved to {out}")
        return animated


__all__ = ["RDF", "ColorTheme"]
