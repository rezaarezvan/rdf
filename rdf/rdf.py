import io
import inspect
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont

from rdf.svg import process as process_svg
from rdf.theme import AXIS_SENTINEL, BLACK_SENTINEL, GRID_SENTINEL, ColorTheme
from rdf.animate import AnimationType, animate

MPLSTYLE = Path(__file__).parent / "academic.mplstyle"

BLOG_WIDTH = 6.5
GOLDEN = 1.618

_font_checked = False


def _check_font() -> None:
    global _font_checked
    if _font_checked:
        return
    _font_checked = True
    family = plt.rcParams["font.family"][0]
    wanted = (plt.rcParams.get(f"font.{family}") or [family])[0]
    try:
        findfont(FontProperties(family=wanted), fallback_to_default=False)
    except Exception:
        warnings.warn(
            f"font {wanted!r} is not installed; matplotlib will substitute "
            f"silently. Install it and clear ~/.matplotlib/fontlist-*.json.",
            stacklevel=3,
        )


def _rc_overrides(theme: ColorTheme) -> dict:
    """Axis furniture as sentinels, so process() can make it var(--axis-color)."""
    return {
        "text.color": AXIS_SENTINEL,
        "axes.edgecolor": AXIS_SENTINEL,
        "axes.labelcolor": AXIS_SENTINEL,
        "axes.titlecolor": AXIS_SENTINEL,
        "xtick.color": AXIS_SENTINEL,
        "ytick.color": AXIS_SENTINEL,
        "xtick.labelcolor": AXIS_SENTINEL,
        "ytick.labelcolor": AXIS_SENTINEL,
        "legend.labelcolor": AXIS_SENTINEL,
        "grid.color": GRID_SENTINEL,
        "axes.prop_cycle": cycler(color=list(theme.base.values())),
    }


def _theme_3d(ax) -> None:
    """3D panes, gridlines and the z-axis ignore the 2D rcParams."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor("none")
            axis.pane.set_edgecolor(AXIS_SENTINEL)
            axis.pane.set_alpha(1.0)
            axis.line.set_color(AXIS_SENTINEL)
            axis.label.set_color(AXIS_SENTINEL)
            axis.set_tick_params(colors=AXIS_SENTINEL)
            axis._axinfo["grid"]["color"] = GRID_SENTINEL
        except (AttributeError, KeyError):
            pass


def _dispatch(plot_func: Callable[..., None], fig: Figure, ax, color_map: dict):
    """Pass only the kwargs the plot function actually declares."""
    params = inspect.signature(plot_func).parameters
    available = {"fig": fig, "ax": ax, "color_map": color_map}
    kwargs = {k: v for k, v in available.items() if k in params}
    return plot_func(**kwargs)


class RDF:
    """Generate themed, tagged, and optionally animated SVGs."""

    def __init__(
        self,
        theme: Optional[ColorTheme] = None,
        *,
        save_root: str | Path = "result",
        style: Optional[Path] = None,
        name: Optional[str] = None,
        width: float = BLOG_WIDTH,
    ) -> None:
        self.theme = theme or ColorTheme.default()
        stem = name or Path(sys.argv[0]).stem
        self.save_root = Path(save_root) / stem
        self.style = style or MPLSTYLE
        self.width = width
        assert self.style.exists(), f"style file not found: {self.style}"

    def _render(
        self,
        plot_func: Callable[..., None],
        *,
        is_3d: bool = False,
        height: Optional[float] = None,
        **kw,
    ) -> str:
        with (
            plt.style.context(str(self.style)),
            plt.rc_context(_rc_overrides(self.theme)),
        ):
            _check_font()
            fig = plt.figure(figsize=(self.width, height or self.width / GOLDEN))
            params = inspect.signature(plot_func).parameters
            ax = (
                fig.add_subplot(111, projection="3d" if is_3d else None)
                if "ax" in params
                else None
            )
            if is_3d and ax is not None:
                _theme_3d(ax)
            color_map = kw.pop("color_map", None) or {
                **self.theme.light,
                **self.theme.base,
                "black": BLACK_SENTINEL,
            }
            returned = _dispatch(plot_func, fig, ax, color_map)
            if isinstance(returned, Figure) and returned is not fig:
                plt.close(returned)
                plt.close(fig)
                raise ValueError(
                    f"{plot_func.__name__}() built and returned its own figure; rdf "
                    "renders the one it hands you. Declare `fig` and use "
                    "fig.subplots(...) instead of plt.subplots(...)."
                )
            buf = io.StringIO()
            fig.savefig(buf, format="svg", transparent=True, metadata={"Date": None})
            plt.close(fig)
            return buf.getvalue()

    def create(
        self,
        name: str,
        plot_func: Callable[..., None],
        *,
        is_3d: bool = False,
        height: Optional[float] = None,
        **kw,
    ) -> str:
        """Render a static themed SVG."""
        os.makedirs(self.save_root, exist_ok=True)
        svg = process_svg(
            self._render(plot_func, is_3d=is_3d, height=height, **kw), self.theme
        )
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
        height: Optional[float] = None,
        **kw,
    ) -> str:
        """Render an animated themed SVG."""
        static = self.create(
            f"{name}_static", plot_func, is_3d=is_3d, height=height, **kw
        )
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
