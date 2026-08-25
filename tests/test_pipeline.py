from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rdf import RDF, tag
from rdf.rdf import BLOG_WIDTH
from rdf.theme import ColorTheme
from rdf.svg import build_css, process
from rdf.animate import AnimationType, animate


@pytest.fixture
def rdf(tmp_path: Path) -> RDF:
    return RDF(save_root=tmp_path, name="t")


def _line(ax, color_map):
    ax.plot([0, 1, 2], [0, 1, 4], color=color_map["c1"])


def test_static_line_renders_with_themed_class(rdf: RDF):
    svg = rdf.create("line", _line)
    assert "<svg" in svg
    assert "var(--c1)" in svg
    assert (rdf.save_root / "line.svg").exists()


def test_3d_plot_renders_and_tags(rdf: RDF):
    def plot(ax, color_map):
        x = np.linspace(0, 1, 5)
        ax.plot(x, x, x, color=color_map["c1"])

    svg = rdf.create("3d", plot, is_3d=True)
    assert 'class="plot3d"' in svg


def test_signature_dispatch_passes_only_declared_kwargs(rdf: RDF):
    seen: dict = {}

    def plot(ax):
        seen["ax_only"] = ax is not None
        ax.plot([1, 2, 3], [1, 2, 3])

    rdf.create("dispatch_ax", plot)
    assert seen == {"ax_only": True}


def test_signature_dispatch_passes_fig_when_requested(rdf: RDF):
    seen: dict = {}

    def plot(fig, color_map):
        seen["fig"] = fig
        ax = fig.gca()
        ax.plot([0, 1], [0, 1], color=color_map["c2"])

    rdf.create("dispatch_fig", plot)
    assert seen["fig"] is not None


def test_build_css_is_deterministic():
    a = build_css(ColorTheme.default())
    b = build_css(ColorTheme.default())
    assert a == b
    assert "--primary" in a
    assert "--c1" in a


def test_process_themes_subplot_axes(rdf: RDF):
    def plot(fig, color_map):
        axes = fig.subplots(1, 3)
        for i, ax in enumerate(axes):
            ax.plot([0, 1, 2], [i, i + 1, i + 2], color=color_map[f"c{i + 1}"])

    svg = rdf.create("subplots", plot)
    assert svg.count('class="axis"') == 6


def test_animate_injects_style_and_class():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    import io
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    out = animate(buf.getvalue(), AnimationType.DRAW, duration=1.0, loop=False)
    assert "<style>" in out
    assert "anim-draw" in out


def test_tag_helper_sets_gid(rdf: RDF):
    seen: dict = {}

    def plot(ax):
        (line,) = ax.plot([0, 1, 2], [0, 1, 4])
        tag(line, "series-A")
        seen["line"] = line

    svg = rdf.create("tagged", plot)
    assert seen["line"].get_gid() == "series-A"
    assert 'id="series-A"' in svg


def test_canvas_is_the_standard_width(rdf: RDF):
    # A tight bbox would crop this to the content and every figure would
    # land at a different width on the page.
    for name, height in [("wide", None), ("tall", 5.0)]:
        svg = rdf.create(name, _line, height=height)
        assert f'width="{BLOG_WIDTH * 72:g}pt"' in svg


def test_axis_furniture_is_themed(rdf: RDF):
    svg = rdf.create("axes", _line)
    assert "var(--axis-color)" in svg
    assert "var(--grid-color)" in svg
    assert "stroke: #000000" not in svg


def test_render_is_deterministic(rdf: RDF):
    assert rdf.create("a", _line) == rdf.create("b", _line)


def test_returning_a_foreign_figure_raises(rdf: RDF):
    def sneaky(ax, color_map):
        fig, (a, b) = plt.subplots(1, 2)
        a.plot([0, 1], [0, 1], color=color_map["c1"])
        return fig

    with pytest.raises(ValueError, match="returned its own figure"):
        rdf.create("sneaky", sneaky)


def test_black_survives_matplotlibs_default_fill_elision(rdf: RDF):
    def black_text(ax, color_map):
        ax.plot([0, 1], [0, 1], color=color_map["c1"])
        ax.text(0.5, 0.5, "label", color=color_map["black"])

    svg = rdf.create("black", black_text)
    assert "var(--black)" in svg
