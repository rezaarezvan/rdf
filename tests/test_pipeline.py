from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rdf import RDF, tag
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


def test_save_name_deprecation_still_works(rdf: RDF):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rdf.create_themed_plot(save_name="legacy", plot_func=_line)
    assert any(issubclass(rec.category, DeprecationWarning) for rec in w)
    assert (rdf.save_root / "legacy.svg").exists()


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
