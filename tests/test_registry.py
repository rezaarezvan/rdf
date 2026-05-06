from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

from rdf import figure, build, registered, reset, RDF


@pytest.fixture(autouse=True)
def _clean():
    reset()
    yield
    reset()


def test_figure_registers_and_does_not_wrap():
    @figure("plot_a")
    def plot_a(ax):
        ax.plot([0, 1], [0, 1])

    specs = registered()
    assert len(specs) == 1
    assert specs[0].name == "plot_a"
    assert specs[0].fn is plot_a  # decorator must not wrap


def test_figure_default_name_is_function_name():
    @figure()
    def my_figure(ax):
        ax.plot([0, 1], [0, 1])

    assert registered()[0].name == "my_figure"


def test_build_renders_static_and_animated(tmp_path: Path):
    @figure("static_one")
    def static_one(ax):
        ax.plot([0, 1, 2], [0, 1, 4])

    @figure("anim_one", animation="draw", duration=1.0, loop=False)
    def anim_one(ax):
        ax.plot([0, 1, 2], [2, 1, 0])

    build(RDF(save_root=tmp_path, name="t"))
    out = tmp_path / "t"
    assert (out / "static_one.svg").exists()
    assert (out / "anim_one.svg").exists()
    assert (out / "anim_one_static.svg").exists()
    assert "anim-draw" in (out / "anim_one.svg").read_text()


def test_reset_clears_registry():
    @figure("temp")
    def temp(ax):
        ax.plot([0, 1], [0, 1])

    assert registered()
    reset()
    assert not registered()
