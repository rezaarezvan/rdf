# rdf
rezvan's dynamic figures

## setup
```bash
git clone https://github.com/rezaarezvan/rdf.git
cd rdf/
uv sync
```

## usage
```python
import numpy as np
from rdf import figure, build

@figure("sine")
def plot_sine(ax, color_map):
    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.sin(x), color=color_map["c1"])
    ax.set_xlabel("x"); ax.set_ylabel("sin x")

@figure("cosine", animation="draw", duration=2.0)
def plot_cos(ax, color_map):
    x = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x, np.cos(x), color=color_map["c2"])


@figure("square", height=5.0)          # width is fixed; height is yours
def plot_square(ax, color_map):
    ax.set_aspect("equal")

if __name__ == "__main__":
    build()
```

Each plot function receives only what it asks for (`ax`, `fig`, `color_map`).
Take `fig` if you need subplots — call `fig.subplots(...)`, never
`plt.subplots(...)`; rdf renders the figure it hands you and refuses one you
build yourself.

Or build a whole directory:
```bash
python -m rdf build path/to/figures/
```

A file whose name starts with `_` is a helper shared within a set, not a figure
file; `build` skips it and puts its directory on `sys.path` so siblings can
import it.

## canvas
Every figure renders at one width — `BLOG_WIDTH`, 6.5in / 468pt / 624 CSS px —
so a page of figures shares one size, one set of margins, and one apparent type
size. `height=` is the only free dimension; give it to equal-aspect diagrams,
which otherwise collapse to a narrow strip on a golden-ratio canvas. Override
with `RDF(width=...)`.

Output is deterministic: the same source renders byte-identical SVGs, so a
rebuild is a readable diff.

## CSS
Embedded SVGs reference these CSS variables. Define them in your site, override per theme:

The default palette is Okabe–Ito (colorblind-safe), with the yellow slot darkened for light surfaces, `c7` reserved as the neutral gray, and a lightness-retargeted dark variant.
```css
:root {
  /* axis furniture: text, spines, ticks; and the gridlines */
  --axis-color: #2a241c;
  --grid-color: rgba(42, 36, 28, 0.14);

  /* indexed slots — color_map["c1"]..["c8"] in plot code map to these */
  --c1: #D55E00;  --c2: #009E73;  --c3: #56B4E9;  --c4: #CC79A7;
  --c5: #E69F00;  --c6: #C7B42E;  --c7: #999999;  --c8: #0072B2;
}

[data-theme="dark"] {
  --axis-color: #e6dfd2;
  --grid-color: rgba(230, 223, 210, 0.16);
  --c1: #E36A1C;  --c2: #23AB7F;  --c3: #3F9FD3;  --c4: #C673A1;
  --c5: #C38819;  --c6: #847500;  --c7: #929292;  --c8: #1F81C2;
}
```

Axis colours are not yours to set in a style file — rdf injects them as
sentinels so they come out as `var(--axis-color)` / `var(--grid-color)`. Same
for `color_map["black"]`: matplotlib omits a `fill` that equals SVG's default,
so a literal black would lose its theming, and rdf routes it through a sentinel
to stop that.

## style
The default mplstyle targets web SVG: fixed blog width, transparent background,
no top/right spines, frameless legends, and **EB Garamond** with STIX for maths
so figures read as the prose they sit in. Install EB Garamond (then clear
`~/.matplotlib/fontlist-*.json`) or rdf warns and matplotlib substitutes
silently.

For a paper/PDF style or anything else, pass your own:

```python
from pathlib import Path
RDF(style=Path("paper.mplstyle"))
```

## CLI
```
python -m rdf build <path>             render every @figure in a file or directory (recursive)
python -m rdf inkscape <file.svg>      convert Inkscape SVG to RDF format
python -m rdf palette --css            CSS variables block
python -m rdf palette --inkscape       Inkscape palette (.gpl)
```

## gallery
`scripts/<level>/<course>/` mirrors the blog (`bachelor`, `master`, `exchange`, `essays`); figures build to `result/<level>/<course>/`.
