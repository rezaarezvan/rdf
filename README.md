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

Or build a whole directory:
```bash
python -m rdf build path/to/figures/
```

## CSS
Embedded SVGs reference these CSS variables.
Define them in your site, override per theme:

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

## style
The default mplstyle targets web SVG: fixed blog width, transparent background, no top/right spines, frameless legends, and **EB Garamond** with STIX for maths so figures read as the prose they sit in.
Install EB Garamond (then clear `~/.matplotlib/fontlist-*.json`) or rdf warns and matplotlib substitutes silently.

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
`scripts/<level>/<course>/` mirrors my blog (`bachelor`, `master`, `exchange`, `essays`); figures build to `result/<level>/<course>/`.
