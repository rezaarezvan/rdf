# rdf

Themed, tagged SVG figures for math-heavy ML blogs. Pure-CSS theming — no JS required for light/dark.

## Install

```bash
python3 -m pip install -e .
```

## Quickstart

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

if __name__ == "__main__":
    build()
```

Or build a whole directory without entry-point boilerplate:

```bash
python -m rdf build path/to/figures/
```

Each plot function receives only what it asks for (`ax`, `fig`, `color_map`). Take `fig` if you need subplots or 3D.

## CSS contract

Embedded SVGs reference these CSS variables. Define them in your site, override per theme:

```css
:root {
  /* axis foreground (text, spines, ticks) */
  --axis-color: #1a1a1a;

  /* semantic colors — used when you write color="primary" etc. */
  --primary: #d62728;  --secondary: #2ca02c;  --tertiary: #1f77b4;
  --quaternary: #9467bd; --quinary: #8c564b;  --senary: #e377c2;
  --septenary: #7f7f7f;  --octonary: #bcbd22;

  /* indexed slots — color_map["c1"]..["c8"] in plot code map to these */
  --c1: #d62728;  --c2: #2ca02c;  --c3: #1f77b4;  --c4: #9467bd;
  --c5: #8c564b;  --c6: #e377c2;  --c7: #7f7f7f;  --c8: #bcbd22;
}

[data-theme="dark"] {
  --axis-color: #eaeaea;
  --c1: #FF4A98;  /* etc. */
}
```

`color_map["cN"]` resolves to a sentinel hex at render time (so the SVG embeds it deterministically); `process()` then rewrites those to `var(--cN)` references. If the variable is undefined in the host page, the browser uses the baked-in sentinel color — figures still render, just unthemed.

`python -m rdf palette --css` prints a complete starter block.

## Custom style

The default mplstyle targets web SVG (golden-ratio figsize, transparent background, no top/right spines, Computer Modern). For a paper/PDF style or anything else, pass your own:

```python
from pathlib import Path
RDF(style=Path("paper.mplstyle"))
```

## CLI

```
python -m rdf build <path>             render every @figure in a file or directory
python -m rdf inkscape <file.svg>      convert Inkscape SVG to RDF format
python -m rdf palette --css            CSS variables block
python -m rdf palette --inkscape       Inkscape palette (.gpl)
```

## Gallery

`scripts/` — every subdirectory is a built figure set.
