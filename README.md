# rdf
rezvan dynamic figures - a tool to create dynamic SVG figures for my blog

Supports 3D and subplots.

# Installation
```bash
python3 -m pip install -e .
```

# Usage
```python
from rdf import RDF

plotter = RDF()

# Automatically saves to `result/{SCRIPT_NAME}/example.svg`
svg_content = plotter.create_themed_plot(
    save_name="example", plot_func=plot_example
)
```
