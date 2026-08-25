#!/usr/bin/env python3
"""RDF CLI - python -m rdf <command> [args]"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from rdf import inkscape
from rdf.svg import build_css
from rdf.theme import ColorTheme
from rdf import registry


def usage():
    print("""usage: python -m rdf <command> [args]

commands:
  build <path>                          Render every @figure in a file or directory
  inkscape <file.svg> [-o output.svg]   Convert Inkscape SVG to RDF format
  inkscape <*.svg> --batch              Batch convert multiple SVGs
  palette --css                         Export palette as CSS variables
  palette --inkscape                    Export palette for Inkscape
  palette --json <file.json>            Export palette as JSON
""")


def cmd_inkscape(args: list[str]):
    assert args, "usage: python -m rdf inkscape <file.svg> [-o output.svg]"
    batch = "--batch" in args
    if batch:
        args = [a for a in args if a != "--batch"]
    out_path = None
    if "-o" in args:
        idx = args.index("-o")
        out_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2 :]
    paths = [Path(a) for a in args]
    assert all(p.exists() for p in paths), "file not found"
    if batch or len(paths) > 1:
        inkscape.convert_batch(paths)
    else:
        inkscape.convert(paths[0], out_path)


def cmd_palette(args: list[str]):
    theme = ColorTheme.default()
    if "--css" in args:
        print(build_css(theme))
    elif "--inkscape" in args:
        # Export as Inkscape palette format
        print("GIMP Palette")
        print("Name: RDF")
        print("#")
        for name, color in {**theme.light, **theme.base}.items():
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            print(f"{r:3d} {g:3d} {b:3d}  {name}")
    elif "--json" in args:
        idx = args.index("--json")
        out = Path(args[idx + 1]) if len(args) > idx + 1 else Path("palette.json")
        theme.to_json(out)
        print(f"saved: {out}")
    else:
        usage()


def cmd_build(args: list[str]):
    from rdf import RDF

    assert args, "usage: python -m rdf build <path>"
    target = Path(args[0])
    assert target.exists(), f"not found: {target}"
    files = (
        sorted(p for p in target.glob("*.py") if not p.name.startswith("_"))
        if target.is_dir()
        else [target]
    )
    saved_argv, saved_path = sys.argv, list(sys.path)
    for f in files:
        registry.reset()
        sys.argv = [str(f)]
        sys.path.insert(0, str(f.parent.resolve()))
        try:
            runpy.run_path(str(f), run_name="__rdf_build__")
        finally:
            sys.argv, sys.path = saved_argv, list(saved_path)
        if registry.registered():
            registry.build(RDF(name=f.stem))


def main():
    match sys.argv[1:]:
        case ["build", *args]:
            cmd_build(args)
        case ["inkscape", *args]:
            cmd_inkscape(args)
        case ["palette", *args]:
            cmd_palette(args)
        case _:
            usage()


if __name__ == "__main__":
    main()
