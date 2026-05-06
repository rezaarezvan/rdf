from rdf.rdf import RDF
from rdf import inkscape
from rdf.theme import ColorTheme
from rdf.registry import figure, build, registered, reset


def tag(artist, gid: str) -> None:
    """Tag a matplotlib artist so the resulting SVG element has id=gid."""
    artist.set_gid(gid)


__all__ = [
    "RDF",
    "ColorTheme",
    "inkscape",
    "tag",
    "figure",
    "build",
    "registered",
    "reset",
]
