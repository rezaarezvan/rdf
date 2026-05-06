from rdf.rdf import RDF
from rdf import inkscape
from rdf.theme import ColorTheme


def tag(artist, gid: str) -> None:
    """Tag a matplotlib artist so the resulting SVG element has id=gid."""
    artist.set_gid(gid)


__all__ = ["RDF", "ColorTheme", "inkscape", "tag"]
