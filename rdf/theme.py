import json
from pathlib import Path
from dataclasses import dataclass, field

Palette = dict[str, str]

AXIS_SENTINEL = "#FE0102"
GRID_SENTINEL = "#FE0304"
BLACK_SENTINEL = "#FE0506"

AXIS_INK = ("#2a241c", "#e6dfd2")
GRID_INK = ("rgba(42, 36, 28, 0.14)", "rgba(230, 223, 210, 0.16)")


@dataclass(slots=True)
class ColorTheme:
    """Light palette, dark palette, and fixed 8-colour base mapping."""

    light: Palette
    dark: Palette
    base: Palette = field(repr=False)
    # (light, dark) chrome; not part of the indexed palette.
    axis: tuple[str, str] = AXIS_INK
    grid: tuple[str, str] = GRID_INK

    @classmethod
    def default(cls) -> "ColorTheme":
        return cls(
            light={
                "c1": "#D55E00",  # vermillion
                "c2": "#009E73",  # bluish green
                "c3": "#56B4E9",  # sky blue
                "c4": "#CC79A7",  # reddish purple
                "c5": "#E69F00",  # orange
                "c6": "#F0E442",  # yellow
                "c7": "#999999",  # neutral gray (de-emphasis slot)
                "c8": "#0072B2",  # blue
                "primary": "#D55E00",
                "secondary": "#009E73",
                "tertiary": "#56B4E9",
                "quaternary": "#CC79A7",
                "quinary": "#E69F00",
                "senary": "#F0E442",
                "septenary": "#999999",
                "octonary": "#0072B2",
                "black": "#000000",
                "gray": "#999999",
                "grey": "#999999",
                "white": "#ffffff",
            },
            dark={
                "c1": "#D55E00",
                "c2": "#009E73",
                "c3": "#56B4E9",
                "c4": "#CC79A7",
                "c5": "#E69F00",
                "c6": "#F0E442",
                "c7": "#999999",
                "c8": "#0072B2",
                "primary": "#D55E00",
                "secondary": "#009E73",
                "tertiary": "#56B4E9",
                "quaternary": "#CC79A7",
                "quinary": "#E69F00",
                "senary": "#F0E442",
                "septenary": "#999999",
                "octonary": "#0072B2",
                "black": "#ECECE6",
                "gray": "#999999",
                "grey": "#999999",
                "white": "#1E1F1C",
            },
            base={
                "c1": "#FF0001",
                "c2": "#00FF02",
                "c3": "#0000FF",
                "c4": "#FF00FF",
                "c5": "#00FFFF",
                "c6": "#FFFF00",
                "c7": "#FF8000",
                "c8": "#8000FF",
            },
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).expanduser().write_text(
            json.dumps(
                {
                    "light": self.light,
                    "dark": self.dark,
                    "base": self.base,
                    "axis": list(self.axis),
                    "grid": list(self.grid),
                },
                indent=2,
            ),
            "utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ColorTheme":
        data = json.loads(Path(path).expanduser().read_text("utf-8"))
        return cls(
            light=data["light"],
            dark=data["dark"],
            base=data["base"],
            axis=tuple(data.get("axis", AXIS_INK)),
            grid=tuple(data.get("grid", GRID_INK)),
        )
