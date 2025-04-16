from __future__ import annotations

import json

from pathlib import Path
from typing import Mapping, Union
from dataclasses import dataclass, field

Palette = Mapping[str, str]


@dataclass(slots=True)
class ColorTheme:
    """
    Light palette, dark palette, and fixed 8‑colour *base* mapping.
    """

    light: Palette
    dark: Palette
    base: Palette = field(repr=False)

    @classmethod
    def default(cls) -> "ColorTheme":
        return cls(
            light={
                "primary": "#d62728",
                "secondary": "#2ca02c",
                "tertiary": "#1f77b4",
                "quaternary": "#9467bd",
                "quinary": "#8c564b",
                "senary": "#e377c2",
                "septenary": "#7f7f7f",
                "octonary": "#bcbd22",
            },
            dark={
                "primary": "#FF4A98",
                "secondary": "#0AFAFA",
                "tertiary": "#7F83FF",
                "quaternary": "#B4A0FF",
                "quinary": "#FFB86B",
                "senary": "#FF79C6",
                "septenary": "#CCCCCC",
                "octonary": "#E6DB74",
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

    # Serialisation helpers (future convenience) -----------------------
    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).expanduser().write_text(
            json.dumps(
                {"light": self.light, "dark": self.dark, "base": self.base}, indent=2
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ColorTheme":
        data = json.loads(Path(path).expanduser().read_text("utf-8"))
        return cls(light=data["light"], dark=data["dark"], base=data["base"])
