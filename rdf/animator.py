from __future__ import annotations

import re
from enum import Enum, auto


class AnimationStyle(Enum):
    NONE = auto()
    DRAW = auto()
    FADE = auto()
    PULSE = auto()


_CSS_TEMPLATE = """
<style>
    @keyframes drawLine {{ from {{ stroke-dashoffset: var(--len); }} to {{ stroke-dashoffset: 0; }} }}
    @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
    @keyframes pulse {{ 0% {{ opacity:.7; }} 50% {{ opacity:1; }} 100% {{ opacity:.7; }} }}

    .anim-draw {{ --len:1000; stroke-dasharray:var(--len); stroke-dashoffset:var(--len); animation:drawLine {dur}s ease-in-out {iter}; animation-delay:{delay}s; }}
    .anim-fade {{ opacity:0; animation:fadeIn {dur}s ease-in-out {iter}; animation-delay:{delay}s; }}
    .anim-pulse {{ animation:pulse {dur}s ease-in-out infinite; animation-delay:{delay}s; }}
</style>
"""

_JS_SNIPPET = """
<script>
    document.addEventListener('DOMContentLoaded',()=>{
        document.querySelectorAll('.anim-draw').forEach(p=>{
            if(p.getTotalLength){
                const L=p.getTotalLength();
                p.style.setProperty('--len',L);
                p.setAttribute('stroke-dasharray',L);
                p.setAttribute('stroke-dashoffset',L);
            }
        });
    });
</script>
"""


def add_animation(
    svg: str,
    *,
    style: AnimationStyle,
    duration: float = 2.0,
    delay: float = 0.0,
    loop: bool = True,
) -> str:
    """
    Return **new SVG text** with CSS + JS added for the requested animation.
    """

    if style is AnimationStyle.NONE:
        return svg

    css = _CSS_TEMPLATE.format(
        dur=duration, delay=delay, iter="infinite" if loop else "1"
    )

    idx = svg.find("<svg")
    if idx != -1:
        close = svg.find(">", idx)
        if close != -1:
            svg = svg[: close + 1] + "" + css + svg[close + 1 :]

    if style is AnimationStyle.DRAW:
        pattern = r"<path[^>]*d=\"[^\"]*\"[^>]*>"
        svg = re.sub(pattern, _add_cls("anim-draw"), svg)
    elif style is AnimationStyle.FADE:
        pattern = r"<(g|path|circle|rect|text)[^>]*>"
        svg = re.sub(pattern, _add_cls("anim-fade"), svg)
    elif style is AnimationStyle.PULSE:
        pattern = r"<path[^>]*d=\"[^\"]*\"[^>]*>"
        svg = re.sub(pattern, _add_cls("anim-pulse"), svg)

    # JS snippet to set actual path lengths (only needed for draw)
    if style is AnimationStyle.DRAW:
        svg = svg.replace("</svg>", _JS_SNIPPET + "</svg>")
    return svg


def _add_cls(cls_name: str):
    """Closure returning a replacement fn for re.sub."""

    def repl(match):
        el = match.group(0)
        if 'class="' in el:
            return el.replace('class="', f'class="{cls_name} ')
        front = el.split(" ", 1)[0]
        return el.replace(front, front + f' class="{cls_name}"', 1)

    return repl
