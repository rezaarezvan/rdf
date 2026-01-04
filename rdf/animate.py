from __future__ import annotations

import re
from enum import Enum, auto


class AnimationType(Enum):
    NONE = auto()
    DRAW = auto()
    FADE = auto()
    PULSE = auto()


CSS = """<style>
@keyframes drawLine {{ from {{ stroke-dashoffset:var(--len); }} to {{ stroke-dashoffset:0; }} }}
@keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
@keyframes pulse {{ 0%{{ opacity:.7; }} 50%{{ opacity:1; }} 100%{{ opacity:.7; }} }}
.anim-draw {{ --len:1000; stroke-dasharray:var(--len); stroke-dashoffset:var(--len); animation:drawLine {dur}s ease-in-out {iter}; animation-delay:{delay}s; }}
.anim-fade {{ opacity:0; animation:fadeIn {dur}s ease-in-out {iter}; animation-delay:{delay}s; }}
.anim-pulse {{ animation:pulse {dur}s ease-in-out infinite; animation-delay:{delay}s; }}
</style>"""

JS = """<script>
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.anim-draw').forEach(p=>{
    if(p.getTotalLength){ const L=p.getTotalLength(); p.style.setProperty('--len',L); p.setAttribute('stroke-dasharray',L); p.setAttribute('stroke-dashoffset',L); }
  });
});
</script>"""


def _add_class(cls: str):
    def repl(m):
        el = m.group(0)
        return (
            el.replace('class="', f'class="{cls} ')
            if 'class="' in el
            else el.replace(
                el.split(" ", 1)[0], el.split(" ", 1)[0] + f' class="{cls}"', 1
            )
        )

    return repl


def animate(
    svg: str,
    style: AnimationType,
    *,
    duration: float = 2.0,
    delay: float = 0.0,
    loop: bool = True,
) -> str:
    """Add animation CSS/JS to SVG."""
    if style is AnimationType.NONE:
        return svg
    css = CSS.format(dur=duration, delay=delay, iter="infinite" if loop else "1")
    # Inject CSS after opening <svg> tag
    if (idx := svg.find("<svg")) != -1 and (close := svg.find(">", idx)) != -1:
        svg = svg[: close + 1] + css + svg[close + 1 :]
    # Apply animation classes
    if style is AnimationType.DRAW:
        svg = re.sub(r'<path[^>]*d="[^"]*"[^>]*>', _add_class("anim-draw"), svg)
        svg = svg.replace("</svg>", JS + "</svg>")
    elif style is AnimationType.FADE:
        svg = re.sub(r"<(g|path|circle|rect|text)[^>]*>", _add_class("anim-fade"), svg)
    elif style is AnimationType.PULSE:
        svg = re.sub(r'<path[^>]*d="[^"]*"[^>]*>', _add_class("anim-pulse"), svg)
    return svg


# Backwards compatibility
AnimationStyle = AnimationType
add_animation = animate
