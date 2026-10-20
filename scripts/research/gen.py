import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SCALE = 0.3
RADIUS = {"C": 0.12, "N": 0.12, "O": 0.12, "H": 0.07}
FILL = {"C": "c7", "N": "c8", "O": "c1", "H": "white"}


def splice(path, name, body):
    text = path.read_text()
    pat = re.compile(rf"(% BEGIN {name}\n).*?(% END {name}\n)", re.S)
    path.write_text(pat.sub(lambda m: m[1] + body + m[2], text))


def molecule(xyz, el, bonds, rot, center):
    p = (np.asarray(xyz) - center) @ rot * SCALE
    items = [(p[a, 2], "atom", a) for a in range(len(el))]
    items += [((p[a, 2] + p[b, 2]) / 2 - 0.05, "bond", (a, b)) for a, b in bonds]
    out = []
    for _, kind, i in sorted(items, key=lambda t: t[0]):
        if kind == "bond":
            (ax, ay), (bx, by) = p[i[0], :2], p[i[1], :2]
            out.append(rf"\draw[bond] ({ax:.3f},{ay:.3f}) -- ({bx:.3f},{by:.3f});")
        else:
            x, y, z = p[i]
            r = RADIUS[el[i]] * (1 + 0.06 * z)
            out.append(rf"\fill[{FILL[el[i]]}] ({x:.3f},{y:.3f}) circle ({r:.3f}); \draw[atomline] ({x:.3f},{y:.3f}) circle ({r:.3f});")
    return "\n".join(out)


def semla_ito():
    d = json.loads((HERE / "data/ala2.json").read_text())
    el, bonds, frames = d["elements"], d["bonds"], d["frames"]
    ref = np.asarray(frames["x_t"])
    center = ref.mean(0)
    _, _, vt = np.linalg.svd(ref - center)
    rot = vt.T
    body = ""
    for name, key in (("molxt", "x_t"), ("molxttau", "x_t_tau"), ("molxhat", "x_hat")):
        body += rf"\newcommand{{\{name}}}{{%" + "\n" + molecule(frames[key], el, bonds, rot, center) + "\n}\n"
    noise = np.random.default_rng(0).standard_normal((len(el), 3)) * 1.6
    body += r"\newcommand{\molnoise}{%" + "\n" + molecule(noise, el, [], np.eye(3), np.zeros(3)) + "\n}\n"
    splice(HERE / "semla_ito.tikz", "molecules", body)


def mnist():
    px = np.asarray(json.loads((HERE / "data/mnist5.json").read_text())["pixels"])
    s = 0.055
    cells = [
        rf"\fill[black, fill opacity={v:.2f}] ({c * s:.3f},{-r * s:.3f}) rectangle ++({s + 0.004:.3f},{-s - 0.004:.3f});"
        for r, row in enumerate(px) for c, v in enumerate(row) if v > 0.05
    ]
    body = rf"\fill[black, fill opacity=0.06] (0,0) rectangle ({28 * s:.3f},{-28 * s:.3f});" + "\n" + "\n".join(cells) + "\n"
    splice(HERE / "claudeslens.tikz", "mnist", body)


if __name__ == "__main__":
    semla_ito()
    mnist()
