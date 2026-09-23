import numpy as np
from scipy.special import ellipe, ellipk


def loop_field(rho, z, a=1.0, z0=0.0):
    z = z - z0
    r = np.abs(rho)
    q = (a + r) ** 2 + z**2
    m = np.clip(4 * a * r / q, 0, 1 - 1e-12)
    K, E = ellipk(m), ellipe(m)
    d = (a - r) ** 2 + z**2
    bz = (K + (a**2 - r**2 - z**2) / d * E) / np.sqrt(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = np.where(r > 1e-9, z / (r * np.sqrt(q)) * (-K + (a**2 + r**2 + z**2) / d * E), 0.0)
    return np.sign(rho) * br, bz


def coil_field(rho, z, turns, a=1.0):
    fields = [loop_field(rho, z, a, t) for t in turns]
    return sum(f[0] for f in fields), sum(f[1] for f in fields)


def loop_flux(rho, z, a=1.0, z0=0.0):
    r = np.abs(rho) + 1e-12
    m = np.clip(4 * a * r / ((a + r) ** 2 + (z - z0) ** 2), 1e-12, 1 - 1e-12)
    return np.sqrt(a * r / m) * ((2 - m) * ellipk(m) - 2 * ellipe(m))


def coil_flux(rho, z, turns, a=1.0):
    return sum(loop_flux(rho, z, a, t) for t in turns)


def field_arrows(ax, pts, turns, color, a=1.0, flip=False):
    for r, z in pts:
        br, bz = coil_field(np.array(r), np.array(z), turns, a)
        u, v = (bz, br) if flip else (br, bz)
        n = np.hypot(u, v)
        (x, y) = (z, r) if flip else (r, z)
        ax.annotate("", xy=(x + 0.12 * u / n, y + 0.12 * v / n), xytext=(x, y), arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=12))


def contour_arrows(ax, cs, turns, color, flip=False, a=1.0):
    for segs in cs.allsegs:
        for seg in segs:
            if len(seg) < 20:
                continue
            x, y = seg[:, 0], seg[:, 1]
            closed = np.hypot(*(seg[0] - seg[-1])) < 0.05
            if closed or flip:
                picks = {int(np.argmax(x)), int(np.argmin(x))} | ({int(np.argmin(y))} if not flip else set())
            else:
                picks = {int(np.argmin(np.abs(y - (y.min() + 0.6)))), int(np.argmin(np.abs(y - (y.max() - 0.6))))}
            for i in picks:
                r, z = (y[i], x[i]) if flip else (x[i], y[i])
                br, bz = coil_field(np.array(r), np.array(z), turns, a)
                u, v = (bz, br) if flip else (br, bz)
                n = np.hypot(u, v)
                ax.annotate("", xy=(x[i] + 0.12 * u / n, y[i] + 0.12 * v / n), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=12))
