"""Turbulence diagnostics computed from pooled flux products, and Reynolds-
stress anisotropy (barycentric map, Banerjee et al. 2007).
"""
from dataclasses import dataclass

import numpy as np


def transport_efficiency(s_all: float, s_pos: float, s_neg: float) -> float:
    """Transport efficiency from pooled flux sums: S_all / S_pos if S_all > 0
    (the flux is upward-dominated), else S_all / S_neg. NaN if the chosen
    denominator is 0 (Salesky et al. 2017).
    """
    denom = s_pos if s_all > 0 else s_neg
    if denom == 0:
        return float("nan")
    return s_all / denom


@dataclass(frozen=True)
class Anisotropy:
    l1: float
    l2: float
    l3: float
    beta: float
    phi: float
    aniso_class: str | None


def _in_triangle(x: float, y: float, triangle) -> bool:
    (x1, y1), (x2, y2), (x3, y3) = triangle
    c1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    c2 = (x3 - x2) * (y - y2) - (y3 - y2) * (x - x2)
    c3 = (x1 - x3) * (y - y3) - (y1 - y3) * (x - x3)
    return (c1 <= 0 and c2 <= 0 and c3 <= 0) or (c1 >= 0 and c2 >= 0 and c3 >= 0)


def _classify_anisotropy_state(evals: np.ndarray, k: float) -> str:
    """Barycentric-map class for one eigenvalue triple (descending), edge
    width k: `1c`/`2c`/`3c` (pure), `prolate`/`oblate`/`ellipse` (transitional
    edges of width k around the pure vertices), else `mixed`.
    """
    l2, l3 = evals[1], evals[2]
    c2 = 2 * (l2 - l3)
    c3 = 3 * l3 + 1
    xb = c2 + 0.5 * c3
    yb = (np.sqrt(3) / 2) * c3

    v_1c, v_2c, v_3c = (0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)
    v_1c_2c_l, v_1c_2c_r = (0.5 * k, 0.0), (1 - 0.5 * k, 0.0)
    v_1c_3c_b, v_1c_3c_t = (0.25 * k, 0.25 * k * np.sqrt(3)), (0.5 - 0.25 * k, np.sqrt(3) * (0.5 - 0.25 * k))
    v_2c_3c_b, v_2c_3c_t = (1 - 0.25 * k, 0.25 * k * np.sqrt(3)), (0.5 + 0.25 * k, np.sqrt(3) * (0.5 - 0.25 * k))
    tip_1c, tip_2c = (0.5 * k, k * np.sqrt(3) / 6), (1 - 0.5 * k, k * np.sqrt(3) / 6)
    tip_3c = (0.5, np.sqrt(3) * (0.5 - k / 3))

    triangles = {
        "1c": [(v_1c, v_1c_3c_b, v_1c_2c_l), (v_1c_3c_b, tip_1c, v_1c_2c_l)],
        "2c": [(v_2c, v_1c_2c_r, v_2c_3c_b), (v_1c_2c_r, v_2c_3c_b, tip_2c)],
        "3c": [(v_2c_3c_t, v_1c_3c_t, v_3c), (v_2c_3c_t, v_1c_3c_t, tip_3c)],
        "prolate": [(v_1c_3c_b, v_1c_3c_t, tip_3c), (v_1c_3c_b, tip_3c, tip_1c)],
        "oblate": [(v_2c_3c_b, v_2c_3c_t, tip_3c), (v_2c_3c_b, tip_3c, tip_2c)],
        "ellipse": [(v_1c_2c_l, v_1c_2c_r, tip_2c), (v_1c_2c_l, tip_2c, tip_1c)],
    }
    for name, tris in triangles.items():
        if any(_in_triangle(xb, yb, tri) for tri in tris):
            return name
    return "mixed"


def _anisotropy_angles(evecs: np.ndarray) -> tuple[float, float]:
    """Inclination (beta) and orientation (phi), degrees, of the anisotropy
    eigenvectors in a streamwise-aligned frame (Gucci et al. 2025). `evecs`
    columns are eigenvectors in descending eigenvalue order; v1 is forced to
    positive streamwise (x), v3 to positive vertical (z).
    """
    v1, v3 = evecs[:, 0], evecs[:, 2]
    sign_v1 = -1.0 if v1[0] < 0 else 1.0
    v1x, v1y = v1[0] * sign_v1, v1[1] * sign_v1
    sign_v3 = -1.0 if v3[2] < 0 else 1.0
    v3x = v3[0] * sign_v3

    beta = np.degrees(np.arcsin(np.clip(v3x, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(-v1y, v1x))
    return float(beta), float(phi)


def anisotropy(var_u: float, var_v: float, var_w: float, cov_uv: float, cov_uw: float,
               cov_vw: float, tke: float, k: float = 2.0 / 3.0) -> Anisotropy:
    """Reynolds-stress anisotropy for one (slot, boom, variant): the tensor
    b_ij = R_ij/(2*tke) - delta_ij/3, its eigenvalues (descending) and
    eigenvectors, the barycentric-map class and the inclination/orientation
    angles. Every field is NaN (class None) if tke is 0 or any input isn't
    finite.
    """
    inputs = (var_u, var_v, var_w, cov_uv, cov_uw, cov_vw, tke)
    if tke == 0 or not all(np.isfinite(x) for x in inputs):
        return Anisotropy(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), None)

    r = np.array([
        [var_u, cov_uv, cov_uw],
        [cov_uv, var_v, cov_vw],
        [cov_uw, cov_vw, var_w],
    ])
    b = r / (2 * tke) - np.eye(3) / 3.0

    evals, evecs = np.linalg.eigh(b)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    aniso_class = _classify_anisotropy_state(evals, k)
    beta, phi = _anisotropy_angles(evecs)
    return Anisotropy(float(evals[0]), float(evals[1]), float(evals[2]), beta, phi, aniso_class)
