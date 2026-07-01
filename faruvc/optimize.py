"""Minimum-lamp placement optimiser.

Given a room, the B1 photometry, a target room-average fluence rate, an exposure
standard, and a placement mode, choose the FEWEST candidate lamps such that:

    average fluence over the room volume  >=  target
    skin-plane irradiance everywhere      <=  skin cap
    eye worst-case irradiance everywhere  <=  eye cap

Every quantity is linear in the binary "is this candidate used?" variables, so this
is an integer linear program (solved with PuLP/CBC). A second, optional pass dims all
selected lamps uniformly to recover headroom / improve the exposure margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pulp

from .candidates import generate_candidates
from .field import (LampInstance, exposure_field, fluence_field,
                    lamp_exposure_terms)
from .geometry import Room
from .photometry import Photometry
from .regs import ExposureLimits, Standard, limits_for_spectrum

_Z = np.array([0.0, 0.0, 1.0])

# Stage-2 goal presets -> coverage weight w in the blended objective:
#   throughput : pure average fluence (max germicidal delivery; best for most rooms)
#   balanced   : mostly average, with some weight on the worst-covered point
#   coverage   : pure max-min (even coverage; best for corridors / L-shaped rooms)
_GOAL_W = {"throughput": 0.0, "balanced": 0.4, "coverage": 1.0}


@dataclass
class OptimizeResult:
    status: str                       # "optimal", "infeasible", ...
    n_lamps: int
    lamps: list[dict] = field(default_factory=list)   # {x,y,z,aim,kind}
    avg_fluence: float = 0.0
    min_fluence: float = 0.0
    max_skin: float = 0.0
    max_eye: float = 0.0
    skin_cap: float = 0.0
    eye_cap: float = 0.0
    target_fluence: float = 0.0
    max_util: float = 0.0             # worst exposure as a fraction of its cap (0..1)
    goal: str = ""                    # stage-2 goal preset actually used
    message: str = ""


def optimize(
    room: Room,
    photometry: Photometry,
    target_fluence: float,
    standard: Standard,
    mode: str = "downlight",
    *,
    goal: str = "throughput",
    margin_tax: float = 0.2,
    spectrum_csv: str | None = None,
    fluence_spacing: float = 0.4,
    plane_spacing: float = 0.4,
    coverage_top: float = 2.0,
    occupant_height: float | None = None,
    n_azimuths: int = 16,
    cap_margin: float = 0.95,
    max_coverage_points: int = 160,
    time_limit: float = 20.0,
    solver_msg: bool = False,
    **candidate_kwargs,
) -> OptimizeResult:
    if goal not in _GOAL_W:
        raise ValueError(f"unknown goal {goal!r}; expected one of {sorted(_GOAL_W)}")
    w = _GOAL_W[goal]

    limits = _limits(standard, spectrum_csv)
    eye_cap, skin_cap = limits.eye_uw, limits.skin_uw
    # Constrain to a margin inside the true caps so azimuth/grid discretisation error
    # (the realised field is evaluated on a finer grid) cannot push us over the limit.
    eye_lim, skin_lim = eye_cap * cap_margin, skin_cap * cap_margin

    # Per-standard assessment geometry: which calc mode for eye/skin, and plane height.
    zone = standard.zone
    eye_mode, skin_mode = zone.eye_mode, zone.skin_mode
    height = occupant_height if occupant_height is not None else zone.height_m

    candidates = generate_candidates(room, photometry, mode=mode, **candidate_kwargs)
    if not candidates:
        return OptimizeResult("infeasible", 0, message="no candidate positions",
                              eye_cap=eye_cap, skin_cap=skin_cap,
                              target_fluence=target_fluence)

    vol = room.volume_grid(spacing=fluence_spacing)
    plane = room.plane_grid(z=height, spacing=plane_spacing)

    nC = len(candidates)
    # Per-candidate contributions (linear coefficients):
    F = np.empty((nC, len(vol)))              # fluence per volume point
    # Eye/skin exposure "terms": (nC, n_plane, n_samples) per the standard's calc mode.
    skin_t = [lamp_exposure_terms(candidates[c], plane, skin_mode, n_azimuths) for c in range(nC)]
    eye_t = [lamp_exposure_terms(candidates[c], plane, eye_mode, n_azimuths) for c in range(nC)]
    skin_t = np.stack(skin_t)                  # (nC, nP, Ks)
    eye_t = np.stack(eye_t)                    # (nC, nP, Ke)
    for c, lamp in enumerate(candidates):
        F[c] = lamp.fluence(vol)
    # Evaluation zone: the occupied band (floor up to coverage_top). BOTH the germicidal
    # average AND the worst-spot minimum are measured here, not over the full volume.
    # Points up near the ceiling sit centimetres from the lamps, where the 1/r^2 point-
    # source model diverges (a grid point 2 cm from a lamp reads ~13000 uW/cm2). Averaging
    # those in lets a layout inflate its "average" by parking a lamp next to a grid point
    # instead of actually lighting the room -- which is exactly what the max-average goal
    # was doing. Restricting to the occupied band drops the singular near-field (lamps
    # mount above it) and matches the zone people/pathogens actually occupy.
    cov_top = min(coverage_top, room.height - 0.1)
    cov = np.where(vol[:, 2] <= cov_top)[0]
    if len(cov) == 0:
        cov = np.arange(len(vol))
    fvec = F[:, cov].mean(axis=1)             # mean fluence per candidate (occupied zone)
    # Subsample coverage points used in the (heavier) max-min stage to keep it fast.
    if len(cov) > max_coverage_points:
        cov_stage2 = cov[np.linspace(0, len(cov) - 1, max_coverage_points).astype(int)]
    else:
        cov_stage2 = cov

    def add_exposure_caps(prob, x):
        for terms, lim in ((skin_t, skin_lim), (eye_t, eye_lim)):
            nP, K = terms.shape[1], terms.shape[2]
            for j in range(nP):
                for k in range(K):
                    col = terms[:, j, k]
                    # Skip constraints that can never bind: if every candidate on still
                    # can't reach the cap (Σ col ≤ lim), no subset can either. Exact and
                    # safe, and removes the many far-from-lamp points — big speedup.
                    if col.sum() <= lim:
                        continue
                    prob += pulp.lpSum(col[c] * x[c] for c in range(nC) if col[c] > 0) <= lim

    def add_utilization_envelope(prob, x, u_max):
        # u_max >= realised exposure / true cap, over the same near-cap samples used for
        # the hard caps. Samples that can't approach the cap even with every candidate on
        # are skipped (they can never set the worst utilisation), so the tax is inert when
        # nothing runs near the limit and only bites hot / clustered layouts.
        for terms, cap, lim in ((skin_t, skin_cap, skin_lim), (eye_t, eye_cap, eye_lim)):
            if cap <= 0:
                continue
            nP, K = terms.shape[1], terms.shape[2]
            for j in range(nP):
                for k in range(K):
                    col = terms[:, j, k]
                    if col.sum() <= lim:
                        continue
                    prob += pulp.lpSum((col[c] / cap) * x[c]
                                       for c in range(nC) if col[c] > 0) <= u_max

    # --- Stage 1: minimise lamp count -------------------------------------
    p1 = pulp.LpProblem("min_lamps", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{c}", cat="Binary") for c in range(nC)]
    p1 += pulp.lpSum(x)
    p1 += pulp.lpSum(fvec[c] * x[c] for c in range(nC)) >= target_fluence
    add_exposure_caps(p1, x)
    p1.solve(pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0, timeLimit=time_limit))
    status = pulp.LpStatus[p1.status].lower()
    if status != "optimal":
        return OptimizeResult(
            status="infeasible" if status == "infeasible" else status,
            n_lamps=0, eye_cap=eye_cap, skin_cap=skin_cap,
            target_fluence=target_fluence,
            message=_infeasibility_hint(candidates, fvec, target_fluence,
                                        skin_cap, eye_cap, height,
                                        skin_mode, eye_mode),
        )
    n_star = int(round(sum(v.value() for v in x)))

    # --- Stage 2: among min-count layouts, choose the best arrangement -----
    # Blended linear objective, all terms normalised to ~O(1):
    #   maximise  (1-w)*(avg/target) + w*(min/target) - margin_tax*u_max
    # w (from the goal preset) trades raw germicidal throughput (avg fluence, which is
    # separable and blind to lamp spacing) against even coverage (the worst-covered
    # point). u_max is the worst exposure utilisation, so the tax steers away from
    # layouts that run hot / cluster lamps even when they stay legal.
    p2 = pulp.LpProblem("stage2", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y{c}", cat="Binary") for c in range(nC)]
    p2 += pulp.lpSum(y) == n_star
    p2 += pulp.lpSum(fvec[c] * y[c] for c in range(nC)) >= target_fluence
    add_exposure_caps(p2, y)

    tgt = target_fluence if target_fluence > 1e-9 else 1e-9
    obj = (1.0 - w) * (pulp.lpSum(fvec[c] * y[c] for c in range(nC)) / tgt)
    if w > 0:
        t = pulp.LpVariable("t", lowBound=0)
        for p in cov_stage2:
            col = F[:, p]
            p2 += pulp.lpSum(col[c] * y[c] for c in range(nC) if col[c] > 0) >= t
        obj += w * (t / tgt)
    if margin_tax > 0:
        u_max = pulp.LpVariable("u_max", lowBound=0)
        add_utilization_envelope(p2, y, u_max)
        obj += -margin_tax * u_max
    p2.setObjective(obj)
    p2.solve(pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0, timeLimit=time_limit))

    src = y if pulp.LpStatus[p2.status].lower() == "optimal" else x
    chosen = [c for c in range(nC) if src[c].value() and src[c].value() > 0.5]
    sel = [candidates[c] for c in chosen]

    # Realised metrics, evaluated with the standard's own calc modes.
    f_eval = fluence_field(sel, vol)
    s_eval = exposure_field(sel, plane, skin_mode, n_az=max(16, n_azimuths))
    e_eval = exposure_field(sel, plane, eye_mode, n_az=max(16, n_azimuths))
    max_util = max(
        (float(s_eval.max()) / skin_cap) if (len(s_eval) and skin_cap > 0) else 0.0,
        (float(e_eval.max()) / eye_cap) if (len(e_eval) and eye_cap > 0) else 0.0,
    )

    return OptimizeResult(
        status="optimal",
        n_lamps=len(sel),
        lamps=[_lamp_dict(l) for l in sel],
        avg_fluence=float(f_eval[cov].mean()) if len(cov) else float(f_eval.mean()),
        min_fluence=float(f_eval[cov].min()) if len(cov) else 0.0,
        max_skin=float(s_eval.max()) if len(s_eval) else 0.0,
        max_eye=float(e_eval.max()) if len(e_eval) else 0.0,
        skin_cap=skin_cap, eye_cap=eye_cap, target_fluence=target_fluence,
        max_util=float(max_util), goal=goal,
        message=(f"{len(sel)} lamp(s) | {standard.label} | goal={goal} | "
                 f"eye={eye_mode}, skin={skin_mode} @ {height:.2f} m | {nC} candidates"),
    )


# --- helpers --------------------------------------------------------------
def _limits(standard: Standard, spectrum_csv: str | None) -> ExposureLimits:
    csv = spectrum_csv or "data/lamp_data/ushio_b1.csv"
    return limits_for_spectrum(csv, standard)


def _lamp_dict(lamp: LampInstance) -> dict:
    return {
        "x": float(lamp.pos[0]), "y": float(lamp.pos[1]), "z": float(lamp.pos[2]),
        "aim": [float(a) for a in lamp.aim],
        "kind": "downlight" if abs(lamp.aim[2] + 1.0) < 1e-6 else "tilted",
    }


def _infeasibility_hint(candidates, fvec, target, skin_cap, eye_cap, occ_h,
                        skin_mode, eye_mode) -> str:
    """Explain why no layout works: fluence unreachable vs single-lamp over-exposure."""
    if fvec.sum() < target:
        return (f"target avg fluence {target:.2f} unreachable: all "
                f"{len(candidates)} candidates together give only {fvec.sum():.2f} uW/cm2")
    # Exact worst case for a single lamp: its own skin/eye field at occupant height
    # directly beneath it (where a downlight peaks), under the standard's calc modes.
    best_skin = float("inf")
    best_eye = float("inf")
    for lamp in candidates:
        nadir = np.array([[lamp.pos[0], lamp.pos[1], occ_h]])
        best_skin = min(best_skin, float(exposure_field([lamp], nadir, skin_mode)[0]))
        best_eye = min(best_eye, float(exposure_field([lamp], nadir, eye_mode)[0]))
    if best_skin > skin_cap:
        return (f"even a single lamp exceeds the skin cap directly beneath it "
                f"({best_skin:.1f} > {skin_cap:.1f} uW/cm2 at {occ_h:.1f} m). "
                f"Raise the ceiling/mount, dim the lamps, or relax the standard.")
    if best_eye > eye_cap:
        return (f"even a single lamp exceeds the eye cap ({best_eye:.2f} > "
                f"{eye_cap:.2f} uW/cm2). Raise/retilt the mount, dim, or relax the standard.")
    return ("no arrangement reaches the target without exceeding an exposure cap; "
            "try dimming, a higher mount, or a less strict standard")
