"""Minimum-lamp placement optimiser.

Given a room, the B1 photometry, a target room-average fluence rate, an exposure
standard, and a placement mode, choose the FEWEST candidate lamps such that:

    average fluence over the room volume  >=  target
    skin-plane irradiance everywhere      <=  skin cap
    eye worst-case irradiance everywhere  <=  eye cap

Every quantity is linear in the binary "is this candidate used?" variables, so this is
an integer linear program. Constraints are assembled as vectorised numpy matrices and
solved with HiGHS via ``scipy.optimize.milp`` -- far faster than building the model term
by term, and it keeps the field maths (the cheap part) out of the solver's way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

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
    fluence_spacing: float = 0.25,
    plane_spacing: float = 0.4,
    coverage_top: float = 2.0,
    fluence_cap: float = 20.0,
    occupant_height: float | None = None,
    n_azimuths: int = 16,
    cap_margin: float = 0.95,
    max_coverage_points: int = 400,
    time_limit: float = 15.0,
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

    # Capped occupied-zone average per candidate. Each point's fluence is clipped at
    # fluence_cap: a saturated spot gains nothing more (well-mixed air can't all funnel
    # through one hotspot), and the clip also bounds any residual near-field 1/r^2 spike.
    # We clip PER-LAMP -- min(cap, F[c,p]) -- so the average stays a separable linear
    # function of the lamp selection and the ILP stays small/fast. That's exact for the
    # dominant single-lamp case and over-counts only where two beams both exceed the cap
    # at the same occupied-zone point (rare, since lamps mount above the zone). The
    # REPORTED avg_fluence below uses the exact total cap, so the displayed number is honest.
    fvec = np.minimum(F[:, cov], fluence_cap).mean(axis=1)
    # Subsample coverage points used in the (heavier) max-min stage to keep it fast.
    if len(cov) > max_coverage_points:
        cov_stage2 = cov[np.linspace(0, len(cov) - 1, max_coverage_points).astype(int)]
    else:
        cov_stage2 = cov

    # --- Exposure/utilisation constraint matrices (vectorised, built in one shot) ---
    # Each exposure sample is a row  Σ_c A[s,c]·x_c ≤ lim.  skin_t/eye_t are (nC, nP, K);
    # reshape to (samples, nC).  Rows that can't bind even with every candidate lit
    # (row-sum ≤ lim) are dropped -- exact, and a big size cut.  This replaces the old
    # term-by-term PuLP build (which was ~90% of the runtime; the field maths is ~0.1 s).
    exp_mats, exp_ubs, util_mats = [], [], []
    for terms, cap, lim in ((skin_t, skin_cap, skin_lim), (eye_t, eye_cap, eye_lim)):
        M = terms.reshape(nC, -1).T                    # (samples, nC)
        keep = M.sum(axis=1) > lim
        if keep.any():
            Mk = M[keep]
            exp_mats.append(Mk)
            exp_ubs.append(np.full(Mk.shape[0], lim))
            if cap > 0:
                util_mats.append(Mk / cap)             # utilisation = exposure / true cap
    A_exp = np.vstack(exp_mats) if exp_mats else np.zeros((0, nC))
    ub_exp = np.concatenate(exp_ubs) if exp_ubs else np.zeros(0)
    A_util = np.vstack(util_mats) if util_mats else np.zeros((0, nC))
    # The margin tax only needs u_max = the WORST utilisation, which for any layout sits at
    # some selected lamp's own peak. Keeping each candidate's top few peak samples
    # reproduces u_max exactly while collapsing thousands of rows -- all tied to the single
    # u_max var, which cripples branch-and-bound -- down to ~nC. ~8x faster, same answer.
    if A_util.shape[0] > 0:
        peak_rows = np.unique(np.argsort(-A_util, axis=0)[:2].ravel())
        A_util = A_util[peak_rows]
    # A 1% optimality gap is far below the modelling noise here and lets HiGHS stop as
    # soon as it has a provably near-optimal layout instead of grinding to close the gap.
    opts = {"time_limit": time_limit, "mip_rel_gap": 0.01, "disp": bool(solver_msg)}

    # --- Stage 1: minimise lamp count -------------------------------------
    cons1 = [LinearConstraint(fvec, target_fluence, np.inf)]
    if A_exp.shape[0]:
        cons1.append(LinearConstraint(sparse.csr_matrix(A_exp), -np.inf, ub_exp))
    res1 = milp(c=np.ones(nC), constraints=cons1, integrality=np.ones(nC),
                bounds=Bounds(0, 1), options=opts)
    if res1.x is None:
        return OptimizeResult(
            status="infeasible", n_lamps=0, eye_cap=eye_cap, skin_cap=skin_cap,
            target_fluence=target_fluence,
            message=_infeasibility_hint(candidates, fvec, target_fluence,
                                        skin_cap, eye_cap, height, skin_mode, eye_mode),
        )
    n_star = int(round(res1.x.sum()))

    # --- Stage 2: among min-count layouts, choose the best arrangement -----
    # Blended objective (milp minimises, so we pass the negative):
    #   maximise  (1-w)*(avg/target) + w*(min/target) - margin_tax*u_max
    # w (goal preset) trades raw germicidal throughput (avg fluence -- separable, blind to
    # spacing) against even coverage (the worst-covered point t). u_max is the worst
    # exposure utilisation, so the tax steers off layouts that run hot / cluster lamps.
    # Variables: y (nC binaries) then t (min fluence) then u_max (worst utilisation).
    tgt = target_fluence if target_fluence > 1e-9 else 1e-9
    nv = nC + 2
    t_idx, u_idx = nC, nC + 1

    def _rows(A, extra):
        """Embed an (m, nC) coefficient block into the (m, nv) variable layout."""
        R = np.zeros((A.shape[0], nv))
        R[:, :nC] = A
        for col, val in extra:
            R[:, col] = val
        return sparse.csr_matrix(R)

    c2 = np.zeros(nv)
    c2[:nC] = -(1.0 - w) / tgt * fvec          # throughput term
    c2[t_idx] = -(w / tgt)                       # even-coverage term (0 when w==0)
    c2[u_idx] = margin_tax                       # exposure-margin tax (0 when off)

    cons2 = []
    lamp_count = np.zeros(nv); lamp_count[:nC] = 1.0
    cons2.append(LinearConstraint(lamp_count, n_star, n_star))     # exactly n_star lamps
    avg_row = np.zeros(nv); avg_row[:nC] = fvec
    cons2.append(LinearConstraint(avg_row, target_fluence, np.inf))  # hit the target
    if A_exp.shape[0]:
        cons2.append(LinearConstraint(_rows(A_exp, []), -np.inf, ub_exp))
    if w > 0 and len(cov_stage2):                                   # t ≤ fluence at each cov pt
        cons2.append(LinearConstraint(_rows(F[:, cov_stage2].T, [(t_idx, -1.0)]), 0, np.inf))
    if margin_tax > 0 and A_util.shape[0]:                          # u_max ≥ each utilisation
        cons2.append(LinearConstraint(_rows(A_util, [(u_idx, -1.0)]), -np.inf, 0))

    integ = np.zeros(nv); integ[:nC] = 1
    lb = np.zeros(nv); ub = np.ones(nv); ub[t_idx] = np.inf; ub[u_idx] = np.inf
    res2 = milp(c=c2, constraints=cons2, integrality=integ,
                bounds=Bounds(lb, ub), options=opts)

    yv = res2.x[:nC] if res2.x is not None else res1.x
    chosen = [c for c in range(nC) if yv[c] > 0.5]
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
        avg_fluence=(float(np.minimum(f_eval[cov], fluence_cap).mean())
                     if len(cov) else float(f_eval.mean())),
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
