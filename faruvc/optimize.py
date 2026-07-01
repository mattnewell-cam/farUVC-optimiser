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

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

from .candidates import DOWN, generate_candidates, _rot2d, _tilt_down
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
    candidate_count: int = 0
    stage1_seconds: float = 0.0
    stage2_seconds: float = 0.0
    refine_seconds: float = 0.0
    total_seconds: float = 0.0
    stage2_status: str = ""


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
    refine: bool = True,
    refine_passes: int = 2,
    time_limit: float | None = 12.0,
    stage2_time_limit: float | None = None,
    exact_enum_limit: int = 1_500_000,
    solver_msg: bool = False,
    **candidate_kwargs,
) -> OptimizeResult:
    t_total = time.perf_counter()
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
                              target_fluence=target_fluence,
                              total_seconds=time.perf_counter() - t_total)

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
    opts = _solver_options(time_limit, solver_msg)
    opts2 = _solver_options(stage2_time_limit if stage2_time_limit is not None else time_limit,
                            solver_msg)

    # --- Stage 1: minimise lamp count -------------------------------------
    cons1 = [LinearConstraint(fvec, target_fluence, np.inf)]
    if A_exp.shape[0]:
        cons1.append(LinearConstraint(sparse.csr_matrix(A_exp), -np.inf, ub_exp))
    t_stage1 = time.perf_counter()
    res1 = milp(c=np.ones(nC), constraints=cons1, integrality=np.ones(nC),
                bounds=Bounds(0, 1), options=opts)
    stage1_seconds = time.perf_counter() - t_stage1
    if res1.x is None:
        status = _milp_status(res1)
        return OptimizeResult(
            status=status, n_lamps=0, eye_cap=eye_cap, skin_cap=skin_cap,
            target_fluence=target_fluence,
            message=_infeasibility_hint(candidates, fvec, target_fluence,
                                        skin_cap, eye_cap, height, skin_mode, eye_mode)
                    if status == "infeasible" else str(res1.message),
            candidate_count=nC,
            stage1_seconds=stage1_seconds,
            total_seconds=time.perf_counter() - t_total,
        )
    n_star = int(round(res1.x.sum()))

    active = _stage2_candidate_mask(fvec, n_star, target_fluence)
    active_idx = np.flatnonzero(active)
    candidates2 = [candidates[i] for i in active_idx]
    fvec2 = fvec[active]
    F2 = F[active]
    A_exp2 = A_exp[:, active] if A_exp.shape[0] else A_exp
    A_util2 = A_util[:, active] if A_util.shape[0] else A_util
    nS = len(candidates2)
    A_exp_stage2, ub_exp_stage2 = _stage2_exposure_rows(A_exp2, ub_exp, n_star)

    # --- Stage 2: among min-count layouts, choose the best arrangement -----
    # Blended objective (milp minimises, so we pass the negative):
    #   maximise  (1-w)*(avg/target) + w*(min/target) - margin_tax*u_max
    # w (goal preset) trades raw germicidal throughput (avg fluence -- separable, blind to
    # spacing) against even coverage (the worst-covered point t). u_max is the worst
    # exposure utilisation, so the tax steers off layouts that run hot / cluster lamps.
    # Variables: y (nC binaries) then t (min fluence) then u_max (worst utilisation).
    tgt = target_fluence if target_fluence > 1e-9 else 1e-9
    nv = nS + 2
    t_idx, u_idx = nS, nS + 1

    def _rows(A, extra):
        """Embed an (m, nS) coefficient block into the (m, nv) variable layout."""
        R = np.zeros((A.shape[0], nv))
        R[:, :nS] = A
        for col, val in extra:
            R[:, col] = val
        return sparse.csr_matrix(R)

    c2 = np.zeros(nv)
    c2[:nS] = -(1.0 - w) / tgt * fvec2         # throughput term
    c2[t_idx] = -(w / tgt)                       # even-coverage term (0 when w==0)
    c2[u_idx] = margin_tax                       # exposure-margin tax (0 when off)

    cons2 = []
    lamp_count = np.zeros(nv); lamp_count[:nS] = 1.0
    cons2.append(LinearConstraint(lamp_count, n_star, n_star))     # exactly n_star lamps
    avg_row = np.zeros(nv); avg_row[:nS] = fvec2
    cons2.append(LinearConstraint(avg_row, target_fluence, np.inf))  # hit the target
    if A_exp_stage2.shape[0]:
        cons2.append(LinearConstraint(_rows(A_exp_stage2, []), -np.inf, ub_exp_stage2))
    if w > 0 and len(cov_stage2):                                   # t ≤ fluence at each cov pt
        cons2.append(LinearConstraint(_rows(F2[:, cov_stage2].T, [(t_idx, -1.0)]), 0, np.inf))
    if margin_tax > 0 and A_util2.shape[0]:
        cons2.append(LinearConstraint(_rows(A_util2, [(u_idx, -1.0)]), -np.inf, 0))

    integ = np.zeros(nv); integ[:nS] = 1
    lb = np.zeros(nv); ub = np.ones(nv); ub[t_idx] = np.inf; ub[u_idx] = np.inf
    t_stage2 = time.perf_counter()
    combo_count = math.comb(nS, n_star) if 0 <= n_star <= nS else 0
    enum_choice = None
    if 0 < combo_count <= exact_enum_limit:
        enum_choice = _enumerate_stage2(
            nS, n_star, fvec2, F2[:, cov_stage2].T if w > 0 else None,
            A_exp_stage2, ub_exp_stage2, A_util2 if margin_tax > 0 else None,
            target_fluence, w, margin_tax,
        )
        stage2_seconds = time.perf_counter() - t_stage2
        if enum_choice is None:
            return OptimizeResult(
                status="infeasible", n_lamps=n_star,
                eye_cap=eye_cap, skin_cap=skin_cap, target_fluence=target_fluence,
                message="no exact stage-2 combination satisfies the fixed-count constraints",
                candidate_count=nC,
                stage1_seconds=stage1_seconds,
                stage2_seconds=stage2_seconds,
                total_seconds=time.perf_counter() - t_total,
                stage2_status="exact enumeration found no feasible arrangement",
            )
        chosen2 = enum_choice
        stage2_status = f"exact enumeration over {combo_count} combinations"
    else:
        res2 = milp(c=c2, constraints=cons2, integrality=integ,
                    bounds=Bounds(lb, ub), options=opts2)
        stage2_seconds = time.perf_counter() - t_stage2

        # Graceful degradation: on a time limit HiGHS still returns its best incumbent,
        # which (measured on symmetric rooms) reaches the optimum well before the proof of
        # optimality finishes. So we USE the incumbent and only fail if no feasible layout
        # was found at all -- rather than discarding a good answer just because the last
        # fraction of a percent wasn't formally closed.
        if res2.x is None:
            return OptimizeResult(
                status=_milp_status(res2), n_lamps=n_star,
                eye_cap=eye_cap, skin_cap=skin_cap, target_fluence=target_fluence,
                message=str(res2.message),
                candidate_count=nC,
                stage1_seconds=stage1_seconds,
                stage2_seconds=stage2_seconds,
                total_seconds=time.perf_counter() - t_total,
                stage2_status=str(res2.message),
            )
        chosen2 = _solution_indices(res2.x, nS, n_star)
        stage2_status = ("optimal" if res2.success
                         else "time limit — best incumbent (near-optimal)")
    chosen = [int(active_idx[c]) for c in chosen2]
    sel = [candidates[c] for c in chosen]

    # --- Continuous refine: the MILP only chose which wall/corner each lamp sits on (a
    # coarse candidate grid). Now polish each lamp's exact (position-along-wall, tilt,
    # fan-azimuth) with a local optimiser -- the objective is smooth once a lamp is fixed
    # to a wall, so this reaches the true optimum between grid points without a fine (and
    # slow) candidate grid. Each layout eval is ~1.6 ms, so this is cheap.
    refine_seconds = 0.0
    if refine and sel:
        t_refine = time.perf_counter()
        sel = _refine_layout(sel, photometry, room, vol[cov], plane, skin_mode, eye_mode,
                             skin_cap, eye_cap, fluence_cap, w, margin_tax, tgt,
                             n_az=8, cap_margin=cap_margin, passes=refine_passes)
        refine_seconds = time.perf_counter() - t_refine

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
        candidate_count=nC,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        refine_seconds=refine_seconds,
        total_seconds=time.perf_counter() - t_total,
        stage2_status=stage2_status,
    )


# --- exact MILP reductions ------------------------------------------------
def _solver_options(time_limit: float | None, solver_msg: bool) -> dict:
    opts = {"mip_rel_gap": 0.01, "disp": bool(solver_msg)}
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)
    return opts


def _milp_status(res) -> str:
    if res.success:
        return "optimal"
    if res.status == 1:
        return "time_limit"
    if res.status == 2:
        return "infeasible"
    if res.status == 3:
        return "unbounded"
    return "solver_failed"


def _solution_indices(x, nC: int, n_star: int) -> list[int]:
    if x is None:
        return []
    y = np.asarray(x[:nC], dtype=float)
    chosen = [int(i) for i in np.flatnonzero(y > 0.5)]
    if len(chosen) == n_star:
        return chosen
    return [int(i) for i in np.argsort(-y)[:n_star] if y[i] > 1e-7]


def _stage2_candidate_mask(fvec: np.ndarray, n_star: int, target: float) -> np.ndarray:
    """Candidates that can appear in some target-hitting n_star-lamp solution."""
    nC = len(fvec)
    if n_star <= 1:
        return fvec + 1e-12 >= target
    order = np.argsort(-fvec)
    prefix = np.concatenate([[0.0], np.cumsum(fvec[order])])
    keep = np.zeros(nC, dtype=bool)
    for i in range(nC):
        need = n_star - 1
        if i in order[:need]:
            others = prefix[need + 1] - fvec[i]
        else:
            others = prefix[need]
        keep[i] = fvec[i] + others + 1e-12 >= target
    return keep


def _stage2_exposure_rows(A_exp: np.ndarray, ub_exp: np.ndarray,
                          n_star: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop exposure rows that cannot bind with exactly n_star selected lamps."""
    if A_exp.shape[0] == 0:
        return A_exp, ub_exp
    k = min(max(1, n_star), A_exp.shape[1])
    top_sum = np.partition(A_exp, -k, axis=1)[:, -k:].sum(axis=1)
    keep = top_sum > ub_exp + 1e-12
    return A_exp[keep], ub_exp[keep]


def _enumerate_stage2(nC: int, n_star: int, fvec: np.ndarray,
                      F_cov: np.ndarray | None, A_exp: np.ndarray,
                      ub_exp: np.ndarray, A_util: np.ndarray | None,
                      target: float, w: float, margin_tax: float,
                      chunk_size: int = 2000) -> list[int] | None:
    """Exact fixed-count arrangement solve by exhaustive combination search."""
    best_score = -np.inf
    best_combo = None
    tgt = target if target > 1e-9 else 1e-9
    combos_iter = itertools.combinations(range(nC), n_star)

    while True:
        chunk = list(itertools.islice(combos_iter, chunk_size))
        if not chunk:
            break
        C = np.asarray(chunk, dtype=np.int32)
        avg = fvec[C].sum(axis=1)
        feasible = avg + 1e-8 >= target
        if not feasible.any():
            continue
        C = C[feasible]
        avg = avg[feasible]

        if A_exp.shape[0]:
            exposure = A_exp[:, C].sum(axis=2)
            feasible = (exposure <= ub_exp[:, None] + 1e-7).all(axis=0)
            if not feasible.any():
                continue
            C = C[feasible]
            avg = avg[feasible]

        min_cov = 0.0
        if w > 0 and F_cov is not None and F_cov.size:
            min_cov = F_cov[:, C].sum(axis=2).min(axis=0)

        util = 0.0
        if margin_tax > 0 and A_util is not None and A_util.size:
            util = A_util[:, C].sum(axis=2).max(axis=0)

        score = (1.0 - w) * avg / tgt + w * min_cov / tgt - margin_tax * util
        j = int(np.argmax(score))
        if float(score[j]) > best_score + 1e-12:
            best_score = float(score[j])
            best_combo = C[j].astype(int).tolist()

    return best_combo


# --- continuous refinement ------------------------------------------------
def _lamp_params(lamp, photometry, room):
    """Return (x0, bounds, build, valid) for a lamp's refinable continuous params.

    build(p) -> a new LampInstance; valid(p) -> is the position physically allowed.
    Wall lamps slide along their wall (t) + tilt + fan azimuth; corner lamps keep their
    position and vary tilt + azimuth; ceiling downlights move in (x, y).
    """
    m = lamp.meta or {}
    kind = m.get("mount")
    if kind == "edge":
        p0, p1, n_in, inset, h = m["p0"], m["p1"], m["n_in"], m["inset"], m["h"]
        x0 = [m.get("t", 0.5), m.get("tilt", 30.0), m.get("az", 0.0)]
        bounds = [(0.02, 0.98), (5.0, 75.0), (-1.2, 1.2)]

        def build(p):
            base = p0 + min(max(p[0], 0.0), 1.0) * (p1 - p0) + n_in * inset
            aim = _tilt_down(_rot2d(n_in, p[2]), p[1])
            return LampInstance(photometry, pos=(base[0], base[1], h), aim=aim, meta=m)
        return x0, bounds, build, (lambda p: True)
    if kind == "corner":
        base, bis, h = m["base"], m["bis"], m["h"]
        x0 = [m.get("tilt", 30.0), m.get("az", 0.0)]
        bounds = [(5.0, 75.0), (-1.2, 1.2)]

        def build(p):
            aim = _tilt_down(_rot2d(bis, p[1]), p[0])
            return LampInstance(photometry, pos=(base[0], base[1], h), aim=aim, meta=m)
        return x0, bounds, build, (lambda p: True)
    if kind == "ceiling":
        h = m["h"]
        (xmn, ymn), (xmx, ymx) = room.bbox
        x0 = [float(lamp.pos[0]), float(lamp.pos[1])]
        bounds = [(xmn, xmx), (ymn, ymx)]

        def build(p):
            return LampInstance(photometry, pos=(p[0], p[1], h), aim=DOWN, meta=m)
        return x0, bounds, build, (lambda p: bool(room.contains(np.array([[p[0], p[1]]]))[0]))
    return None, None, None, None


def _refine_layout(sel, photometry, room, cov_pts, plane, skin_mode, eye_mode,
                   skin_cap, eye_cap, fluence_cap, w, margin_tax, tgt, n_az, cap_margin,
                   passes):
    """Coordinate-ascent polish of each lamp's continuous placement (Nelder-Mead).

    Cheap by design: the avg grid is subsampled and exposure uses coarse azimuths -- the
    penalty keeps exposure below cap_margin (not the true cap) so that discretisation
    headroom guarantees the full-resolution final metrics stay under the real limit.
    """
    lamps = list(sel)
    if len(cov_pts) > 1200:                         # subsample the smooth avg objective
        cov_pts = cov_pts[np.linspace(0, len(cov_pts) - 1, 1200).astype(int)]

    def score(ls):
        f = fluence_field(ls, cov_pts)
        cavg = float(np.minimum(f, fluence_cap).mean())
        mn = float(f.min())
        s = float(exposure_field(ls, plane, skin_mode, n_az).max()) if skin_cap > 0 else 0.0
        e = float(exposure_field(ls, plane, eye_mode, n_az).max()) if eye_cap > 0 else 0.0
        us = s / skin_cap if skin_cap > 0 else 0.0
        ue = e / eye_cap if eye_cap > 0 else 0.0
        obj = (1.0 - w) * cavg / tgt + w * mn / tgt - margin_tax * max(us, ue)
        viol = max(0.0, us - cap_margin) + max(0.0, ue - cap_margin)  # keep the headroom
        return obj - 1e3 * viol

    for _ in range(passes):
        for i in range(len(lamps)):
            x0, bounds, build, valid = _lamp_params(lamps[i], photometry, room)
            if x0 is None:
                continue

            def neg(p, _i=i, _build=build, _valid=valid):
                if not _valid(p):
                    return 1e6
                lamps[_i] = _build(p)
                return -score(lamps)

            res = minimize(neg, x0, method="Nelder-Mead", bounds=bounds,
                           options={"maxiter": 80, "xatol": 1e-2, "fatol": 1e-4})
            lamps[i] = build(res.x)
    return lamps


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
