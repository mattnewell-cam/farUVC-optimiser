# faruvcOptimiser

A far-UVC (222 nm KrCl) luminaire-placement optimiser. You draw a custom-shaped
room (SolidWorks-sketch style), set a **target room-average fluence rate** and a
**regulatory limit** (RP 27.1-22, UL 8802, or ICNIRP), choose whether lamps are
**downlight-only** or **corner/edge lamps are allowed**, and the tool finds the
**minimum number of Ushio Care222 B1 lamps** and their best arrangement that hits
the target average fluence *without* exceeding the exposure limit anywhere in the
occupied zone.

Inspired by [Illuminate](https://illuminate.osluv.org/) (The OSLUV Project), but
deliberately scoped down: **B1 only** (B1.5 later), **no pathogen-reduction
curves**, **no ozone**, and the headline feature Illuminate lacks — **optimisation**.

---

## Status — working v1

All phases built and validated. The physics matches OSLUV's open-source
[`guv-calcs`](https://github.com/jvbelenky/guv-calcs) engine (the same one Illuminate
runs on) to 4 significant figures; the B1 photometry is its real measured IES file.

**Run it:**
```bash
pip install -r requirements.txt
python -m uvicorn faruvc.api:app --reload      # then open http://127.0.0.1:8000
```
Draw a room (click corners, click the first to close, drag to adjust), set the target
average fluence, regulation, and placement mode, then **Optimise**. You get the
minimum compliant number of B1 lamps, their positions, a fluence coverage heatmap,
and eye/skin margins.

**Use the core directly:**
```python
from faruvc.photometry import Photometry
from faruvc.geometry import Room
from faruvc.regs import Standard
from faruvc.optimize import optimize

phot = Photometry.from_ies("data/lamp_data/ushio_b1.ies")
room = Room([(0,0),(5,0),(5,4),(0,4)], height=3.0)
r = optimize(room, phot, target_fluence=2.0, standard=Standard.RP27_1, mode="downlight")
print(r.n_lamps, r.max_skin, r.skin_cap)
```

**Tests:** `PYTHONPATH=. python tests/test_field.py` (inverse-square, downlight-eye,
verified TLVs, and a `guv_calcs` cross-check that skips if the dev oracle is absent).
Run the optimizer eval/performance suite with
`.venv\Scripts\python.exe tests\optimizer_eval.py --quick` or omit `--quick` for the
full room-shape suite.

**Validation results:** our scalar fluence equals `guv_calcs.Lamp.irradiance_at` with
ratio 1.0000 across angles/distances; total B1 radiant power 118.6 mW vs 118.2 mW
(<0.3%); eye/skin caps reproduce `guv_calcs` exactly (see §4).

**Known v1 limitations / next steps:** average-fluence target can leave cold corners
with few lamps (min-fluence reported); large rooms at high targets take a few seconds
to solve; no per-lamp dimming yet (would unlock ICNIRP / low-ceiling cases — guv-calcs
computes the required dim factor); direct illumination only; B1.5 not wired in yet
(its IES is already vendored in `data/lamp_data/`).

---

## 1. Scope (v1)

**In scope**
- One luminaire type: **Ushio Care222 B1** (axially-symmetric far-UVC downlight).
- 2.5D room: a 2D floor polygon + a single ceiling height (extruded prism).
- Two distinct UV quantities (see §3):
  - **Target:** room-average **fluence rate** (germicidal, omnidirectional).
  - **Constraint:** max **skin & eye irradiance** in the occupied zone ≤ the chosen
    8-hour TLV.
- Placement modes: **downlight-only** (ceiling, pointing down) or **corner/edge
  allowed** (also wall/upper-edge mounts, angled).
- Optimiser: **minimum lamp count** + arrangement.

**Explicitly out of scope (for now)**
- B1.5 (different photometry — add after B1 works).
- Pathogen log-reduction / eACH curves.
- Ozone generation.
- Inter-reflection between surfaces (direct illumination only — see §3.4).

---

## 2. Architecture

**Python core + lightweight web UI** (this is the PyCharm project home; the
SolidWorks-style drawing lives in the browser).

```
faruvcOptimiser/
├── README.md                  ← this plan
├── requirements.txt
├── data/
│   └── b1_photometry.json     ← normalized B1 angular intensity I(theta)
├── faruvc/                    ← Python package (the core)
│   ├── photometry.py          ← Lamp model, load I(theta), intensity lookup
│   ├── geometry.py            ← Room polygon, occupied-zone, sample grids
│   ├── field.py               ← fluence-rate field + skin/eye irradiance field
│   ├── regs.py                ← ACGIH eye/skin + ICNIRP TLVs, dose->irradiance
│   ├── candidates.py          ← candidate lamp positions/orientations
│   ├── optimize.py            ← ILP min-count solve + continuous refinement
│   └── api.py                 ← FastAPI app (/optimize), serves the frontend
├── web/                       ← single-page frontend (Canvas sketcher + results)
│   ├── index.html
│   ├── sketch.js              ← polygon drawing, controls
│   └── render.js              ← lamp layout + heatmap overlays
├── scripts/
│   └── build_b1_photometry.py ← turn raw source data into b1_photometry.json
└── tests/
    └── test_field.py          ← analytic point-source checks, etc.
```

Stack: **numpy/scipy** (fields, refinement), **PuLP or OR-Tools (CBC)** (ILP),
**FastAPI + uvicorn** (API), vanilla **HTML5 Canvas + JS** frontend (no build step
for v1; can swap to Svelte/Vite later).

---

## 3. Physics model

The B1 is modelled as a **point source** with an axially-symmetric radiant-intensity
distribution `I(θ)` [W/sr], where `θ` is the polar angle from the lamp's optical
axis. (Point-source is valid since room distances ≫ the lamp aperture; revisit for
near-field corner mounts.)

### 3.1 Per-lamp irradiance at a field point
For a lamp at position `p_L` with unit axis `â`, and a field point `p`:
```
d = p − p_L ,   r = |d| ,   θ = angle(d, â)
E_perp(p) = I(θ) / r²        # W/m², onto a surface facing the lamp
```

### 3.2 Target quantity — fluence rate (germicidal)
The **scalar/spherical fluence rate** at `p` is the omnidirectional sum (no cosine
weighting — a virion in air has no orientation):
```
E0(p) = Σ_lamps  I_i(θ_i) / r_i²
```
**Room-average fluence** = mean of `E0` over the occupied-zone sample grid. This is
the user's headline target (µW/cm²). 1 W/m² = 100 µW/cm².

### 3.3 Constraint quantity — exposure (skin & eye)
The TLV applies to the **directional radiant exposure** on a person. We evaluate, at
every occupant sample point in the occupied zone, the worst-case plane irradiance:
- **Skin:** irradiance on the body; conservatively the max single-lamp-facing plane
  irradiance, summed appropriately (treated as incident on a plane facing the
  dominant source). We start with the unweighted incident sum as a conservative
  upper bound and refine the geometry in §7-validation.
- **Eye:** evaluated on a (near-)vertical plane at eye height (~1.6–1.7 m); the eye
  is largely shielded from overhead downlights, so the eye case usually binds for
  corner/edge mounts, the skin case for downlights.

TLV (an 8-hour **dose**, mJ/cm²) → allowable **irradiance**:
```
E_limit [µW/cm²] = TLV [mJ/cm²] × 1000 / 28800 s
```
The chosen standard (RP 27.1-22 / UL 8802 / ICNIRP) selects the caps and calc modes — see
§4. **Constraint:** `max over occupied zone of exposure ≤ E_limit`.

### 3.4 Assumptions / simplifications (v1)
- **Direct illumination only**, no inter-reflection. Far-UVC wall reflectance is low
  (typ. a few %); ignoring it *under*-estimates fluence (conservative for efficacy)
  and is the standard simplifying choice. Flagged for a future reflective pass.
- Air absorption at 222 nm over room distances is negligible — ignored.
- Lamp output treated as steady (no decay/maintenance factor in v1; the OSLUV data
  has burn-in curves we can fold in later as a derating factor).

---

## 4. Regulations module

Two parts: the **limit** (a TLV from a spectral action curve) and the **assessment
geometry** (how eye/skin irradiance is measured). `regs.py` reproduces guv-calcs'
method exactly: an action curve S(λ) is spectrum-weighted against the B1's emission →
effective 8-h TLV = 3/s [mJ/cm²] → cap = TLV·1000/28800 [µW/cm²]. **Verified** B1 caps
(reproduced to 4 s.f. against `guv_calcs`):

| Standard | eye cap | skin cap | eye calc mode | skin calc mode | plane |
|----------|---------|----------|---------------|----------------|-------|
| **RP 27.1-22 (ACGIH)** | 5.23 | 15.92 | eye worst-case (planar) | planar-normal | 1.8 m |
| **UL 8802 (ACGIH)**    | 5.23 | 15.92 | **fluence rate** | planar-max | 1.9 m |
| **IEC 62471-6:2022 (ICNIRP)** | 0.80 | 0.80 | eye worst-case (planar) | planar-normal | 1.8 m |

(caps in µW/cm². ICNIRP uses one eye/skin curve.) The two ACGIH standards share the
same *limits* but differ in *geometry*: UL 8802 assesses the **eye with fluence rate**
(omnidirectional) and **skin with worst-orientation planar-max**, which is markedly
more conservative — a full-power B1 downlight that passes RP 27.1-22's eye check
(~2.7 µW/cm²) fails UL 8802's (~11 µW/cm² fluence directly beneath) unless the ceiling
is high or the lamp is dimmed. ICNIRP's 0.80 µW/cm² cap is so strict that full power
is non-compliant under any normal ceiling. The optimiser enforces both eye and skin
caps everywhere and reports the binding one. Calc modes mirror guv-calcs'
`PlaneCalcMode`; see `field.py`.

---

## 5. Optimisation

**Goal:** minimise lamp count `N`, then arrangement quality, subject to
(a) average fluence ≥ target and (b) exposure ≤ limit everywhere in the occupied zone.

This is naturally a **set-selection ILP** because both the average-fluence target and
the per-point exposure caps are **linear** in the per-lamp contributions:

1. **Candidate generation** (`candidates.py`)
   - Downlight mode: a grid of ceiling positions, axis pointing straight down.
   - Corner/edge mode: additionally wall/upper-edge positions and room corners, with
     a small set of inward/along-wall orientations.
2. **Precompute contributions**: for each candidate `c`, its fluence contribution to
   every fluence-sample point and its exposure contribution to every occupant point.
3. **ILP** (PuLP / OR-Tools CBC): binary `x_c ∈ {0,1}` (lamp present or not)
   ```
   minimise   Σ_c x_c
   s.t.       (1/|P_f|) Σ_c x_c · E0_{c,p}  ≥  target_fluence      (avg over fluence grid)
              Σ_c x_c · exposure_{c,j}      ≤  E_limit   ∀ occupant point j
   ```
   Adding lamps raises both fluence (good) and exposure (bounded above), so the two
   constraints are in genuine tension — ILP balances them and returns the minimum
   feasible set. Infeasible ⇒ report "target unreachable under this limit/mode".
4. **Arrangement stage** (`optimize.py`, scipy/HiGHS): after the minimum lamp count
   is known, a second MILP chooses the best same-count arrangement. The model applies
   exact fixed-count reductions before solving. Small fixed-count arrangement spaces
   are solved by exhaustive enumeration, which is also exact; larger ones use HiGHS
   under a time budget. If the proof of optimality doesn't finish in time, the best
   incumbent is returned (it converges to the optimum well before the proof does) and
   flagged in `stage2_status` rather than discarded.
5. **Continuous refinement** (`optimize.py`, scipy): nudge the selected lamps off the
   candidate grid (position + orientation) to improve uniformity / margin, keeping
   count fixed. Optional polish stage.

Optional later: a uniformity term (min-fluence / CV) since average-only can leave
cold/hot spots.

---

## 6. Web UI (room sketcher + results)

- **Sketcher** (`sketch.js`): click to drop polygon vertices, drag to adjust, close
  the loop; numeric entry for edge lengths and ceiling height; snapping/ortho like a
  SolidWorks sketch. Mode toggles + numeric inputs:
  - target average fluence (µW/cm²)
  - regulation (RP 27.1-22 (ACGIH) / UL 8802 (ACGIH) / ICNIRP)
  - placement mode (downlight-only / corner+edge allowed)
- **Run** → POST `/optimize` → backend returns lamp positions, field stats, and
  heatmap arrays.
- **Results** (`render.js`): overlay lamp positions on the room, draw the fluence
  heatmap and the exposure map, and show a summary (N lamps, achieved avg fluence,
  max skin/eye exposure vs limit, feasibility).

---

## 7. Validation

- **Analytic:** single lamp, on-axis at distance `r` → `E = I(0)/r²`; unit-test the
  field code against this and against inverse-square falloff.
- **Datasheet/measured:** compare computed B1 on-axis irradiance at a reference
  distance to the OSLUV measured value / Ushio datasheet.
- **Cross-check vs Illuminate:** reproduce a simple rectangular room in both tools and
  compare average fluence and peak exposure; also confirm our TLV numbers match
  Illuminate's.

---

## 8. Photometry data sourcing (B1)

**Resolved:** we use OSLUV's real measured B1 IES, shipped in `guv-calcs`
(`data/lamp_data/ushio_b1.ies`, IESNA LM-63-2002, mW/sr, OSLUV-tested). The two
sources originally considered:

1. **OSLUV (authoritative).** `OSLUV/Sweep-Analyzer` repo ships raw goniometer sweeps
   for the B1 under `OSLUV Data/OSLUV Experiments/B1-*` as `.sw3` files (documented in
   that repo's `docs/FILE_FORMATS.md`: rows carry `yaw_deg`, `roll_deg`, `lin_mm`,
   `integral_result` in W/m²). These are **large Git-LFS research runs** (decay /
   burn-in / full angular sweeps); we extract a clean angular sweep, convert
   `E(θ)·r² → I(θ)`, and normalise.
2. **Illuminate (fallback / cross-check).** The "details" graph in Illuminate is a
   pre-processed angular distribution. Captured from the running app's network/runtime
   data (not static scraping) if the OSLUV sweep proves too noisy.

Output: `data/b1_photometry.json` — `{ angles_deg: [...], intensity_rel: [...],
abs_scale: <I(0) in W/sr or a calibrated 1 m irradiance>, source: ... }`.
`scripts/build_b1_photometry.py` regenerates it from the raw source.

---

## 9. Open items to confirm during execution

- Exact ACGIH (eye & skin) and ICNIRP 222 nm dose values + their spectral weighting.
- Occupied-zone definition: height band (e.g. floor→1.8 m) and any wall setback for
  the exposure check; eye-plane height and shielding assumption.
- Whether the average-fluence target should be volume-averaged or plane-averaged
  (default: occupied-volume grid).
- B1 absolute calibration (W/sr or µW/cm² at a stated distance) for real-unit output.

---

## 10. Build phases

1. **Photometry** — produce `b1_photometry.json` from OSLUV `.sw3` (fallback
   Illuminate). 
2. **Physics core** — `field.py` fluence + exposure; analytic unit tests. 
3. **Regulations** — `regs.py` with verified TLVs. 
4. **Optimiser** — candidates + ILP + refinement. 
5. **API + UI** — FastAPI `/optimize`, Canvas sketcher, result visualisation. 
6. **Validation** — analytic + datasheet + Illuminate cross-check.
