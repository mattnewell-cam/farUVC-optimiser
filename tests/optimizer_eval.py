r"""Optimizer performance/eval suite.

Run from the repo root:

    .venv\Scripts\python.exe tests\optimizer_eval.py --quick
    .venv\Scripts\python.exe tests\optimizer_eval.py
    .venv\Scripts\python.exe tests\optimizer_eval.py --json

The cases are deliberately small enough to run on a laptop but varied enough to catch
solver regressions: rectangles, corridors, concave rooms, corner/edge mounts, and a
strict-standard infeasible setup.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faruvc.geometry import Room  # noqa: E402
from faruvc.optimize import optimize  # noqa: E402
from faruvc.photometry import Photometry  # noqa: E402
from faruvc.regs import Standard  # noqa: E402

IES = ROOT / "data" / "lamp_data" / "ushio_b1.ies"
SPECTRUM = ROOT / "data" / "lamp_data" / "ushio_b1.csv"


@dataclass(frozen=True)
class EvalCase:
    name: str
    vertices: tuple[tuple[float, float], ...]
    height: float
    target: float
    standard: Standard
    mode: str
    goal: str
    expected_status: str = "optimal"


@dataclass
class EvalResult:
    name: str
    status: str
    expected_status: str
    ok: bool
    seconds_median: float
    seconds_min: float
    seconds_max: float
    n_lamps: int
    avg_fluence: float
    min_fluence: float
    max_util: float
    candidates: int
    stage1_seconds: float
    stage2_seconds: float
    refine_seconds: float
    stage2_status: str


CASES = [
    EvalCase(
        "small_square_down",
        ((0, 0), (4, 0), (4, 4), (0, 4)),
        3.0, 1.2, Standard.RP27_1, "downlight", "throughput",
    ),
    EvalCase(
        "medium_rectangle_down",
        ((0, 0), (8, 0), (8, 5), (0, 5)),
        3.0, 1.2, Standard.RP27_1, "downlight", "throughput",
    ),
    EvalCase(
        "long_corridor_corner_coverage",
        ((0, 0), (12, 0), (12, 2.4), (0, 2.4)),
        3.0, 1.0, Standard.RP27_1, "corner_edge", "coverage",
    ),
    EvalCase(
        "l_shape_corner_balanced",
        ((0, 0), (7, 0), (7, 3), (4, 3), (4, 6), (0, 6)),
        3.0, 1.1, Standard.RP27_1, "corner_edge", "balanced",
    ),
    EvalCase(
        "u_shape_corner_coverage",
        ((0, 0), (8, 0), (8, 6), (5.5, 6), (5.5, 2.2), (2.5, 2.2), (2.5, 6), (0, 6)),
        3.0, 1.0, Standard.RP27_1, "corner_edge", "coverage",
    ),
    EvalCase(
        "wide_open_corner_balanced",
        ((0, 0), (10, 0), (10, 6), (0, 6)),
        3.2, 1.1, Standard.RP27_1, "corner_edge", "balanced",
    ),
    EvalCase(
        "strict_icnirp_down_infeasible",
        ((0, 0), (5, 0), (5, 4), (0, 4)),
        3.0, 0.8, Standard.ICNIRP, "downlight", "throughput", "infeasible",
    ),
]

QUICK_CASES = {
    "small_square_down",
    "medium_rectangle_down",
    "l_shape_corner_balanced",
}


def run_case(case: EvalCase, phot: Photometry, repeat: int, refine: bool,
             stage2_time_limit: float | None) -> EvalResult:
    timings = []
    last = None
    room = Room(np.array(case.vertices, dtype=float), case.height)
    for _ in range(repeat):
        t0 = time.perf_counter()
        last = optimize(
            room, phot,
            target_fluence=case.target,
            standard=case.standard,
            mode=case.mode,
            goal=case.goal,
            spectrum_csv=str(SPECTRUM),
            refine=refine,
            stage2_time_limit=stage2_time_limit,
        )
        timings.append(time.perf_counter() - t0)

    assert last is not None
    ok = last.status == case.expected_status
    if last.status == "optimal":
        ok = ok and last.n_lamps > 0 and last.avg_fluence + 1e-6 >= case.target
        ok = ok and last.max_util <= 1.0 + 1e-6

    return EvalResult(
        name=case.name,
        status=last.status,
        expected_status=case.expected_status,
        ok=ok,
        seconds_median=statistics.median(timings),
        seconds_min=min(timings),
        seconds_max=max(timings),
        n_lamps=last.n_lamps,
        avg_fluence=last.avg_fluence,
        min_fluence=last.min_fluence,
        max_util=last.max_util,
        candidates=last.candidate_count,
        stage1_seconds=last.stage1_seconds,
        stage2_seconds=last.stage2_seconds,
        refine_seconds=last.refine_seconds,
        stage2_status=last.stage2_status,
    )


def print_table(results: list[EvalResult]) -> None:
    headers = ("case", "ok", "sec", "status", "n", "avg", "min", "util", "cand", "s1", "s2")
    rows = []
    for r in results:
        rows.append((
            r.name,
            "yes" if r.ok else "NO",
            f"{r.seconds_median:.3f}",
            r.status,
            str(r.n_lamps),
            f"{r.avg_fluence:.2f}",
            f"{r.min_fluence:.2f}",
            f"{r.max_util:.2f}",
            str(r.candidates),
            f"{r.stage1_seconds:.3f}",
            f"{r.stage2_seconds:.3f}",
        ))
    widths = [max(len(str(v)) for v in col) for col in zip(headers, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optimizer eval/performance cases.")
    parser.add_argument("--quick", action="store_true", help="run a representative subset")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--refine", action="store_true", help="include continuous refinement")
    parser.add_argument(
        "--stage2-time-limit", type=float, default=-1.0,
        help="seconds for arrangement MILP; default -1 uses the full exact optimizer limit",
    )
    args = parser.parse_args()

    selected = [c for c in CASES if not args.quick or c.name in QUICK_CASES]
    stage2_time_limit = None if args.stage2_time_limit < 0 else args.stage2_time_limit
    phot = Photometry.from_ies(IES)
    results = [run_case(c, phot, args.repeat, args.refine, stage2_time_limit) for c in selected]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print_table(results)
        failed = [r.name for r in results if not r.ok]
        if failed:
            print("\nFailed cases: " + ", ".join(failed))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
