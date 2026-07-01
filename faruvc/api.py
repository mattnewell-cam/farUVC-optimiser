"""FastAPI backend: serves the room-sketcher UI and runs the optimiser.

Run with:  uvicorn faruvc.api:app --reload   (or: python -m faruvc.api)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .field import LampInstance, column_fluence_grid
from .geometry import Room
from .optimize import optimize
from .photometry import Photometry
from .regs import Standard

_ROOT = Path(__file__).resolve().parent.parent
_WEB = _ROOT / "web"
_B1_IES = _ROOT / "data" / "lamp_data" / "ushio_b1.ies"
_B1_SPECTRUM = _ROOT / "data" / "lamp_data" / "ushio_b1.csv"

app = FastAPI(title="faruvcOptimiser")
_PHOT = Photometry.from_ies(_B1_IES)


class OptimizeRequest(BaseModel):
    vertices: list[list[float]] = Field(..., description="[[x,y], ...] metres")
    height: float = 3.0
    target_fluence: float = 1.0
    standard: str = "rp27_1"         # "rp27_1" | "ul8802" | "icnirp"
    mode: str = "downlight"          # "downlight" | "corner_edge"
    goal: str = "throughput"         # "throughput" | "balanced" | "coverage"
    occupant_height: float | None = None   # None -> use the standard's plane height


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


@app.post("/optimize")
def run_optimize(req: OptimizeRequest) -> dict:
    room = Room(vertices=np.array(req.vertices, dtype=float), height=req.height)
    standard = Standard.from_token(req.standard)

    result = optimize(
        room, _PHOT,
        target_fluence=req.target_fluence,
        standard=standard,
        mode=req.mode,
        goal=req.goal,
        spectrum_csv=str(_B1_SPECTRUM),
        occupant_height=req.occupant_height,
    )

    payload: dict = {
        "status": result.status,
        "n_lamps": result.n_lamps,
        "lamps": result.lamps,
        "message": result.message,
        "metrics": {
            "avg_fluence": result.avg_fluence,
            "min_fluence": result.min_fluence,
            "max_skin": result.max_skin,
            "max_eye": result.max_eye,
            "skin_cap": result.skin_cap,
            "eye_cap": result.eye_cap,
            "target_fluence": result.target_fluence,
            "max_util": result.max_util,
        },
        "goal": result.goal,
        "standard_label": standard.label,
    }

    if result.status == "optimal" and result.lamps:
        sel = [LampInstance(_PHOT, pos=(l["x"], l["y"], l["z"]), aim=l["aim"])
               for l in result.lamps]
        vals, extent = column_fluence_grid(room, sel)
        grid = np.where(np.isnan(vals), None, np.round(vals, 4))
        # Clip the colour scale to a robust max so the room-wide coverage gradient is
        # visible rather than washed out by the spike directly under each lamp.
        finite = vals[np.isfinite(vals)]
        vmax = float(np.percentile(finite, 95)) if finite.size else 0.0
        payload["heatmap"] = {
            "extent": extent,
            "values": grid.tolist(),
            "vmax": max(vmax, 1e-6),
            "peak": float(np.nanmax(vals)) if finite.size else 0.0,
        }
    return payload


app.mount("/web", StaticFiles(directory=str(_WEB)), name="web")


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
