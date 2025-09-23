#!/usr/bin/env python3
"""Aggregate key pipeline metrics for the MDK design pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
METRICS_DIR = ROOT / "metrics"


def load_json(path: Path) -> Dict:
    """Load JSON data if the file exists, otherwise return an empty dict."""
    if path.exists():
        with path.open() as handle:
            return json.load(handle)
    return {}


def main() -> None:
    """Collect pipeline metrics and write them to metrics/smoke_test_run.csv."""
    METRICS_DIR.mkdir(exist_ok=True)

    rows: List[Dict[str, str]] = []

    epitope_path = ROOT / "data/epitopes/lrp1_patch.json"
    epitope_data = load_json(epitope_path)
    if epitope_data:
        rows.append(
            {
                "script_path": "03_define_epitope.py",
                "metric": "expanded_epitope_residue_count",
                "value": str(len(epitope_data.get("expanded_residues", []))),
                "output_path": "data/epitopes/lrp1_patch.json",
                "details": "target=data/structures/1mkc.pdb",
            }
        )
        rows.append(
            {
                "script_path": "03_define_epitope.py",
                "metric": "core_epitope_residue_count",
                "value": str(len(epitope_data.get("core_residues", []))),
                "output_path": "data/epitopes/lrp1_patch.json",
                "details": "basic surface residues in C-terminal domain",
            }
        )
        rows.append(
            {
                "script_path": "03_define_epitope.py",
                "metric": "visualization_script_generated",
                "value": "1",
                "output_path": "results/visualizations/lrp1_epitope.pml",
                "details": "PyMOL script for epitope visualization",
            }
        )

    designs_path = ROOT / "results/designs/scores.json"
    designs = load_json(designs_path)
    if isinstance(designs, list) and designs:
        rows.append(
            {
                "script_path": "04_run_bindcraft.py",
                "metric": "top_designs_saved",
                "value": str(len(designs)),
                "output_path": "results/designs/scores.json",
                "details": "top-ranked designs exported after filtering",
            }
        )
        top_design = designs[0]
        rows.append(
            {
                "script_path": "04_run_bindcraft.py",
                "metric": "top_design_composite_score",
                "value": f"{top_design.get('composite_score', 0):.3f}",
                "output_path": "results/designs/design_summary.txt",
                "details": f"design_id={top_design.get('design_id')}",
            }
        )

    nanobody_path = ROOT / "results/nanobodies/nanobody_data.json"
    nanobodies = load_json(nanobody_path)
    if isinstance(nanobodies, list):
        rows.append(
            {
                "script_path": "05_convert_to_nanobody.py",
                "metric": "nanobodies_generated",
                "value": str(len(nanobodies)),
                "output_path": "results/nanobodies/nanobody_data.json",
                "details": "top BindCraft designs reformatted to VHH scaffolds",
            }
        )

    ranked_path = ROOT / "results/validation/ranked_nanobodies.json"
    ranked = load_json(ranked_path)
    if isinstance(ranked, list) and ranked:
        top_nb = ranked[0]
        rows.append(
            {
                "script_path": "06_analyze_results.py",
                "metric": "top_nanobody_final_score",
                "value": f"{top_nb.get('final_score', 0):.2f}",
                "output_path": "results/validation/final_validation_report.txt",
                "details": f"nanobody_id={top_nb.get('id')}",
            }
        )

    csv_path = METRICS_DIR / "smoke_test_run.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "script_path",
                "metric",
                "value",
                "output_path",
                "details",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
