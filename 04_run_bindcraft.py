#!/usr/bin/env python3
"""Invoke the upstream BindCraft pipeline and marshal results for the MDX flow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

BINDCRAFT_ROOT = Path("/home/eqk3/apps/BindCraft")
DEFAULT_FILTERS = BINDCRAFT_ROOT / "settings_filters" / "default_filters.json"
DEFAULT_ADVANCED = BINDCRAFT_ROOT / "settings_advanced" / "default_4stage_multimer.json"
TOP_DESIGN_LIMIT = 20
RESULTS_ROOT = Path("results")
DESIGNS_DIR = RESULTS_ROOT / "designs"
SUMMARY_FILE = DESIGNS_DIR / "design_summary.txt"
SEQUENCES_FASTA = DESIGNS_DIR / "sequences.fasta"
SCORES_JSON = DESIGNS_DIR / "scores.json"
BINDCRAFT_RUN_LOG = RESULTS_ROOT / "bindcraft" / "last_run.txt"


def load_config() -> Dict:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_epitope() -> Dict:
    epitope_path = Path("data/epitopes/lrp1_patch.json")
    with open(epitope_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_settings(config: Dict, epitope: Dict) -> Tuple[Path, Path]:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    design_root = (RESULTS_ROOT / "bindcraft" / run_stamp).resolve()
    design_root.mkdir(parents=True, exist_ok=True)

    target_cfg = config["target"]
    bindcraft_cfg = config["bindcraft"]
    lengths = bindcraft_cfg["hallucination"]["binder_length_range"]
    hotspot_core = epitope.get("core_residues", [])
    hotspot_config = config.get("epitope", {}).get("residues", [])
    hotspots = sorted({int(r) for r in hotspot_core + hotspot_config})

    pdb_path = Path(target_cfg["pdb_file"]).resolve()
    chain_id = target_cfg.get("chain", "A").strip() or "A"

    def _available_residues(pdb_file: Path, chain: str) -> List[int]:
        residues: set[int] = set()
        if not pdb_file.exists():
            raise FileNotFoundError(f"Target PDB missing: {pdb_file}")
        with open(pdb_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")) and line[21].strip() == chain:
                    try:
                        residues.add(int(line[22:26]))
                    except ValueError:
                        continue
        return sorted(residues)

    available_positions = set(_available_residues(pdb_path, chain_id))
    missing_hotspots = [pos for pos in hotspots if pos not in available_positions]
    hotspots = [pos for pos in hotspots if pos in available_positions]

    if not hotspots:
        raise RuntimeError(
            "Epitope hotspot residues missing after validating against target PDB. "
            f"Missing: {missing_hotspots} in chain {chain_id} of {pdb_path}"
        )
    if missing_hotspots:
        print(
            "Warning: dropping hotspot residues absent from target PDB:",
            ",".join(str(h) for h in missing_hotspots),
        )

    settings = {
        "design_path": str(design_root),
        "binder_name": target_cfg.get("name", "binder"),
        "starting_pdb": str(pdb_path),
        "chains": chain_id,
        "target_hotspot_residues": ",".join(str(r) for r in hotspots),
        "lengths": [int(lengths[0]), int(lengths[-1])],
        "number_of_final_designs": bindcraft_cfg.get("number_of_final_designs", 20),
    }

    settings_path = (design_root / "settings.generated.json").resolve()
    with open(settings_path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    # Determine advanced settings, optionally disabling animations when ffmpeg is unavailable
    ffmpeg_available = shutil.which("ffmpeg") is not None
    skip_animation_env = os.environ.get("BINDCRAFT_SKIP_ANIMATION", "").lower() in {"1", "true", "yes", "on"}
    advanced_path = DEFAULT_ADVANCED

    if skip_animation_env or not ffmpeg_available:
        with open(DEFAULT_ADVANCED, "r", encoding="utf-8") as handle:
            advanced_config = json.load(handle)

        advanced_config["save_design_animations"] = False
        advanced_config["zip_animations"] = False
        advanced_config["save_design_trajectory_plots"] = False

        advanced_path = (design_root / "advanced.generated.json").resolve()
        with open(advanced_path, "w", encoding="utf-8") as handle:
            json.dump(advanced_config, handle, indent=2)

        if not ffmpeg_available and not skip_animation_env:
            print("Warning: ffmpeg not found; skipping animation export for this run.")
        elif skip_animation_env:
            print("Info: BINDCRAFT_SKIP_ANIMATION set; skipping animation export.")

    return settings_path, advanced_path


def invoke_bindcraft(settings_path: Path, advanced_path: Path) -> Path:
    if not BINDCRAFT_ROOT.exists():
        raise RuntimeError(f"BindCraft repository missing at {BINDCRAFT_ROOT}")
    if not DEFAULT_FILTERS.exists() or not DEFAULT_ADVANCED.exists():
        raise RuntimeError("BindCraft auxiliary settings are missing; please reinstall the repo.")

    cmd = [
        "python",
        "bindcraft.py",
        "--settings",
        str(settings_path),
        "--filters",
        str(DEFAULT_FILTERS),
        "--advanced",
        str(advanced_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), str(BINDCRAFT_ROOT)]))

    print("\nRunning BindCraft pipeline:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(BINDCRAFT_ROOT), env=env, check=True)
    return settings_path.parent


def load_design_table(design_path: Path) -> pd.DataFrame:
    final_csv = design_path / "final_design_stats.csv"
    if not final_csv.exists():
        raise FileNotFoundError(f"BindCraft did not emit {final_csv}; check upstream logs.")

    table = pd.read_csv(final_csv)
    if table.empty:
        raise RuntimeError("BindCraft completed but produced no accepted designs.")

    if "Rank" in table.columns:
        table = table.sort_values("Rank")

    table = table[table["Sequence"].notna() & (table["Sequence"].str.len() > 0)]
    if table.empty:
        raise RuntimeError("All BindCraft designs are missing sequences.")

    return table.reset_index(drop=True)


def compute_composite(record: pd.Series) -> float:
    pl_ddt = float(record.get("Average_pLDDT", 0.0))
    iptm = float(record.get("Average_i_pTM", record.get("Average_pTM", 0.0)))
    pae = float(record.get("Average_i_pAE", record.get("Average_pAE", 1.0)))
    interface_area = float(record.get("Average_dSASA", 0.0))

    pl_term = max(min(pl_ddt, 1.0), 0.0)
    iptm_term = max(min(iptm, 1.0), 0.0)
    pae_term = 1.0 - max(min(pae, 1.0), 0.0)
    area_term = min(max(interface_area, 0.0) / 1000.0, 1.0)

    return 0.3 * pl_term + 0.4 * iptm_term + 0.2 * pae_term + 0.1 * area_term


def write_sequences(designs: List[Dict]) -> None:
    DESIGNS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SEQUENCES_FASTA, "w", encoding="utf-8") as fasta:
        for entry in designs[:TOP_DESIGN_LIMIT]:
            fasta.write(f">{entry['design_id']}_score_{entry['composite_score']:.3f}\n")
            fasta.write(f"{entry['sequence']}\n")

    with open(SCORES_JSON, "w", encoding="utf-8") as handle:
        json.dump(designs[:TOP_DESIGN_LIMIT], handle, indent=2)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    data = [float(v) for v in values]
    if not data:
        return 0.0, 0.0
    mean = sum(data) / len(data)
    if len(data) == 1:
        return mean, 0.0
    variance = sum((val - mean) ** 2 for val in data) / (len(data) - 1)
    return mean, variance ** 0.5


def render_summary(designs: List[Dict], epitope: Dict, design_path: Path) -> None:
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write("BindCraft Design Campaign Summary\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Design run directory: {design_path}\n")
        handle.write(f"Epitope size (expanded): {len(epitope.get('expanded_residues', []))} residues\n")
        handle.write(f"Core hotspots: {', '.join(map(str, epitope.get('core_residues', [])))}\n\n")

        handle.write(f"Designs returned: {len(designs)}\n")
        top_preview = designs[:5]
        if top_preview:
            pl_ddt = [entry["scores"]["plddt"] for entry in top_preview]
            iptm = [entry["scores"]["iptm"] for entry in top_preview]
            interface_area = [entry["scores"]["interface_area"] for entry in top_preview]

            pl_mean, pl_sd = _mean_std(pl_ddt)
            iptm_mean, iptm_sd = _mean_std(iptm)
            area_mean, area_sd = _mean_std(interface_area)

            handle.write("\nTop 5 design statistics:\n")
            handle.write(f"  pLDDT (mean±sd): {pl_mean:.1f} ± {pl_sd:.1f}\n")
            handle.write(f"  ipTM  (mean±sd): {iptm_mean:.3f} ± {iptm_sd:.3f}\n")
            handle.write(f"  Interface area (Å²): {area_mean:.0f} ± {area_sd:.0f}\n")

            handle.write("\nTop Candidates:\n")
            for idx, entry in enumerate(top_preview, start=1):
                handle.write(f"\n  {idx}. {entry['design_id']}\n")
                handle.write(f"     Composite score: {entry['composite_score']:.3f}\n")
                handle.write(f"     pLDDT: {entry['scores']['plddt']:.1f}\n")
                handle.write(f"     ipTM: {entry['scores']['iptm']:.3f}\n")
                handle.write(f"     Interface area: {entry['scores']['interface_area']:.0f} Å²\n")

    with open(BINDCRAFT_RUN_LOG, "w", encoding="utf-8") as handle:
        handle.write(str(design_path) + "\n")


def make_design_records(table: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    for _, row in table.head(TOP_DESIGN_LIMIT).iterrows():
        composite = compute_composite(row)
        record = {
            "rank": int(row.get("Rank", len(records) + 1)),
            "design_id": str(row.get("Design", f"design_{len(records) + 1}")),
            "sequence": str(row["Sequence"]),
            "sequence_length": int(row.get("Length", len(row["Sequence"]))),
            "scores": {
                "plddt": float(row.get("Average_pLDDT", 0.0) * 100.0),
                "iptm": float(row.get("Average_i_pTM", row.get("Average_pTM", 0.0))),
                "pae": float(row.get("Average_i_pAE", row.get("Average_pAE", 0.0)) * 30.0),
                "interface_area": float(row.get("Average_dSASA", 0.0)),
            },
            "composite_score": float(composite),
        }
        records.append(record)

    if not records:
        raise RuntimeError("No designs available for downstream analysis.")

    return records


def main() -> None:
    print("\n" + "=" * 60)
    print("Running BindCraft (real pipeline)")
    print("=" * 60)

    config = load_config()
    epitope = load_epitope()
    settings_path, advanced_path = build_settings(config, epitope)
    design_path = invoke_bindcraft(settings_path, advanced_path)

    print("\nCollecting BindCraft outputs...")
    table = load_design_table(design_path)
    design_records = make_design_records(table)
    write_sequences(design_records)
    render_summary(design_records, epitope, design_path)

    best = design_records[0]
    print("\nBindCraft design complete!")
    print(f"Run directory: {design_path}")
    print(f"Top design: {best['design_id']} (composite {best['composite_score']:.3f})")
    print(f"Sequences: {SEQUENCES_FASTA}")
    print(f"Scores: {SCORES_JSON}")


if __name__ == "__main__":
    main()
