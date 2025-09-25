#!/usr/bin/env python3
"""Execute BindCraft for MDK campaigns with dual LRP1/integrin gating."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import yaml

BINDCRAFT_ROOT = Path("/home/eqk3/apps/BindCraft")
DEFAULT_FILTERS = BINDCRAFT_ROOT / "settings_filters" / "default_filters.json"
DEFAULT_ADVANCED = BINDCRAFT_ROOT / "settings_advanced" / "default_4stage_multimer.json"
TOP_DESIGN_LIMIT = 20
RESULTS_ROOT = Path("results")


@dataclass
class CampaignPaths:
    campaign: str
    root: Path
    bindcraft_root: Path
    designs_dir: Path
    summary_file: Path
    sequences_fasta: Path
    scores_json: Path
    run_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BindCraft for MDK dual-block campaign")
    parser.add_argument("--config", default="config.yaml", help="Configuration YAML (default: config.yaml)")
    parser.add_argument(
        "--epitope-json",
        default=None,
        help="Optional override for epitope JSON; defaults to config.epitope.json",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Override campaign name to segregate outputs (defaults to config output.campaign)",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_paths(campaign: str) -> CampaignPaths:
    base = RESULTS_ROOT / campaign
    bindcraft_root = base / "bindcraft"
    designs_dir = base / "designs"
    summary_file = designs_dir / "design_summary.txt"
    sequences_fasta = designs_dir / "sequences.fasta"
    scores_json = designs_dir / "scores.json"
    run_log = bindcraft_root / "last_run.txt"
    return CampaignPaths(
        campaign=campaign,
        root=base,
        bindcraft_root=bindcraft_root,
        designs_dir=designs_dir,
        summary_file=summary_file,
        sequences_fasta=sequences_fasta,
        scores_json=scores_json,
        run_log=run_log,
    )


def load_epitope(epitope_path: Path) -> Dict:
    with epitope_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derive_hotspots(config: Dict, epitope: Dict, available_positions: Iterable[int]) -> Tuple[List[int], List[int]]:
    config_core = [int(r) for r in config.get("epitope", {}).get("residues", [])]
    epitope_core = [int(r) for r in epitope.get("core_residues", [])]
    integrin_aux = [int(r) for r in epitope.get("integrin_aux", [])]

    core_required = sorted({*config_core, *epitope_core})
    aux_candidates = sorted({int(r) for r in config.get("epitope", {}).get("integrin_aux", [])} | set(integrin_aux))

    available = set(int(pos) for pos in available_positions)
    missing_core = [pos for pos in core_required if pos not in available]
    missing_aux = [pos for pos in aux_candidates if pos not in available]

    if missing_core:
        raise RuntimeError(
            "Epitope hotspots missing from target PDB chain: "
            + ",".join(str(m) for m in missing_core)
        )

    if missing_aux:
        print("Warning: dropping auxiliary residues absent from structure:", ",".join(map(str, missing_aux)))

    aux_present = [pos for pos in aux_candidates if pos in available]

    return core_required, aux_present


def available_residues(pdb_file: Path, chain: str) -> List[int]:
    residues: Set[int] = set()
    with pdb_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[21].strip() == chain:
                try:
                    residues.add(int(line[22:26]))
                except ValueError:
                    continue
    return sorted(residues)


def build_settings(
    config: Dict,
    epitope: Dict,
    paths: CampaignPaths,
) -> Tuple[Path, Path, Path, Path, Dict]:
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    design_root = (paths.bindcraft_root / run_stamp).resolve()
    design_root.mkdir(parents=True, exist_ok=True)

    target_cfg = config["target"]
    bindcraft_cfg = config["bindcraft"]
    lengths = bindcraft_cfg["hallucination"]["binder_length_range"]

    pdb_path = Path(target_cfg["pdb_file"]).resolve()
    chain_id = target_cfg.get("chain", "A").strip() or "A"

    residues_present = available_residues(pdb_path, chain_id)
    core_hotspots, integrin_aux = derive_hotspots(config, epitope, residues_present)
    union_hotspots = sorted({*core_hotspots, *integrin_aux})

    settings = {
        "design_path": str(design_root),
        "binder_name": target_cfg.get("name", "binder"),
        "starting_pdb": str(pdb_path),
        "chains": chain_id,
        "target_hotspot_residues": ",".join(str(r) for r in union_hotspots),
        "lengths": [int(lengths[0]), int(lengths[-1])],
        "number_of_final_designs": bindcraft_cfg.get("number_of_final_designs", TOP_DESIGN_LIMIT),
    }

    settings_path = design_root / "settings.generated.json"
    with settings_path.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)

    # Advanced settings overrides and animation toggles
    with DEFAULT_ADVANCED.open("r", encoding="utf-8") as handle:
        advanced_config = json.load(handle)

    modified_advanced = False
    advanced_overrides = bindcraft_cfg.get("advanced_overrides", {})
    for key, value in advanced_overrides.items():
        if key in advanced_config:
            advanced_config[key] = value
            modified_advanced = True
        else:
            print(f"Warning: Unknown BindCraft advanced override '{key}'; skipping.")

    if os.environ.get("BINDCRAFT_SKIP_ANIMATION", "").lower() in {"1", "true", "yes", "on"} or shutil.which("ffmpeg") is None:
        advanced_config["save_design_animations"] = False
        advanced_config["zip_animations"] = False
        advanced_config["save_design_trajectory_plots"] = False
        modified_advanced = True

    advanced_path = DEFAULT_ADVANCED
    if modified_advanced:
        advanced_path = design_root / "advanced.generated.json"
        with advanced_path.open("w", encoding="utf-8") as handle:
            json.dump(advanced_config, handle, indent=2)

    with DEFAULT_FILTERS.open("r", encoding="utf-8") as handle:
        filters_config = json.load(handle)

    # Tighten key thresholds
    filters_config.setdefault("Average_dSASA", {"threshold": None, "higher": True})
    filters_config["Average_dSASA"]["threshold"] = 600.0
    filters_config.setdefault("Average_i_pTM", {"threshold": None, "higher": True})
    filters_config["Average_i_pTM"]["threshold"] = 0.6
    filters_config.setdefault("Average_i_pAE", {"threshold": None, "higher": False})
    filters_config["Average_i_pAE"]["threshold"] = 0.35
    filters_config.setdefault("Average_Binder_pLDDT", {"threshold": None, "higher": True})
    filters_config["Average_Binder_pLDDT"]["threshold"] = 0.8

    filters_path = design_root / "filters.generated.json"
    with filters_path.open("w", encoding="utf-8") as handle:
        json.dump(filters_config, handle, indent=2)

    gating = {
        "core_hotspots": set(core_hotspots),
        "integrin_aux": set(integrin_aux),
        "required_aux_contacts": 8,
        "min_interface_area": 600.0,
        "preferred_interface_area": 800.0,
        "min_iptm": 0.60,
        "max_ipae": 10.0,
        "min_binder_plddt": 0.80,
    }

    return settings_path, advanced_path, filters_path, design_root, gating


def invoke_bindcraft(settings_path: Path, advanced_path: Path, filters_path: Path) -> None:
    if not BINDCRAFT_ROOT.exists():
        raise RuntimeError(f"BindCraft repository missing at {BINDCRAFT_ROOT}")

    cmd = [
        "python",
        "bindcraft.py",
        "--settings",
        str(settings_path),
        "--filters",
        str(filters_path),
        "--advanced",
        str(advanced_path if advanced_path != DEFAULT_ADVANCED else DEFAULT_ADVANCED),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH"), str(BINDCRAFT_ROOT)]))

    print("\nRunning BindCraft pipeline:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=str(BINDCRAFT_ROOT), env=env, check=True)


def load_design_table(design_root: Path) -> pd.DataFrame:
    final_csv = design_root / "final_design_stats.csv"
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


def parse_hotspot_contacts(value: Optional[str]) -> Set[int]:
    contacts: Set[int] = set()
    if not isinstance(value, str):
        return contacts
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            contacts.add(int(token))
        except ValueError:
            continue
    return contacts


def apply_dual_block_filters(table: pd.DataFrame, gating: Dict) -> Tuple[pd.DataFrame, List[str]]:
    rejects: List[str] = []
    table = table.copy()
    table["hotspot_contacts"] = table["Target_Hotspot"].apply(parse_hotspot_contacts)
    table["core_ok"] = table["hotspot_contacts"].apply(lambda s: gating["core_hotspots"].issubset(s))
    table["aux_count"] = table["hotspot_contacts"].apply(lambda s: len(s & gating["integrin_aux"]))

    table["interface_area"] = table["Average_dSASA"].astype(float)
    table["iptm"] = table["Average_i_pTM"].astype(float)
    table["ipae"] = table.get("Average_i_pAE", pd.Series([0.0] * len(table))).astype(float)
    table["binder_plddt"] = table.get("Average_Binder_pLDDT", pd.Series([0.0] * len(table))).astype(float)

    table["interface_tier"] = table["interface_area"].apply(
        lambda area: "preferred" if area >= gating["preferred_interface_area"] else (
            "fallback" if area >= gating["min_interface_area"] else "reject"
        )
    )

    keep_mask = (
        table["core_ok"]
        & (table["aux_count"] >= gating["required_aux_contacts"])
        & (table["interface_tier"] != "reject")
        & (table["iptm"] >= gating["min_iptm"])
        & (table["ipae"] <= gating["max_ipae"])
        & (table["binder_plddt"] >= gating["min_binder_plddt"])
    )

    rejected = table[~keep_mask]
    for _, row in rejected.iterrows():
        reasons = []
        if not row["core_ok"]:
            reasons.append("missing_core_contacts")
        if row["aux_count"] < gating["required_aux_contacts"]:
            reasons.append(f"aux_contacts_{row['aux_count']}")
        if row["interface_tier"] == "reject":
            reasons.append(f"interface_{row['interface_area']:.1f}Å²")
        if row["iptm"] < gating["min_iptm"]:
            reasons.append(f"iptm_{row['iptm']:.2f}")
        if row["ipae"] > gating["max_ipae"]:
            reasons.append(f"ipae_{row['ipae']:.2f}")
        if row["binder_plddt"] < gating["min_binder_plddt"]:
            reasons.append(f"plddt_{row['binder_plddt']:.2f}")
        rejects.append(f"{row['Design']}: {', '.join(reasons)}")

    filtered = table[keep_mask].reset_index(drop=True)
    return filtered, rejects


def compute_composite(row: pd.Series) -> float:
    pl_ddt = float(row.get("Average_pLDDT", 0.0))
    iptm = float(row.get("Average_i_pTM", row.get("Average_pTM", 0.0)))
    pae = float(row.get("Average_i_pAE", row.get("Average_pAE", 1.0)))
    interface_area = float(row.get("interface_area", row.get("Average_dSASA", 0.0)))

    pl_term = max(min(pl_ddt, 1.0), 0.0)
    iptm_term = max(min(iptm, 1.0), 0.0)
    pae_term = 1.0 - max(min(pae, 1.0), 0.0)
    area_term = min(max(interface_area, 0.0) / 1000.0, 1.0)

    return 0.25 * pl_term + 0.45 * iptm_term + 0.2 * pae_term + 0.1 * area_term


def write_sequences(designs: List[Dict], paths: CampaignPaths) -> None:
    paths.designs_dir.mkdir(parents=True, exist_ok=True)
    with paths.sequences_fasta.open("w", encoding="utf-8") as fasta:
        for entry in designs[:TOP_DESIGN_LIMIT]:
            fasta.write(f">{entry['design_id']}_score_{entry['composite_score']:.3f}\n")
            fasta.write(f"{entry['sequence']}\n")

    with paths.scores_json.open("w", encoding="utf-8") as handle:
        json.dump(designs[:TOP_DESIGN_LIMIT], handle, indent=2)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    data = [float(v) for v in values]
    if not data:
        return 0.0, 0.0
    if len(data) == 1:
        return data[0], 0.0
    mean = sum(data) / len(data)
    variance = sum((val - mean) ** 2 for val in data) / (len(data) - 1)
    return mean, variance ** 0.5


def render_summary(designs: List[Dict], epitope: Dict, design_root: Path, paths: CampaignPaths) -> None:
    paths.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with paths.summary_file.open("w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write(f"BindCraft Design Campaign Summary ({paths.campaign})\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Design run directory: {design_root}\n")
        handle.write(f"Epitope core (LRP1) count: {len(epitope.get('core_residues', []))}\n")
        handle.write(f"Integrin auxiliary count: {len(epitope.get('integrin_aux', []))}\n")
        handle.write(f"Expanded shell size: {len(epitope.get('expanded_residues', []))}\n\n")

        handle.write(f"Designs returned after gating: {len(designs)}\n")
        top_preview = designs[:5]
        if top_preview:
            pl_ddt = [entry["scores"]["plddt"] for entry in top_preview]
            iptm = [entry["scores"]["iptm"] for entry in top_preview]
            interface_area = [entry["competition"]["interface_area"] for entry in top_preview]

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
                handle.write(f"     Interface area: {entry['competition']['interface_area']:.0f} Å² ({entry['competition']['interface_tier']})\n")
                handle.write(f"     Hotspot coverage: {sorted(entry['competition']['hotspot_contacts'])}\n")
                handle.write(f"     Integrin contacts: {entry['competition']['integrin_contacts']}\n")

    paths.run_log.parent.mkdir(parents=True, exist_ok=True)
    with paths.run_log.open("w", encoding="utf-8") as handle:
        handle.write(str(design_root) + "\n")


def make_design_records(table: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    for _, row in table.head(TOP_DESIGN_LIMIT).iterrows():
        composite = compute_composite(row)
        contacts = sorted(row.get("hotspot_contacts", []))
        record = {
            "rank": int(row.get("Rank", len(records) + 1)),
            "design_id": str(row.get("Design", f"design_{len(records) + 1}")),
            "sequence": str(row["Sequence"]),
            "sequence_length": int(row.get("Length", len(row["Sequence"]))),
            "scores": {
                "plddt": float(row.get("Average_pLDDT", 0.0) * 100.0),
                "iptm": float(row.get("Average_i_pTM", row.get("Average_pTM", 0.0))),
                "pae": float(row.get("Average_i_pAE", row.get("Average_pAE", 0.0)) * 30.0),
                "interface_area": float(row.get("interface_area", 0.0)),
            },
            "competition": {
                "hotspot_contacts": contacts,
                "integrin_contacts": int(row.get("aux_count", 0)),
                "interface_area": float(row.get("interface_area", 0.0)),
                "interface_tier": row.get("interface_tier", "unknown"),
            },
            "composite_score": float(composite),
        }
        records.append(record)

    if not records:
        raise RuntimeError("No designs available after dual-block gating.")

    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    epitope_path = Path(args.epitope_json or config.get("epitope", {}).get("json", "data/epitopes/mdk_dual_block_patch.json"))
    epitope = load_epitope(epitope_path)

    campaign_name = args.campaign or config.get("output", {}).get("campaign", "default")
    paths = configure_paths(campaign_name)

    print("\n" + "=" * 60)
    print(f"Running BindCraft campaign: {campaign_name}")
    print("=" * 60)

    settings_path, advanced_path, filters_path, design_root, gating = build_settings(config, epitope, paths)
    invoke_bindcraft(settings_path, advanced_path, filters_path)

    print("\nCollecting BindCraft outputs...")
    table = load_design_table(design_root)
    filtered_table, rejects = apply_dual_block_filters(table, gating)
    if rejects:
        print("Filtered designs:")
        for reason in rejects:
            print(f"  - {reason}")

    if filtered_table.empty:
        raise RuntimeError("No designs passed dual-block gating; inspect BindCraft outputs for troubleshooting.")

    design_records = make_design_records(filtered_table)
    write_sequences(design_records, paths)
    render_summary(design_records, epitope, design_root, paths)

    best = design_records[0]
    print("\nBindCraft design complete!")
    print(f"Run directory: {design_root}")
    print(f"Top design: {best['design_id']} (composite {best['composite_score']:.3f})")
    print(f"Sequences: {paths.sequences_fasta}")
    print(f"Scores: {paths.scores_json}")


if __name__ == "__main__":
    main()
