#!/usr/bin/env python3
"""Analyze nanobody designs with dual LRP1/integrin competition gating."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import ProtParam

sns.set_style("whitegrid")




def load_reference_residue_numbers(pdb_file: Path, chain_id: str) -> List[int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("reference", str(pdb_file))
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id} missing from reference structure {pdb_file}")
    numbers: List[int] = []
    for residue in model[chain_id].get_residues():
        if residue.id[0] == ' ':
            numbers.append(residue.id[1])
    return numbers


def build_residue_number_map(pdb_path: Path, chain_id: str, reference_numbers: List[int]) -> Dict[int, int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("design", str(pdb_path))
    model = structure[0]
    if chain_id not in model:
        return {}
    residues = [residue for residue in model[chain_id].get_residues() if residue.id[0] == ' ']
    mapping: Dict[int, int] = {}
    limit = min(len(residues), len(reference_numbers))
    for idx in range(limit):
        mapping[residues[idx].id[1]] = reference_numbers[idx]
    return mapping
@dataclass
class CampaignPaths:
    campaign: str
    base: Path
    designs_dir: Path
    nanobodies_dir: Path
    validation_dir: Path
    bindcraft_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MDK nanobody designs")
    parser.add_argument("--config", default="config.yaml", help="Configuration YAML (default: config.yaml)")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign name; overrides config output.campaign",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit BindCraft run directory containing Accepted models (overrides last_run.txt)",
    )
    parser.add_argument(
        "--footprints-dir",
        default="data/footprints",
        help="Directory containing receptor footprint JSON files",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_paths(config: Dict, campaign_override: Optional[str]) -> CampaignPaths:
    campaign = campaign_override or config.get("output", {}).get("campaign", "default")
    base = Path(config["output"]["base_dir"]) / campaign
    designs_dir = base / "designs"
    nanobodies_dir = base / "nanobodies"
    validation_dir = base / "validation"
    bindcraft_log = base / "bindcraft" / "last_run.txt"
    return CampaignPaths(campaign, base, designs_dir, nanobodies_dir, validation_dir, bindcraft_log)


def load_nanobodies(nanobody_dir: Path) -> List[Dict]:
    data_file = nanobody_dir / "nanobody_data.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Nanobody data not found: {data_file}")
    with data_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_footprint(footprints_dir: Path, name: str) -> List[int]:
    path = footprints_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Footprint file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    residues = sorted({int(r) for r in payload.get("residues", [])})
    if not residues:
        raise ValueError(f"Footprint {path} does not contain residues.")
    return residues


def resolve_run_directory(paths: CampaignPaths, override: Optional[str]) -> Path:
    if override:
        run_dir = Path(override).resolve()
    else:
        if not paths.bindcraft_log.exists():
            raise FileNotFoundError(
                f"BindCraft log {paths.bindcraft_log} missing; provide --run-dir to locate Accepted structures."
            )
        run_dir = Path(paths.bindcraft_log.read_text().strip()).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"BindCraft run directory not found: {run_dir}")
    return run_dir


def find_design_pdb(run_dir: Path, design_id: str) -> Optional[Path]:
    accepted_dir = run_dir / "Accepted"
    if not accepted_dir.exists():
        return None
    matches = sorted(accepted_dir.glob(f"{design_id}_model*.pdb"))
    if matches:
        return matches[0]
    fallback = accepted_dir / f"{design_id}.pdb"
    return fallback if fallback.exists() else None


def compute_residue_sasa(structure, chain_id: str) -> Dict[int, float]:
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    sasas: Dict[int, float] = {}
    model = structure[0]
    if chain_id not in model:
        return sasas
    chain = model[chain_id]
    for residue in chain.get_residues():
        if residue.id[0] != ' ':
            continue
        sasas[residue.id[1]] = residue.xtra.get("EXP_SASA", 0.0)
    return sasas


def compute_occlusion(pdb_path: Path, target_chain: str, footprint: List[int], sasa_drop: float = 0.2,
                      distance_cutoff: float = 5.0, residue_map: Optional[Dict[int, int]] = None) -> Tuple[float, Dict[int, Dict[str, float]]]:
    parser = PDBParser(QUIET=True)
    complex_structure = parser.get_structure("complex", str(pdb_path))
    model = complex_structure[0]
    if target_chain not in model:
        raise KeyError(f"Chain {target_chain} absent from {pdb_path}")

    binder_atoms = []
    for chain in model:
        if chain.id == target_chain:
            continue
        binder_atoms.extend(atom for atom in chain.get_atoms() if atom.element != 'H')

    ns = NeighborSearch(binder_atoms) if binder_atoms else None

    complex_sasa = compute_residue_sasa(complex_structure, target_chain)

    free_structure = parser.get_structure("target_only", str(pdb_path))
    free_model = free_structure[0]
    for chain in list(free_model):
        if chain.id != target_chain:
            free_model.detach_child(chain.id)
    free_sasa = compute_residue_sasa(free_structure, target_chain)

    occluded = 0
    per_residue: Dict[int, Dict[str, float]] = {}

    target_chain_obj = model[target_chain]
    for residue in target_chain_obj.get_residues():
        if residue.id[0] != ' ':
            continue
        resnum = residue.id[1]
        mapped_num = residue_map.get(resnum, resnum) if residue_map else resnum
        if mapped_num not in footprint:
            continue

        free_val = free_sasa.get(resnum, 0.0)
        complex_val = complex_sasa.get(resnum, free_val)
        drop = (free_val - complex_val) / free_val if free_val > 1e-6 else 0.0

        within_distance = False
        if ns is not None:
            for atom in residue.get_atoms():
                if atom.element == 'H':
                    continue
                close = ns.search(atom.coord, distance_cutoff)
                if close:
                    within_distance = True
                    break

        is_occluded = False
        if free_val > 1e-6 and drop >= sasa_drop:
            is_occluded = True
        elif within_distance:
            is_occluded = True

        if is_occluded:
            occluded += 1

        per_residue[mapped_num] = {
            "sasa_free": free_val,
            "sasa_complex": complex_val,
            "drop_fraction": drop,
            "binder_contact": 1.0 if within_distance else 0.0,
        }

    if not footprint:
        return 0.0, per_residue

    return occluded / len(footprint), per_residue


def check_ptn_cross_reactivity(nanobody_sequence: str) -> Dict[str, object]:
    motifs = ['RGR', 'KKK', 'RKR']
    score = sum(1 for motif in motifs if motif in nanobody_sequence)
    risk = 'low' if score >= 2 else 'medium' if score >= 1 else 'high'
    return {"specificity_score": score, "risk": risk}


def analyze_developability(sequence: str) -> Dict[str, object]:
    analyzer = ProtParam.ProteinAnalysis(sequence)
    length = len(sequence)
    if length == 0:
        raise ValueError('Nanobody sequence is empty.')

    aliphatic_attr = getattr(analyzer, 'aliphatic_index', None)
    if callable(aliphatic_attr):
        aliphatic_index = aliphatic_attr()
    else:
        ala = sequence.count('A') / length * 100.0
        val = sequence.count('V') / length * 100.0
        ile = sequence.count('I') / length * 100.0
        leu = sequence.count('L') / length * 100.0
        aliphatic_index = ala + 2.9 * val + 3.9 * (ile + leu)

    developability = {
        'molecular_weight': analyzer.molecular_weight(),
        'theoretical_pi': analyzer.isoelectric_point(),
        'instability_index': analyzer.instability_index(),
        'aliphatic_index': aliphatic_index,
        'gravy': analyzer.gravy(),
        'aromaticity': analyzer.aromaticity(),
        'secondary_structure': analyzer.secondary_structure_fraction(),
    }

    risks = []
    if developability['theoretical_pi'] < 6.5 or developability['theoretical_pi'] > 9.0:
        risks.append(f"pI outside optimal range: {developability['theoretical_pi']:.1f}")
    if developability['instability_index'] > 40:
        risks.append(f"High instability index: {developability['instability_index']:.1f}")
    if developability['gravy'] > 0:
        risks.append(f"Positive GRAVY score: {developability['gravy']:.2f}")

    hydrophobic_aa = 'FWYLIMV'
    max_stretch = 0
    current = 0
    for aa in sequence:
        if aa in hydrophobic_aa:
            current += 1
            max_stretch = max(max_stretch, current)
        else:
            current = 0
    if max_stretch >= 5:
        risks.append(f"Long hydrophobic stretch: {max_stretch} aa")

    developability['risks'] = risks
    developability['risk_level'] = 'high' if len(risks) >= 3 else 'medium' if risks else 'low'
    return developability


def predict_expression_level(sequence: str) -> Dict[str, object]:
    rare_aa = 'WCM'
    gc_favorable = 'GAPR'
    hydrophobic_aa = 'FWYLIMV'

    factors = {
        'rare_codons': sum(1 for aa in sequence if aa in rare_aa),
        'gc_content': sum(1 for aa in sequence if aa in gc_favorable) / len(sequence),
        'hydrophobicity': sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence),
        'size': len(sequence),
    }

    expression_score = 100
    expression_score -= factors['rare_codons'] * 1.5
    expression_score -= factors['hydrophobicity'] * 40
    expression_score -= max(0, factors['size'] - 120) * 0.2
    expression_score = max(expression_score, 0)

    level = 'high' if expression_score >= 70 else 'medium' if expression_score >= 45 else 'low'

    return {
        'score': expression_score,
        'level': level,
        'factors': factors,
    }


def calculate_production_metrics(nanobody: Dict[str, object]) -> Dict[str, object]:
    mw = nanobody['properties']['molecular_weight_kda']
    production = {
        'estimated_yield_mg_per_l': 50.0,
        'purification_score': 100 if mw <= 12 else 70,
        'production_cost': 'medium'
    }
    return production


def rank_nanobodies(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    for nb in candidates:
        score = 0.0
        score += nb['original_scores'].get('iptm', 0.5) * 30
        score += nb['competition']['lrp1_occlusion'] * 25
        score += nb['competition']['integrin_occlusion'] * 15

        dev_level = nb['developability']['risk_level']
        score += 15 if dev_level == 'low' else 10 if dev_level == 'medium' else 5

        score += (nb['expression']['score'] / 100) * 10

        specificity = nb['ptn_specificity']['risk']
        score += 5 if specificity == 'low' else 3 if specificity == 'medium' else 1

        nb['final_score'] = round(score, 2)

    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return candidates


def generate_visualization(candidates: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for nb in candidates:
        data.append(
            {
                'ID': nb['id'].replace('nb_design_', 'NB'),
                'Final Score': nb['final_score'],
                'pI': nb['developability']['theoretical_pi'],
                'Instability': nb['developability']['instability_index'],
                'Expression': nb['expression']['score'],
                'MW (kDa)': nb['developability']['molecular_weight'] / 1000,
                'LRP1 occlusion': nb['competition']['lrp1_occlusion'],
                'Integrin occlusion': nb['competition']['integrin_occlusion'],
            }
        )

    df = pd.DataFrame(data)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    df_sorted = df.sort_values('Final Score', ascending=True)
    axes[0, 0].barh(df_sorted['ID'], df_sorted['Final Score'])
    axes[0, 0].set_xlabel('Final Score')
    axes[0, 0].set_title('Nanobody Ranking')

    axes[0, 1].scatter(df['LRP1 occlusion'], df['Integrin occlusion'], c=df['Final Score'], cmap='viridis')
    axes[0, 1].axvline(0.7, color='r', linestyle='--', alpha=0.4)
    axes[0, 1].axhline(0.5, color='r', linestyle='--', alpha=0.4)
    axes[0, 1].set_xlabel('LRP1 occlusion')
    axes[0, 1].set_ylabel('Integrin occlusion')
    axes[0, 1].set_title('Competition coverage')

    scatter = axes[1, 0].scatter(df['MW (kDa)'], df['Expression'], c=df['Final Score'], cmap='plasma')
    plt.colorbar(scatter, ax=axes[1, 0], label='Final Score')
    axes[1, 0].set_xlabel('Molecular Weight (kDa)')
    axes[1, 0].set_ylabel('Expression score')
    axes[1, 0].set_title('Expression prediction')

    top_nb = candidates[0]
    components = ['Binding', 'LRP1 occ', 'Integrin occ', 'Developability', 'Expression', 'Specificity']
    values = [
        top_nb['original_scores'].get('iptm', 0.5) * 100,
        top_nb['competition']['lrp1_occlusion'] * 100,
        top_nb['competition']['integrin_occlusion'] * 100,
        100 if top_nb['developability']['risk_level'] == 'low' else 66 if top_nb['developability']['risk_level'] == 'medium' else 33,
        top_nb['expression']['score'],
        100 if top_nb['ptn_specificity']['risk'] == 'low' else 66 if top_nb['ptn_specificity']['risk'] == 'medium' else 33,
    ]
    axes[1, 1].bar(components, values)
    axes[1, 1].set_ylabel('Score (%)')
    axes[1, 1].set_title(f"Top candidate breakdown ({top_nb['id']})")
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(output_dir / "analysis_plots.png", dpi=200)
    plt.close(fig)


def generate_final_report(candidates: List[Dict[str, object]], rejects: List[Dict[str, object]], output_file: Path, passed_gate_count: Optional[int] = None) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as handle:
        handle.write("=" * 80 + "\n")
        handle.write("MDK-LRP1/Integrin Nanobody Validation Report\n")
        handle.write("=" * 80 + "\n\n")
        passed = passed_gate_count if passed_gate_count is not None else len(candidates)
        handle.write(f"Total designs passing competition gate: {passed}\n")
        handle.write(f"Rejected for competition shortfall: {len(rejects)}\n\n")
        if passed_gate_count is not None and passed_gate_count == 0:
            handle.write("NOTE: All candidates failed occlusion gate; ranking presented for diagnostics.\n\n")

        handle.write("TOP CANDIDATES\n")
        handle.write("-" * 40 + "\n")
        for idx, nb in enumerate(candidates[:3], start=1):
            handle.write(f"{idx}. {nb['id']} (Score: {nb['final_score']})\n")
            handle.write(f"   Binding ipTM: {nb['original_scores'].get('iptm', 'N/A'):.3f}\n")
            handle.write(f"   LRP1 occlusion: {nb['competition']['lrp1_occlusion']:.2f}\n")
            handle.write(f"   Integrin occlusion: {nb['competition']['integrin_occlusion']:.2f}\n")
            handle.write(f"   MW: {nb['developability']['molecular_weight']/1000:.1f} kDa\n")
            handle.write(f"   pI: {nb['developability']['theoretical_pi']:.1f}\n")
            handle.write(f"   Risks: {', '.join(nb['developability']['risks']) if nb['developability']['risks'] else 'none'}\n\n")

        if rejects:
            handle.write("DESIGNS REJECTED ON COMPETITION GATE\n")
            handle.write("-" * 40 + "\n")
            for nb in rejects:
                handle.write(
                    f"{nb['id']}: LRP1 occ {nb['competition']['lrp1_occlusion']:.2f}, "
                    f"Integrin occ {nb['competition']['integrin_occlusion']:.2f} -> {nb['competition']['reject_reason']}\n"
                )
            handle.write("\n")

        handle.write("DETAILED SUMMARIES\n")
        handle.write("-" * 40 + "\n")
        for nb in candidates:
            handle.write(f"{nb['id']}\n")
            handle.write(f"  Final score: {nb['final_score']}\n")
            handle.write(f"  Sequence length: {len(nb['sequence'])} aa\n")
            handle.write(f"  LRP1 occlusion: {nb['competition']['lrp1_occlusion']:.2f}\n")
            handle.write(f"  Integrin occlusion: {nb['competition']['integrin_occlusion']:.2f}\n")
            handle.write(f"  Developability risk: {nb['developability']['risk_level']}\n")
            handle.write(f"  Expression: {nb['expression']['level']} ({nb['expression']['score']:.1f})\n")
            handle.write(f"  PTN specificity risk: {nb['ptn_specificity']['risk']}\n")
            if nb['competition'].get('per_residue'):
                handle.write(f"  Occluded residues: {sorted(nb['competition']['per_residue'].keys())}\n")
            handle.write("\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = configure_paths(config, args.campaign)

    print("=" * 60)
    print(f"Final Analysis for campaign: {paths.campaign}")
    print("=" * 60)

    nanobodies = load_nanobodies(paths.nanobodies_dir)
    print(f"Loaded {len(nanobodies)} nanobodies")

    run_dir = resolve_run_directory(paths, args.run_dir)
    print(f"Using BindCraft run directory: {run_dir}")

    footprints_dir = Path(args.footprints_dir)
    lrp1_footprint = load_footprint(footprints_dir, "lrp1_mdkc.json")
    integrin_footprint = load_footprint(footprints_dir, "integrin_b1_headpiece_mdkc.json")

    lrp1_gate = 0.70
    integrin_gate = 0.50

    eligible: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []

    target_chain = config['target'].get('chain', 'A').strip() or 'A'
    reference_pdb = Path(config['target']['pdb_file']).resolve()
    reference_numbers = load_reference_residue_numbers(reference_pdb, target_chain)

    for nb in nanobodies:
        design_id = nb.get('original_design') or nb['id'].replace('nb_', '')
        pdb_path = find_design_pdb(run_dir, design_id)
        competition = {
            'lrp1_occlusion': 0.0,
            'integrin_occlusion': 0.0,
            'per_residue': {},
        }

        if pdb_path and pdb_path.exists():
            try:
                residue_map = build_residue_number_map(pdb_path, target_chain, reference_numbers)
                lrp1_occ, lrp1_res = compute_occlusion(pdb_path, target_chain, lrp1_footprint, residue_map=residue_map)
                integrin_occ, integrin_res = compute_occlusion(pdb_path, target_chain, integrin_footprint, residue_map=residue_map)
                competition['lrp1_occlusion'] = lrp1_occ
                competition['integrin_occlusion'] = integrin_occ
                competition['per_residue'] = {**lrp1_res, **integrin_res}
            except Exception as exc:  # pragma: no cover
                print(f"Warning: failed occlusion analysis for {design_id}: {exc}")
        else:
            print(f"Warning: PDB for {design_id} not found; occlusion set to 0")

        nb['competition'] = competition
        nb['ptn_specificity'] = check_ptn_cross_reactivity(nb['sequence'])
        nb['developability'] = analyze_developability(nb['sequence'])
        nb['expression'] = predict_expression_level(nb['sequence'])
        nb['production'] = calculate_production_metrics(nb)

        passes_lrp1 = competition['lrp1_occlusion'] >= lrp1_gate
        passes_integrin = competition['integrin_occlusion'] >= integrin_gate
        if passes_lrp1 and passes_integrin:
            eligible.append(nb)
        else:
            reason = []
            if not passes_lrp1:
                reason.append(f"LRP1<{lrp1_gate}")
            if not passes_integrin:
                reason.append(f"Integrin<{integrin_gate}")
            competition['reject_reason'] = ",".join(reason)
            rejected.append(nb)

    passed_gate_count = len(eligible)
    ranking_pool = eligible if eligible else nanobodies
    if not eligible:
        print("Warning: no nanobodies passed occlusion gates; ranking full set for diagnostics.")

    ranked_nanobodies = rank_nanobodies(ranking_pool)

    paths.validation_dir.mkdir(parents=True, exist_ok=True)
    generate_visualization(ranked_nanobodies, paths.validation_dir)

    report_file = paths.validation_dir / "final_validation_report.txt"
    generate_final_report(ranked_nanobodies, rejected, report_file, passed_gate_count=passed_gate_count)

    ranked_data_file = paths.validation_dir / "ranked_nanobodies.json"
    with ranked_data_file.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    'id': nb['id'],
                    'final_score': nb['final_score'],
                    'sequence': nb['sequence'],
                    'developability_risk': nb['developability']['risk_level'],
                    'expression_level': nb['expression']['level'],
                    'production_cost': nb['production']['production_cost'],
                    'lrp1_occlusion': nb['competition']['lrp1_occlusion'],
                    'integrin_occlusion': nb['competition']['integrin_occlusion'],
                }
                for nb in ranked_nanobodies
            ],
            handle,
            indent=2,
        )

    print("\nAnalysis complete.")
    print(f"Top candidate: {ranked_nanobodies[0]['id']} (score {ranked_nanobodies[0]['final_score']})")
    print(f"Report: {report_file}")
    if rejected:
        print(f"Rejected designs due to occlusion gate: {len(rejected)}")


if __name__ == "__main__":
    main()
