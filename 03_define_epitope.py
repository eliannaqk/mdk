#!/usr/bin/env python3
"""Define and visualise the dual (LRP1+integrin) epitope on MDK C-domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from Bio.PDB import NeighborSearch, PDBParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MDK dual-block epitope definition")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--epitope-json",
        default="data/epitopes/mdk_dual_block_patch.json",
        help="Epitope JSON describing LRP1 core and integrin auxiliary residues",
    )
    parser.add_argument(
        "--visualisation",
        default="results/visualizations/mdk_dual_block_epitope.pml",
        help="Output PyMOL script for visualisation",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_epitope_json(path: str) -> Tuple[Dict, List[int], List[int], List[int], float]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    core = sorted({int(r) for r in payload.get("core_residues", [])})
    integrin_aux = sorted({int(r) for r in payload.get("integrin_aux", [])})
    expanded = sorted({int(r) for r in payload.get("expanded_residues", [])})
    expand_radius = float(payload.get("expand_radius", 5.0))

    return payload, core, integrin_aux, expanded, expand_radius


def cluster_residues(chain, residues: List[Dict], distance_cutoff: float = 10.0) -> List[List[int]]:
    if not residues:
        return []

    coords: List[np.ndarray] = []
    numbers: List[int] = []

    for info in residues:
        num = info["number"]
        try:
            residue = chain[num]
        except KeyError:
            continue
        if "CA" not in residue:
            continue
        coords.append(residue["CA"].coord)
        numbers.append(num)

    if not coords:
        return []

    coords = np.array(coords)
    patches: List[List[int]] = []
    used = set()

    for idx, res_num in enumerate(numbers):
        if res_num in used:
            continue
        patch = [res_num]
        used.add(res_num)
        for jdx, other in enumerate(numbers):
            if other in used:
                continue
            if np.linalg.norm(coords[idx] - coords[jdx]) < distance_cutoff:
                patch.append(other)
                used.add(other)
        patches.append(sorted(patch))

    return patches


def expand_epitope(structure, chain_id: str, core_numbers: List[int], radius: float) -> List[int]:
    model = structure[0]
    chain = model[chain_id]

    target_atoms = []
    for residue in chain.get_residues():
        if residue.id[0] != ' ':
            continue
        if residue.id[1] in core_numbers:
            target_atoms.extend(list(residue.get_atoms()))

    all_atoms = list(structure.get_atoms())
    ns = NeighborSearch(all_atoms)
    expanded = set(core_numbers)

    for atom in target_atoms:
        for neighbour in ns.search(atom.coord, radius):
            parent = neighbour.get_parent()
            if parent.id[0] == ' ':
                expanded.add(parent.id[1])

    return sorted(expanded)


def collect_residue_metadata(chain, residue_numbers: List[int]) -> List[Dict]:
    metadata: List[Dict] = []
    for number in sorted(residue_numbers):
        try:
            residue = chain[number]
        except KeyError:
            continue
        if residue.id[0] != ' ':
            continue
        metadata.append(
            {
                "number": number,
                "name": residue.get_resname(),
                "chain": chain.id,
            }
        )
    return metadata


def visualize_epitope(pdb_path: str, residues: List[int], output_path: Path) -> None:
    selection = "+".join(str(r) for r in residues)
    pymol_script = f"""# PyMOL script to visualise MDK dual-block epitope

load {pdb_path}, MDK
hide everything, MDK
show cartoon, MDK
color grey80, MDK

select mdk_dual_block, MDK and resi {selection}
show sticks, mdk_dual_block
color marine, mdk_dual_block

show surface, MDK
set transparency, 0.5, MDK
color skyblue, mdk_dual_block

zoom mdk_dual_block, 10
orient mdk_dual_block
bg_color white
set ray_shadows, 0
save {str(output_path).replace('.pml', '.pse')}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(pymol_script)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    pdb_file = config["target"]["pdb_file"]
    chain_id = config["target"].get("chain", "A").strip() or "A"

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("MDK", pdb_file)

    payload, core, integrin_aux, expanded, expand_radius = load_epitope_json(args.epitope_json)

    print("=" * 60)
    print("MDK C-domain dual epitope definition")
    print("=" * 60)
    print(f"PDB: {pdb_file} (chain {chain_id})")
    print(f"LRP1 hotspots ({len(core)}): {core}")
    print(f"Integrin auxiliary residues ({len(integrin_aux)}): {integrin_aux}")

    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain '{chain_id}' not present in {pdb_file}")
    chain = model[chain_id]

    union_numbers = sorted({*core, *integrin_aux})
    residue_info = collect_residue_metadata(chain, union_numbers)
    patches = cluster_residues(chain, residue_info)

    if not expanded:
        print("No expanded shell supplied; computing using neighbour search...")
        expanded = expand_epitope(structure, chain_id, union_numbers, expand_radius)

    print(f"Expanded shell residues ({len(expanded)}): {expanded[:10]}{'...' if len(expanded) > 10 else ''}")

    epitope_payload = {
        "target": payload.get("target", config["target"]["name"]),
        "epitope_name": payload.get("epitope_name", "LRP1_and_Integrin_union"),
        "description": payload.get(
            "description",
            "Union of LRP1-facing and integrin-facing surfaces on MDK C-domain",
        ),
        "core_residues": core,
        "integrin_aux": integrin_aux,
        "expanded_residues": expanded,
        "patches": patches,
        "basic_residues": residue_info,
        "expand_radius": expand_radius,
        "surface_area_estimate": len(expanded) * 50,
        "source_json": args.epitope_json,
    }

    output_path = Path(args.epitope_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(epitope_payload, handle, indent=2)
    print(f"Saved epitope definition to {output_path}")

    visualize_epitope(pdb_file, union_numbers, Path(args.visualisation))
    print(f"Wrote PyMOL script to {args.visualisation}")
    print("Done.")


if __name__ == "__main__":
    main()
