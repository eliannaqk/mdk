#!/usr/bin/env python3
"""Stage, run, and analyse AlphaFold3 complexes for top nanobody designs."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder

REQUIRED_LRP1_RESIDUES = {
    "Ser127": 127,
}
KEY_MDK_RESIDUES = {
    "Lys86": 86,
    "Lys87": 87,
    "Arg89": 89,
    "Arg99": 99,
    "Lys102": 102,
    "Arg28": 28,
}


@dataclass
class NanobodyRecord:
    design_id: str
    sequence: str
    original_scores: Dict[str, float]
    final_score: Optional[float]


@dataclass
class JobPaths:
    job_name: str
    input_json: Path
    output_dir: Path
    analysis_dir: Path
    chain_ids: Dict[str, str]


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_campaign_root(config: Dict) -> Path:
    base = Path(config.get("output", {}).get("base_dir", "results"))
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    campaign = config.get("output", {}).get("campaign", "default")
    return base / campaign


def load_nanobodies(root: Path, top_k: int) -> List[NanobodyRecord]:
    ranked_path = root / "validation" / "ranked_nanobodies.json"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing ranked nanobody file: {ranked_path}")
    payload = json.loads(ranked_path.read_text())
    records: List[NanobodyRecord] = []
    for entry in payload[:top_k]:
        records.append(
            NanobodyRecord(
                design_id=entry["id"],
                sequence=entry["sequence"],
                original_scores=entry.get("original_scores", {}),
                final_score=entry.get("final_score"),
            )
        )
    if not records:
        raise RuntimeError("No nanobodies available for AF3 staging.")
    return records


def extract_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    model = structure[0]
    if chain_id not in model:
        raise KeyError(f"Chain {chain_id} absent from {pdb_path}")
    builder = PPBuilder()
    parts = [str(pp.get_sequence()) for pp in builder.build_peptides(model[chain_id])]
    if not parts:
        raise ValueError(f"No peptide sequence recovered for chain {chain_id} in {pdb_path}")
    return "".join(parts)


def load_fasta(path: Path) -> Tuple[str, str]:
    header = None
    seq: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:]
            else:
                seq.append(line)
    if header is None:
        raise ValueError(f"No records found in FASTA: {path}")
    return header, "".join(seq)


def make_protein_entry(chain_id: str, sequence: str) -> Dict:
    return {
        "protein": {
            "id": chain_id,
            "sequence": sequence,
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": [],
        }
    }


def build_af3_payload(
    job_name: str,
    lrp1_seq: Optional[str],
    mdk_seq: str,
    nb_seq: str,
    seeds: Sequence[int],
) -> Dict:
    sequences: List[Dict] = []
    chain_iter = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chain_ids: Dict[str, str] = {}

    def add_entry(label: str, sequence: str) -> None:
        chain_id = next(chain_iter)
        chain_ids[label] = chain_id
        sequences.append(make_protein_entry(chain_id, sequence))

    if lrp1_seq:
        add_entry("LRP1", lrp1_seq)
    add_entry("MDK", mdk_seq)
    add_entry("NB", nb_seq)

    payload = {
        "name": job_name,
        "sequences": sequences,
        "modelSeeds": list(seeds),
        "dialect": "alphafold3",
        "version": 1,
    }
    return payload, chain_ids


def stage_job(
    af3_root: Path,
    results_root: Path,
    record: NanobodyRecord,
    lrp1_seq: Optional[str],
    mdk_seq: str,
    seeds: Sequence[int],
    job_prefix: str,
) -> JobPaths:
    job_name = f"{job_prefix}_{record.design_id}"
    input_dir = af3_root / "input"
    output_dir = af3_root / "output" / job_name
    analysis_dir = results_root / "af3" / job_name
    input_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    payload, chain_ids = build_af3_payload(job_name, lrp1_seq, mdk_seq, record.sequence, seeds)
    input_json = input_dir / f"{job_name}.json"
    input_json.write_text(json.dumps(payload, indent=2))
    meta = {
        "design_id": record.design_id,
        "job_name": job_name,
        "nanobody_length": len(record.sequence),
        "model_seeds": list(seeds),
        "lrp1_included": bool(lrp1_seq),
        "chain_ids": chain_ids,
    }
    (analysis_dir / "job_metadata.json").write_text(json.dumps(meta, indent=2))
    return JobPaths(
        job_name=job_name,
        input_json=input_json,
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        chain_ids=chain_ids,
    )


def run_af3_job(af3_root: Path, job: JobPaths, log: bool = True) -> None:
    src_root = af3_root / "alphafold3_src"
    if not src_root.exists():
        raise FileNotFoundError(f"AlphaFold3 source directory not found: {src_root}")
    model_dir = Path("/home/eqk3/project_pi_mg269/eqk3/alphafold3")
    model_bin = model_dir / "af3.bin"
    if not model_bin.exists():
        raise FileNotFoundError(f"AlphaFold3 weights missing: {model_bin}")
    job.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "run_alphafold.py",
        f"--json_path={job.input_json}",
        f"--model_dir={model_dir}",
        f"--output_dir={job.output_dir}",
        "--run_data_pipeline=false",
        "--run_inference=true",
    ]
    if log:
        print("[af3]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(src_root), check=True)


def _build_sequence_map(chain: PDB.Chain.Chain) -> str:
    builder = PPBuilder()
    return "".join(str(pp.get_sequence()) for pp in builder.build_peptides(chain))


def assign_chains(structure: PDB.Structure.Structure, sequences: Dict[str, str]) -> Dict[str, PDB.Chain.Chain]:
    mapping: Dict[str, PDB.Chain.Chain] = {}
    for chain in structure[0]:
        seq = _build_sequence_map(chain)
        for label, target_seq in sequences.items():
            if label in mapping:
                continue
            if seq == target_seq:
                mapping[label] = chain
                break
    return mapping


def residue_by_index(chain: PDB.Chain.Chain, index_1based: int) -> Optional[PDB.Residue.Residue]:
    residues = [res for res in chain.get_residues() if res.id[0] == " "]
    if not 1 <= index_1based <= len(residues):
        return None
    return residues[index_1based - 1]


def min_distance(res_a: PDB.Residue.Residue, res_b: PDB.Residue.Residue) -> float:
    coords_a = [atom.coord for atom in res_a.get_atoms() if atom.element != "H"]
    coords_b = [atom.coord for atom in res_b.get_atoms() if atom.element != "H"]
    if not coords_a or not coords_b:
        return float("nan")
    import numpy as np

    arr_a = np.vstack(coords_a)
    arr_b = np.vstack(coords_b)
    dists = np.linalg.norm(arr_a[:, None, :] - arr_b[None, :, :], axis=2)
    return float(dists.min())


def load_structure(path: Path) -> PDB.Structure.Structure:
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def find_prediction(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates: List[Path] = []
    for ext in ("*.pdb", "*.cif", "*.mmcif"):
        candidates.extend(output_dir.glob(ext))
        candidates.extend(output_dir.rglob(ext))
    if not candidates:
        return None
    return sorted(candidates)[0]


def analyse_job(
    job: JobPaths,
    mdk_seq: str,
    nb_seq: str,
    lrp1_seq: Optional[str],
    distance_threshold: float = 5.0,
) -> Dict[str, object]:
    prediction = find_prediction(job.output_dir)
    if prediction is None:
        raise FileNotFoundError(f"No AF3 prediction found in {job.output_dir}")
    structure = load_structure(prediction)
    sequences: Dict[str, str] = {"MDK": mdk_seq, "NB": nb_seq}
    if lrp1_seq:
        sequences["LRP1"] = lrp1_seq
    chain_map = assign_chains(structure, sequences)
    missing = [label for label in sequences if label not in chain_map]
    if missing:
        raise RuntimeError(f"Could not map AF3 chains for: {', '.join(missing)}")
    report: Dict[str, object] = {
        "prediction_path": str(prediction),
        "distances": {},
    }
    mdk_chain = chain_map["MDK"]
    nb_chain = chain_map["NB"]
    if "LRP1" in chain_map:
        lrp_chain = chain_map["LRP1"]
        ser127 = residue_by_index(lrp_chain, REQUIRED_LRP1_RESIDUES["Ser127"])
        arg28 = residue_by_index(mdk_chain, KEY_MDK_RESIDUES["Arg28"])
        if ser127 is not None and arg28 is not None:
            report["distances"]["Ser127:LRP1 vs Arg28:MDK"] = min_distance(ser127, arg28)
        else:
            report["distances"]["Ser127:LRP1 vs Arg28:MDK"] = None
    nb_contacts: Dict[str, float] = {}
    for label, idx in KEY_MDK_RESIDUES.items():
        res = residue_by_index(mdk_chain, idx)
        if res is None:
            nb_contacts[label] = None
            continue
        best = float("inf")
        for nb_res in nb_chain.get_residues():
            if nb_res.id[0] != " ":
                continue
            dist = min_distance(res, nb_res)
            if dist < best:
                best = dist
        nb_contacts[label] = best
    report["mdk_nb_contacts"] = nb_contacts
    report["mdk_nb_contact_flags"] = {
        label: (dist is not None and dist <= distance_threshold)
        for label, dist in nb_contacts.items()
    }
    job.analysis_dir.mkdir(parents=True, exist_ok=True)
    (job.analysis_dir / "af3_contact_report.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaFold3 validation helper for MDK nanobodies")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--af3-root", type=Path, default=Path("/home/eqk3/crispr12/opencrispr-repro-main/alphafold3_work"))
    parser.add_argument("--lrp1-fasta", type=Path, default=None, help="Optional FASTA with LRP1 fragment (AlphaFold numbering)")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--job-prefix", type=str, default="mdk_nb")
    parser.add_argument("--run-af3", action="store_true", help="Launch AlphaFold3 inference after staging")
    parser.add_argument("--analyse", action="store_true", help="Analyse AF3 outputs for residue contacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    campaign_root = get_campaign_root(config)
    nanobodies = load_nanobodies(campaign_root, args.top_k)
    target_cfg = config.get("target", {})
    mdk_chain = target_cfg.get("chain", "A").strip() or "A"
    mdk_seq = extract_sequence_from_pdb(Path(target_cfg["pdb_file"]).resolve(), mdk_chain)
    lrp1_seq: Optional[str] = None
    if args.lrp1_fasta:
        _, lrp1_seq = load_fasta(args.lrp1_fasta)
    jobs: List[Tuple[NanobodyRecord, JobPaths]] = []
    for record in nanobodies:
        job = stage_job(
            args.af3_root,
            campaign_root,
            record,
            lrp1_seq,
            mdk_seq,
            args.seeds,
            args.job_prefix,
        )
        jobs.append((record, job))
        print(f"[stage] Prepared {job.job_name} -> {job.input_json}")
    if args.run_af3:
        for record, job in jobs:
            print(f"[run] {job.job_name}")
            run_af3_job(args.af3_root, job)
    if args.analyse:
        from math import isnan

        for record, job in jobs:
            print(f"[analyse] {job.job_name}")
            report = analyse_job(job, mdk_seq, record.sequence, lrp1_seq)
            summary = {
                key: ("NA" if val is None or (isinstance(val, float) and isnan(val)) else f"{val:.2f}")
                for key, val in report["distances"].items()
            }
            print("    distances:", summary)


if __name__ == "__main__":
    main()
