#!/usr/bin/env python3
"""
Convert successful BindCraft designs to nanobody (VHH) format
"""

import json
import yaml
import numpy as np
from pathlib import Path
from Bio import PDB, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_config():
    """Load configuration"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def load_top_designs(results_dir, top_n=10):
    """Load top scoring designs from BindCraft results"""
    
    sequences_file = Path(results_dir) / "sequences.fasta"
    scores_file = Path(results_dir) / "scores.json"
    
    # Load sequences
    sequences = {}
    for record in SeqIO.parse(sequences_file, "fasta"):
        design_id = record.id.split("_")[1]  # Extract design number
        sequences[f"design_{design_id}"] = str(record.seq)
    
    # Load scores
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    
    # Combine and return top N
    designs = []
    for score_data in scores[:top_n]:
        design_id = score_data['design_id']
        if design_id in sequences:
            designs.append({
                'id': design_id,
                'sequence': sequences[design_id],
                'scores': score_data['scores'],
                'composite_score': score_data['composite_score']
            })
    
    return designs

def identify_paratope_residues(design_sequence, interface_threshold=0.8):
    """
    Identify which residues in the design are part of the binding interface
    This would normally use structural analysis of the complex
    """
    
    # For demonstration, we'll mark residues in the middle third as paratope
    # In reality, you'd analyze the AlphaFold complex structure
    
    length = len(design_sequence)
    start = length // 3
    end = 2 * length // 3
    
    paratope_positions = list(range(start, end))
    
    return paratope_positions

def extract_cdr_like_regions(design_sequence, paratope_positions):
    """
    Extract CDR-like regions from the designed binder
    These will be grafted onto the nanobody scaffold
    """
    
    # Identify continuous stretches of paratope residues
    # These will become our CDR replacements
    
    cdrs = []
    current_cdr = []
    
    for i in range(len(design_sequence)):
        if i in paratope_positions:
            current_cdr.append(i)
        elif current_cdr:
            if len(current_cdr) >= 3:  # Minimum CDR length
                cdrs.append({
                    'start': current_cdr[0],
                    'end': current_cdr[-1],
                    'sequence': design_sequence[current_cdr[0]:current_cdr[-1]+1]
                })
            current_cdr = []
    
    # Handle last CDR if exists
    if current_cdr and len(current_cdr) >= 3:
        cdrs.append({
            'start': current_cdr[0],
            'end': current_cdr[-1],
            'sequence': design_sequence[current_cdr[0]:current_cdr[-1]+1]
        })
    
    # Map to canonical CDRs (simplified mapping)
    cdr_mapping = {}
    if len(cdrs) >= 1:
        cdr_mapping['CDR1'] = cdrs[0]['sequence']
    if len(cdrs) >= 2:
        cdr_mapping['CDR2'] = cdrs[1]['sequence']
    if len(cdrs) >= 3:
        cdr_mapping['CDR3'] = cdrs[2]['sequence']
    
    # Use default CDRs for any missing
    default_cdrs = {
        'CDR1': 'GFTFS',
        'CDR2': 'IS',
        'CDR3': 'AR'
    }
    
    for cdr_name in ['CDR1', 'CDR2', 'CDR3']:
        if cdr_name not in cdr_mapping:
            cdr_mapping[cdr_name] = default_cdrs[cdr_name]
    
    return cdr_mapping

def create_nanobody_sequence(framework_regions, cdr_sequences):
    """
    Assemble nanobody sequence from framework regions and CDRs
    """
    
    # Standard nanobody structure:
    # FR1 - CDR1 - FR2 - CDR2 - FR3 - CDR3 - FR4
    
    nanobody_seq = (
        framework_regions['FR1'] +
        cdr_sequences['CDR1'] +
        framework_regions['FR2'] +
        cdr_sequences['CDR2'] +
        framework_regions['FR3'] +
        cdr_sequences['CDR3'] +
        framework_regions['FR4']
    )
    
    return nanobody_seq

def optimize_nanobody_sequence(nanobody_seq):
    """
    Optimize nanobody sequence for stability and developability
    """
    
    # Check for problematic motifs and suggest modifications
    modifications = []
    
    # Check for unpaired cysteines (except framework cysteines)
    cys_positions = [i for i, aa in enumerate(nanobody_seq) if aa == 'C']
    
    # Framework cysteines should be at approximately positions 22 and 92
    expected_cys = [22, 92]
    unexpected_cys = [pos for pos in cys_positions if not any(abs(pos - exp) < 3 for exp in expected_cys)]
    
    if unexpected_cys:
        modifications.append(f"Warning: Unexpected cysteines at positions {unexpected_cys}")
    
    # Check for N-glycosylation sites (NxS/T)
    import re
    glyco_sites = [(m.start(), m.group()) for m in re.finditer(r'N[^P][ST]', nanobody_seq)]
    if glyco_sites:
        modifications.append(f"N-glycosylation sites found: {glyco_sites}")
    
    # Check for hydrophobic patches in CDRs
    hydrophobic_aa = set('FWYLIMV')
    for i in range(len(nanobody_seq) - 4):
        window = nanobody_seq[i:i+5]
        hydrophobic_count = sum(1 for aa in window if aa in hydrophobic_aa)
        if hydrophobic_count >= 4:
            modifications.append(f"Hydrophobic patch at position {i}-{i+4}: {window}")
    
    return modifications

def calculate_nanobody_properties(sequence):
    """
    Calculate physicochemical properties of nanobody
    """
    
    # Basic properties
    length = len(sequence)
    
    # Molecular weight (approximate)
    mw_dict = {
        'A': 71.08, 'R': 156.19, 'N': 114.11, 'D': 115.09,
        'C': 103.14, 'Q': 128.14, 'E': 129.12, 'G': 57.05,
        'H': 137.14, 'I': 113.16, 'L': 113.16, 'K': 128.17,
        'M': 131.19, 'F': 147.18, 'P': 97.12, 'S': 87.08,
        'T': 101.11, 'W': 186.21, 'Y': 163.18, 'V': 99.13
    }
    mw = sum(mw_dict.get(aa, 110) for aa in sequence) / 1000  # in kDa
    
    # Theoretical pI (simplified calculation)
    charged_aa = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
    net_charge = sum(charged_aa.get(aa, 0) for aa in sequence)
    theoretical_pi = 6.5 + net_charge * 0.5  # Very simplified
    
    # Hydrophobicity
    hydrophobic_aa = set('FWYLIMV')
    hydrophobicity = sum(1 for aa in sequence if aa in hydrophobic_aa) / length
    
    properties = {
        'length': length,
        'molecular_weight_kda': round(mw, 1),
        'theoretical_pi': round(theoretical_pi, 1),
        'hydrophobicity': round(hydrophobicity, 3),
        'net_charge': net_charge
    }
    
    return properties

def convert_design_to_nanobody(design, framework_regions):
    """
    Convert a single BindCraft design to nanobody format
    """
    
    print(f"\n  Converting {design['id']}...")
    
    # Identify paratope residues
    paratope = identify_paratope_residues(design['sequence'])
    print(f"    Identified {len(paratope)} paratope residues")
    
    # Extract CDR-like regions
    cdrs = extract_cdr_like_regions(design['sequence'], paratope)
    print(f"    Extracted CDRs: CDR1={len(cdrs['CDR1'])}aa, CDR2={len(cdrs['CDR2'])}aa, CDR3={len(cdrs['CDR3'])}aa")
    
    # Create nanobody sequence
    nanobody_seq = create_nanobody_sequence(framework_regions, cdrs)
    
    # Optimize sequence
    modifications = optimize_nanobody_sequence(nanobody_seq)
    if modifications:
        print(f"    Optimization notes: {len(modifications)} issues identified")
    
    # Calculate properties
    properties = calculate_nanobody_properties(nanobody_seq)
    
    nanobody = {
        'id': f"nb_{design['id']}",
        'original_design': design['id'],
        'sequence': nanobody_seq,
        'cdrs': cdrs,
        'properties': properties,
        'modifications': modifications,
        'original_scores': design['scores']
    }
    
    return nanobody

def save_nanobodies(nanobodies, output_dir):
    """
    Save nanobody sequences and data
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sequences
    sequences_file = output_dir / "nanobody_sequences.fasta"
    records = []
    
    for nb in nanobodies:
        desc = f"MW={nb['properties']['molecular_weight_kda']}kDa pI={nb['properties']['theoretical_pi']}"
        record = SeqRecord(
            Seq(nb['sequence']),
            id=nb['id'],
            description=desc
        )
        records.append(record)
    
    SeqIO.write(records, sequences_file, "fasta")
    print(f"\n  Saved sequences to: {sequences_file}")
    
    # Save detailed data
    data_file = output_dir / "nanobody_data.json"
    
    # Prepare data for JSON serialization
    json_data = []
    for nb in nanobodies:
        json_data.append({
            'id': nb['id'],
            'sequence': nb['sequence'],
            'cdrs': nb['cdrs'],
            'properties': nb['properties'],
            'modifications': nb['modifications'],
            'original_scores': nb['original_scores']
        })
    
    with open(data_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  Saved data to: {data_file}")
    
    return output_dir

def generate_nanobody_report(nanobodies, output_file):
    """
    Generate report on nanobody conversion
    """
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Nanobody Conversion Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Converted {len(nanobodies)} designs to nanobody format\n\n")
        
        # Summary statistics
        mws = [nb['properties']['molecular_weight_kda'] for nb in nanobodies]
        pis = [nb['properties']['theoretical_pi'] for nb in nanobodies]
        
        f.write("Properties summary:\n")
        f.write(f"  MW range: {min(mws):.1f} - {max(mws):.1f} kDa\n")
        f.write(f"  pI range: {min(pis):.1f} - {max(pis):.1f}\n\n")
        
        # Individual nanobodies
        f.write("Individual nanobodies:\n\n")
        
        for nb in nanobodies:
            f.write(f"{nb['id']}:\n")
            f.write(f"  Original design: {nb['original_design']}\n")
            f.write(f"  Length: {nb['properties']['length']} aa\n")
            f.write(f"  MW: {nb['properties']['molecular_weight_kda']} kDa\n")
            f.write(f"  pI: {nb['properties']['theoretical_pi']}\n")
            f.write(f"  CDR1: {nb['cdrs']['CDR1']}\n")
            f.write(f"  CDR2: {nb['cdrs']['CDR2']}\n")
            f.write(f"  CDR3: {nb['cdrs']['CDR3']}\n")
            
            if nb['modifications']:
                f.write(f"  Notes:\n")
                for mod in nb['modifications']:
                    f.write(f"    - {mod}\n")
            
            f.write("\n")
    
    print(f"  Generated report: {output_file}")

def main():
    """Main nanobody conversion pipeline"""
    
    print("=" * 60)
    print("Nanobody Format Conversion")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    framework_regions = config['nanobody']['framework']
    
    # Load top designs from BindCraft
    print("\n1. Loading top BindCraft designs...")
    results_dir = Path(config['output']['base_dir']) / "designs"
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found!")
        print("Please run 04_run_bindcraft.py first")
        return
    
    designs = load_top_designs(results_dir, top_n=10)
    print(f"  Loaded {len(designs)} designs")
    
    # Convert to nanobody format
    print("\n2. Converting designs to nanobody format...")
    
    nanobodies = []
    for design in designs:
        nanobody = convert_design_to_nanobody(design, framework_regions)
        nanobodies.append(nanobody)
    
    print(f"\n  Successfully converted {len(nanobodies)} designs")
    
    # Save results
    print("\n3. Saving nanobody sequences...")
    output_dir = Path(config['output']['base_dir']) / "nanobodies"
    save_nanobodies(nanobodies, output_dir)
    
    # Generate report
    print("\n4. Generating report...")
    report_file = output_dir / "conversion_report.txt"
    generate_nanobody_report(nanobodies, report_file)
    
    print("\n" + "=" * 60)
    print("Nanobody Conversion Complete!")
    print("=" * 60)
    
    print(f"\nResults saved to: {output_dir}")
    
    # Show top candidates
    print("\nTop 3 nanobody candidates:")
    for nb in nanobodies[:3]:
        print(f"  {nb['id']}: MW={nb['properties']['molecular_weight_kda']}kDa, pI={nb['properties']['theoretical_pi']}")
        if not nb['modifications']:
            print("    ✓ No optimization issues detected")
        else:
            print(f"    ⚠ {len(nb['modifications'])} optimization notes")
    
    print(f"\nNext step: Run 06_analyze_results.py for final validation")

if __name__ == "__main__":
    main()