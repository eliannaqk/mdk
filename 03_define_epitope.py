#!/usr/bin/env python3
"""
Define and visualize the LRP1 binding epitope on MDK
"""

import json
import yaml
import numpy as np
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_config():
    """Load configuration from YAML file"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def get_surface_residues(structure, neighbor_radius=5.0, neighbor_threshold=30):
    """
    Identify surface-exposed residues by counting nearby atoms within a radius.
    """
    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)

    surface_residues = []
    for residue in structure.get_residues():
        if residue.id[0] != ' ':  # Skip heteroatoms
            continue

        min_neighbors = min(len(ns.search(atom.coord, neighbor_radius)) for atom in residue.get_atoms())
        if min_neighbors <= neighbor_threshold:
            surface_residues.append(residue.id[1])

    return surface_residues


def define_lrp1_epitope(pdb_file):
    """
    Define the LRP1 binding epitope on MDK C-terminal domain
    Based on literature: basic patches in C-domain
    """
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("MDK", pdb_file)
    
    # Get first model and chain
    model = structure[0]
    chain = list(model.get_chains())[0]
    
    # Define epitope based on basic residues in C-terminal domain
    # These are conserved basic patches likely involved in LRP1 binding
    epitope_residues = []
    basic_residues = {'ARG', 'LYS', 'HIS'}
    
    # Get surface residues
    surface_res = get_surface_residues(structure)
    
    # Identify basic surface residues in C-domain
    for residue in chain.get_residues():
        if residue.id[0] != ' ':
            continue
        
        res_num = residue.id[1]
        res_name = residue.get_resname()
        
        # Focus on C-terminal domain (roughly residues 80-121 in 1mkc)
        # and surface-exposed basic residues
        if res_num >= 80 and res_num <= 121:
            if res_name in basic_residues and res_num in surface_res:
                epitope_residues.append({
                    'number': res_num,
                    'name': res_name,
                    'chain': chain.id
                })
    
    # Define epitope patches based on spatial clustering
    epitope_patches = cluster_residues(chain, epitope_residues)
    
    return epitope_residues, epitope_patches

def cluster_residues(chain, epitope_residues, distance_cutoff=10.0):
    """
    Cluster epitope residues into patches based on spatial proximity
    """
    if not epitope_residues:
        return []
    
    # Get CA coordinates
    coords = []
    res_nums = []
    for res_info in epitope_residues:
        try:
            residue = chain[res_info['number']]
            if 'CA' in residue:
                coords.append(residue['CA'].coord)
                res_nums.append(res_info['number'])
        except:
            continue
    
    if not coords:
        return []
    
    coords = np.array(coords)
    
    # Simple clustering based on distance
    patches = []
    used = set()
    
    for i, res_num in enumerate(res_nums):
        if res_num in used:
            continue
            
        patch = [res_num]
        used.add(res_num)
        
        # Find nearby residues
        for j, other_res in enumerate(res_nums):
            if other_res not in used:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < distance_cutoff:
                    patch.append(other_res)
                    used.add(other_res)
        
        patches.append(patch)
    
    return patches

def expand_epitope(structure, core_residues, expand_radius=5.0):
    """
    Expand epitope definition to include neighboring residues
    """
    parser = PDBParser(QUIET=True)
    
    # Get all atoms from core epitope residues
    epitope_atoms = []
    chain = list(structure[0].get_chains())[0]
    
    for res_num in core_residues:
        try:
            residue = chain[res_num]
            epitope_atoms.extend(list(residue.get_atoms()))
        except:
            continue
    
    # Find neighboring residues
    all_atoms = list(structure.get_atoms())
    ns = NeighborSearch(all_atoms)
    
    expanded_residues = set(core_residues)
    
    for atom in epitope_atoms:
        neighbors = ns.search(atom.coord, expand_radius)
        for neighbor in neighbors:
            res = neighbor.get_parent()
            if res.id[0] == ' ':  # Regular residue
                expanded_residues.add(res.id[1])
    
    return sorted(list(expanded_residues))

def visualize_epitope(pdb_file, epitope_residues, output_file):
    """
    Create PyMOL script to visualize epitope
    """
    
    pymol_script = f"""# PyMOL script to visualize LRP1 binding epitope on MDK
    
# Load structure
load {pdb_file}, MDK

# Set initial view
hide everything
show cartoon, MDK
color grey80, MDK

# Highlight epitope residues
select lrp1_epitope, MDK and resi {'+'.join(map(str, epitope_residues))}
show sticks, lrp1_epitope
color red, lrp1_epitope

# Label key residues
select key_basic, MDK and resn ARG+LYS and resi {'+'.join(map(str, epitope_residues[:5]))}
label key_basic and name CA, "%s-%s" % (resn, resi)

# Create surface representation
show surface, MDK
set transparency, 0.5, MDK

# Color surface by epitope
color salmon, lrp1_epitope

# Set view
zoom MDK
orient

# Ray trace for publication quality
bg_color white
set ray_shadows, 0
# ray 1200, 1200

# Save session
save {str(output_file).replace('.pml', '.pse')}
"""
    
    with open(output_file, 'w') as f:
        f.write(pymol_script)
    
    print(f"  Created PyMOL visualization script: {output_file}")

def save_epitope_definition(epitope_data, output_file):
    """Save epitope definition to JSON file"""
    
    with open(output_file, 'w') as f:
        json.dump(epitope_data, f, indent=2)
    
    print(f"  Saved epitope definition to: {output_file}")

def main():
    """Main epitope definition pipeline"""
    
    print("=" * 60)
    print("LRP1 Binding Epitope Definition")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Parse MDK structure
    pdb_file = Path(config['target']['pdb_file'])
    if not pdb_file.exists():
        print(f"Error: PDB file {pdb_file} not found!")
        print("Please run 02_download_structures.py first")
        return
    
    print(f"\n1. Analyzing MDK structure: {pdb_file}")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("MDK", pdb_file)
    
    # Define core epitope residues
    print("\n2. Identifying LRP1 binding epitope...")
    epitope_residues, epitope_patches = define_lrp1_epitope(pdb_file)
    
    print(f"  Found {len(epitope_residues)} basic surface residues")
    print(f"  Clustered into {len(epitope_patches)} patches")
    
    # Get core residue numbers
    core_residues = [r['number'] for r in epitope_residues]
    
    # Expand epitope
    print("\n3. Expanding epitope definition...")
    expand_radius = config['epitope'].get('expand_radius', 5.0)
    expanded_residues = expand_epitope(structure, core_residues, expand_radius)
    
    print(f"  Core epitope: {len(core_residues)} residues")
    print(f"  Expanded epitope: {len(expanded_residues)} residues")
    
    # Prepare epitope data
    epitope_data = {
        'target': config['target']['name'],
        'epitope_name': 'LRP1_binding_patch',
        'description': 'Predicted LRP1 binding surface on MDK C-terminal domain',
        'core_residues': core_residues,
        'expanded_residues': expanded_residues,
        'patches': epitope_patches,
        'basic_residues': [r for r in epitope_residues],
        'expand_radius': expand_radius,
        'surface_area_estimate': len(expanded_residues) * 50,  # Rough estimate
    }
    
    # Save epitope definition
    print("\n4. Saving epitope definition...")
    epitope_dir = Path("data/epitopes")
    epitope_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = epitope_dir / "lrp1_patch.json"
    save_epitope_definition(epitope_data, output_json)
    
    # Create visualization
    print("\n5. Creating visualization...")
    vis_dir = Path("results/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    pymol_script = vis_dir / "lrp1_epitope.pml"
    visualize_epitope(pdb_file, expanded_residues, pymol_script)
    
    # Summary
    print("\n" + "=" * 60)
    print("Epitope Definition Summary")
    print("=" * 60)
    
    print(f"\nLRP1 Binding Epitope:")
    print(f"  Core residues: {core_residues[:10]}{'...' if len(core_residues) > 10 else ''}")
    print(f"  Total epitope size: {len(expanded_residues)} residues")
    print(f"  Estimated surface area: ~{epitope_data['surface_area_estimate']} Ų")
    
    print(f"\nKey basic residues for LRP1 binding:")
    for res in epitope_residues[:5]:
        print(f"  - {res['name']}{res['number']}")
    
    print(f"\nOutput files:")
    print(f"  - Epitope definition: {output_json}")
    print(f"  - PyMOL script: {pymol_script}")
    
    print("\nNext step: Run 04_run_bindcraft.py to design binders")

if __name__ == "__main__":
    main()