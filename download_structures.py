#!/usr/bin/env python3
"""
Download and prepare MDK structures from PDB
"""

import os
import requests
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ChainSelect(Select):
    """Select specific chain from PDB"""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id

def download_pdb(pdb_id, output_dir):
    """Download PDB file from RCSB"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = Path(output_dir) / f"{pdb_id}.pdb"
    
    if output_path.exists():
        print(f"  {pdb_id}.pdb already exists, skipping download")
        return output_path
    
    print(f"  Downloading {pdb_id}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"  Saved to {output_path}")
        return output_path
    else:
        print(f"  Failed to download {pdb_id}")
        return None

def combine_domains(n_term_pdb, c_term_pdb, output_path):
    """
    Combine N-terminal and C-terminal domains into full-length model
    Add a flexible linker between domains
    """
    parser = PDBParser(QUIET=True)
    
    # Load structures
    n_structure = parser.get_structure("N_term", n_term_pdb)
    c_structure = parser.get_structure("C_term", c_term_pdb)
    
    # Create new structure for full-length
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    
    full_structure = Structure("MDK_full")
    model = Model(0)
    chain = Chain("A")
    
    # Add N-terminal domain residues
    residue_counter = 1
    n_chain = list(n_structure.get_chains())[0]
    last_n_coord = None
    
    for residue in n_chain.get_residues():
        if residue.id[0] == ' ':  # Skip heteroatoms
            new_residue = residue.copy()
            new_residue.id = (' ', residue_counter, ' ')
            chain.add(new_residue)
            residue_counter += 1
            # Get last CA coordinate for linker
            if 'CA' in residue:
                last_n_coord = residue['CA'].coord.copy()
    
    # Add flexible linker (GGGGS)
    # This is simplified - in reality you'd model this properly
    linker_sequence = "GGGGS"
    linker_start = residue_counter
    
    # Add C-terminal domain residues
    c_chain = list(c_structure.get_chains())[0]
    first_c_coord = None
    
    # Get first CA coordinate of C-terminal
    for residue in c_chain.get_residues():
        if residue.id[0] == ' ' and 'CA' in residue:
            first_c_coord = residue['CA'].coord.copy()
            break
    
    # Translate C-terminal domain to position after linker
    if last_n_coord is not None and first_c_coord is not None:
        translation = last_n_coord + np.array([10, 0, 0]) - first_c_coord
        
        for residue in c_chain.get_residues():
            if residue.id[0] == ' ':
                new_residue = residue.copy()
                new_residue.id = (' ', residue_counter + 5, ' ')  # +5 for linker
                # Apply translation
                for atom in new_residue.get_atoms():
                    atom.coord += translation
                chain.add(new_residue)
                residue_counter += 1
    
    model.add(chain)
    full_structure.add(model)
    
    # Save combined structure
    io = PDBIO()
    io.set_structure(full_structure)
    io.save(str(output_path))
    print(f"  Created full-length model: {output_path}")
    
    return output_path

def prepare_mdk_structures():
    """Main function to download and prepare MDK structures"""
    
    print("=" * 60)
    print("MDK Structure Preparation Pipeline")
    print("=" * 60)
    
    # Create directories
    structure_dir = Path("data/structures")
    structure_dir.mkdir(parents=True, exist_ok=True)
    
    # Download human MDK domains
    print("\n1. Downloading human MDK structures from PDB...")
    
    structures = {
        "1mkc": "C-terminal domain (LRP1 binding region)",
        "1mkn": "N-terminal domain"
    }
    
    downloaded = {}
    for pdb_id, description in structures.items():
        print(f"\n  {pdb_id}: {description}")
        pdb_path = download_pdb(pdb_id, structure_dir)
        if pdb_path:
            downloaded[pdb_id] = pdb_path
    
    # Create full-length model
    print("\n2. Creating full-length MDK model...")
    
    if "1mkn" in downloaded and "1mkc" in downloaded:
        full_model_path = structure_dir / "mdk_full.pdb"
        combine_domains(
            downloaded["1mkn"],
            downloaded["1mkc"],
            full_model_path
        )
    
    # Download nanobody template
    print("\n3. Downloading nanobody template...")
    template_dir = Path("data/templates")
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Using a well-characterized nanobody structure as template
    nanobody_pdb = "5vag"  # Anti-GFP nanobody
    print(f"  Using {nanobody_pdb} as nanobody template")
    nanobody_path = download_pdb(nanobody_pdb, template_dir)
    
    if nanobody_path:
        # Extract just the VHH chain
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("nanobody", nanobody_path)
        io = PDBIO()
        io.set_structure(structure)
        
        output_path = template_dir / "nanobody_template.pdb"
        io.save(str(output_path), ChainSelect("H"))
        print(f"  Extracted VHH chain to {output_path}")
    
    print("\n" + "=" * 60)
    print("Structure preparation complete!")
    print("=" * 60)
    
    # Summary
    print("\nAvailable structures:")
    for pdb_file in structure_dir.glob("*.pdb"):
        print(f"  - {pdb_file.name}")
    
    return downloaded

if __name__ == "__main__":
    prepare_mdk_structures()