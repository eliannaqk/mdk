"""
Structural analysis utilities for nanobody design
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Superimposer, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1


def three_to_one(res_name: str) -> str:
    """Convert three-letter residue name to one-letter code (fallback to X)."""
    return protein_letters_3to1.get(res_name.upper(), 'X')
from Bio.PDB.DSSP import DSSP
import warnings
warnings.filterwarnings('ignore')

def calculate_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two coordinates
    
    Args:
        coord1: First coordinate
        coord2: Second coordinate
    
    Returns:
        Distance in Angstroms
    """
    return np.linalg.norm(coord1 - coord2)

def calculate_rmsd(
    structure1: PDB.Structure.Structure,
    structure2: PDB.Structure.Structure,
    atom_type: str = 'CA'
) -> float:
    """
    Calculate RMSD between two structures
    
    Args:
        structure1: First structure
        structure2: Second structure
        atom_type: Atom type to use for alignment (default: CA)
    
    Returns:
        RMSD value in Angstroms
    """
    
    # Extract atoms
    atoms1 = []
    atoms2 = []
    
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for res1, res2 in zip(chain1, chain2):
                if is_aa(res1) and is_aa(res2):
                    if atom_type in res1 and atom_type in res2:
                        atoms1.append(res1[atom_type])
                        atoms2.append(res2[atom_type])
    
    if not atoms1 or not atoms2:
        return float('inf')
    
    # Align and calculate RMSD
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    
    return super_imposer.rms

def get_surface_residues(
    structure: PDB.Structure.Structure,
    chain_id: str,
    sasa_threshold: float = 20.0,
    probe_radius: float = 1.4
) -> List[int]:
    """
    Identify surface-exposed residues using SASA
    
    Args:
        structure: PDB structure
        chain_id: Chain identifier
        sasa_threshold: Minimum SASA to consider surface-exposed (Ų)
        probe_radius: Probe radius for SASA calculation (Å)
    
    Returns:
        List of surface residue numbers
    """
    
    try:
        # Try to use DSSP for accurate SASA
        model = structure[0]
        dssp = DSSP(model, structure, dssp='dssp', acc_array='Wilke')
        
        surface_residues = []
        
        for key, value in dssp.property_dict.items():
            chain, res_id = key
            if chain == chain_id:
                sasa = value[3]  # Accessible surface area
                if sasa >= sasa_threshold:
                    surface_residues.append(res_id[1])
        
        return sorted(surface_residues)
        
    except:
        # Fallback to neighbor-based method
        return _get_surface_residues_neighbor(structure, chain_id)

def _get_surface_residues_neighbor(
    structure: PDB.Structure.Structure,
    chain_id: str,
    cutoff: float = 12.0
) -> List[int]:
    """
    Fallback method to identify surface residues using neighbor count
    
    Args:
        structure: PDB structure
        chain_id: Chain identifier
        cutoff: Distance cutoff for neighbor search
    
    Returns:
        List of surface residue numbers
    """
    
    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)
    
    surface_residues = []
    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if not is_aa(residue):
                        continue
                    
                    # Check if residue is on surface
                    is_surface = False
                    for atom in residue:
                        neighbors = ns.search(atom.coord, cutoff)
                        # Fewer neighbors suggests surface exposure
                        if len(neighbors) < 20:
                            is_surface = True
                            break
                    
                    if is_surface:
                        surface_residues.append(residue.id[1])
    
    return sorted(surface_residues)

def extract_interface_residues(
    complex_structure: PDB.Structure.Structure,
    chain1_id: str,
    chain2_id: str,
    distance_cutoff: float = 5.0
) -> Tuple[List[int], List[int]]:
    """
    Extract interface residues between two chains
    
    Args:
        complex_structure: Complex structure containing both chains
        chain1_id: First chain identifier
        chain2_id: Second chain identifier
        distance_cutoff: Distance cutoff for interface (Å)
    
    Returns:
        Tuple of (chain1_interface_residues, chain2_interface_residues)
    """
    
    model = complex_structure[0]
    
    if chain1_id not in [c.id for c in model]:
        raise ValueError(f"Chain {chain1_id} not found")
    if chain2_id not in [c.id for c in model]:
        raise ValueError(f"Chain {chain2_id} not found")
    
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]
    
    # Get atoms from each chain
    chain1_atoms = [atom for residue in chain1 if is_aa(residue) for atom in residue]
    chain2_atoms = [atom for residue in chain2 if is_aa(residue) for atom in residue]
    
    # Build neighbor search
    ns1 = NeighborSearch(chain1_atoms)
    ns2 = NeighborSearch(chain2_atoms)
    
    interface1 = set()
    interface2 = set()
    
    # Find interface residues in chain1
    for atom in chain2_atoms:
        close_atoms = ns1.search(atom.coord, distance_cutoff)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            if is_aa(residue):
                interface1.add(residue.id[1])
    
    # Find interface residues in chain2
    for atom in chain1_atoms:
        close_atoms = ns2.search(atom.coord, distance_cutoff)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            if is_aa(residue):
                interface2.add(residue.id[1])
    
    return sorted(list(interface1)), sorted(list(interface2))

def align_structures(
    mobile: PDB.Structure.Structure,
    target: PDB.Structure.Structure,
    mobile_chain: str = 'A',
    target_chain: str = 'A'
) -> Tuple[PDB.Structure.Structure, float]:
    """
    Align mobile structure to target structure
    
    Args:
        mobile: Structure to align
        target: Target structure
        mobile_chain: Chain ID in mobile structure
        target_chain: Chain ID in target structure
    
    Returns:
        Tuple of (aligned structure, RMSD)
    """
    
    # Extract CA atoms for alignment
    mobile_atoms = []
    target_atoms = []
    
    for res_m in mobile[0][mobile_chain]:
        if is_aa(res_m) and 'CA' in res_m:
            res_id = res_m.id[1]
            
            # Find corresponding residue in target
            try:
                res_t = target[0][target_chain][res_id]
                if is_aa(res_t) and 'CA' in res_t:
                    mobile_atoms.append(res_m['CA'])
                    target_atoms.append(res_t['CA'])
            except:
                continue
    
    if len(mobile_atoms) < 3:
        raise ValueError("Not enough atoms for alignment")
    
    # Perform alignment
    super_imposer = Superimposer()
    super_imposer.set_atoms(target_atoms, mobile_atoms)
    super_imposer.apply(mobile.get_atoms())
    
    return mobile, super_imposer.rms

def save_complex_structure(
    receptor: PDB.Structure.Structure,
    ligand: PDB.Structure.Structure,
    output_file: Path,
    receptor_chain: str = 'A',
    ligand_chain: str = 'B'
):
    """
    Save receptor-ligand complex as PDB file
    
    Args:
        receptor: Receptor structure
        ligand: Ligand structure
        output_file: Output PDB file path
        receptor_chain: Chain ID for receptor
        ligand_chain: Chain ID for ligand
    """
    
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    
    # Create new structure
    complex_structure = Structure('complex')
    model = Model(0)
    
    # Add receptor chain
    receptor_chain_obj = Chain(receptor_chain)
    for residue in receptor[0][receptor_chain]:
        receptor_chain_obj.add(residue.copy())
    model.add(receptor_chain_obj)
    
    # Add ligand chain
    ligand_chain_obj = Chain(ligand_chain)
    for residue in ligand[0][ligand_chain]:
        ligand_chain_obj.add(residue.copy())
    model.add(ligand_chain_obj)
    
    complex_structure.add(model)
    
    # Save structure
    io = PDBIO()
    io.set_structure(complex_structure)
    io.save(str(output_file))

def calculate_sasa(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None
) -> Dict[int, float]:
    """
    Calculate solvent accessible surface area for each residue
    
    Args:
        structure: PDB structure
        chain_id: Optional chain identifier
    
    Returns:
        Dictionary mapping residue number to SASA
    """
    
    try:
        model = structure[0]
        dssp = DSSP(model, structure, dssp='dssp')
        
        sasa_dict = {}
        
        for key, value in dssp.property_dict.items():
            chain, res_id = key
            if chain_id is None or chain == chain_id:
                sasa = value[3]
                sasa_dict[res_id[1]] = sasa
        
        return sasa_dict
        
    except:
        # Fallback to approximate method
        return _calculate_sasa_approximate(structure, chain_id)

def _calculate_sasa_approximate(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None
) -> Dict[int, float]:
    """
    Approximate SASA calculation using neighbor count
    
    Args:
        structure: PDB structure
        chain_id: Optional chain identifier
    
    Returns:
        Dictionary mapping residue number to approximate SASA
    """
    
    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)
    
    sasa_dict = {}
    
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                for residue in chain:
                    if not is_aa(residue):
                        continue
                    
                    # Count neighbors as proxy for burial
                    neighbor_count = 0
                    for atom in residue:
                        neighbors = ns.search(atom.coord, 5.0)
                        neighbor_count += len(neighbors)
                    
                    # Approximate SASA (inverse of burial)
                    approx_sasa = max(0, 200 - neighbor_count * 2)
                    sasa_dict[residue.id[1]] = approx_sasa
    
    return sasa_dict

def get_residue_contacts(
    structure: PDB.Structure.Structure,
    residue_num: int,
    chain_id: str = 'A',
    distance_cutoff: float = 5.0
) -> List[Tuple[str, int]]:
    """
    Get residues in contact with a specific residue
    
    Args:
        structure: PDB structure
        residue_num: Residue number
        chain_id: Chain identifier
        distance_cutoff: Distance cutoff for contacts (Å)
    
    Returns:
        List of (chain_id, residue_num) tuples for contacting residues
    """
    
    model = structure[0]
    
    try:
        target_residue = model[chain_id][residue_num]
    except:
        return []
    
    if not is_aa(target_residue):
        return []
    
    # Get all atoms except from target residue
    other_atoms = []
    for chain in model:
        for residue in chain:
            if not (chain.id == chain_id and residue.id[1] == residue_num):
                if is_aa(residue):
                    other_atoms.extend(list(residue.get_atoms()))
    
    ns = NeighborSearch(other_atoms)
    
    contacts = set()
    
    for atom in target_residue:
        nearby = ns.search(atom.coord, distance_cutoff)
        for nearby_atom in nearby:
            res = nearby_atom.get_parent()
            if is_aa(res):
                contacts.add((res.get_parent().id, res.id[1]))
    
    return sorted(list(contacts))

def find_hydrogen_bonds(
    structure: PDB.Structure.Structure,
    chain1_id: str,
    chain2_id: str,
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 120.0
) -> List[Dict]:
    """
    Find potential hydrogen bonds between chains
    
    Args:
        structure: PDB structure
        chain1_id: First chain identifier
        chain2_id: Second chain identifier
        distance_cutoff: Maximum donor-acceptor distance (Å)
        angle_cutoff: Minimum donor-H-acceptor angle (degrees)
    
    Returns:
        List of hydrogen bond dictionaries
    """
    
    model = structure[0]
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]
    
    # Define donor and acceptor atoms
    donors = ['N', 'O']  # Simplified
    acceptors = ['O', 'N']  # Simplified
    
    h_bonds = []
    
    for res1 in chain1:
        if not is_aa(res1):
            continue
            
        for res2 in chain2:
            if not is_aa(res2):
                continue
            
            for atom1 in res1:
                if atom1.element in donors:
                    for atom2 in res2:
                        if atom2.element in acceptors:
                            distance = calculate_distance(atom1.coord, atom2.coord)
                            
                            if distance <= distance_cutoff:
                                h_bonds.append({
                                    'donor': (chain1_id, res1.id[1], atom1.name),
                                    'acceptor': (chain2_id, res2.id[1], atom2.name),
                                    'distance': distance
                                })
    
    return h_bonds

def calculate_center_of_mass(
    structure: PDB.Structure.Structure,
    chain_id: Optional[str] = None,
    residue_list: Optional[List[int]] = None
) -> np.ndarray:
    """
    Calculate center of mass for structure or subset
    
    Args:
        structure: PDB structure
        chain_id: Optional chain identifier
        residue_list: Optional list of residue numbers
    
    Returns:
        Center of mass coordinates
    """
    
    coords = []
    masses = []
    
    # Atomic masses
    mass_dict = {
        'H': 1.008, 'C': 12.01, 'N': 14.01,
        'O': 16.00, 'S': 32.07, 'P': 30.97
    }
    
    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                for residue in chain:
                    if not is_aa(residue):
                        continue
                    
                    if residue_list is None or residue.id[1] in residue_list:
                        for atom in residue:
                            coords.append(atom.coord)
                            masses.append(mass_dict.get(atom.element, 12.0))
    
    if not coords:
        return np.array([0, 0, 0])
    
    coords = np.array(coords)
    masses = np.array(masses)
    
    # Calculate weighted center
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    
    return com
