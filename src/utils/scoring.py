"""
Scoring functions for evaluating nanobody-target interactions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from Bio import PDB
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1


def three_to_one(res_name: str) -> str:
    """Convert three-letter residue name to one-letter code (fallback to X)."""
    return protein_letters_3to1.get(res_name.upper(), 'X')
import warnings
warnings.filterwarnings('ignore')

def calculate_interface_score(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str,
    distance_cutoff: float = 5.0
) -> Dict[str, float]:
    """
    Calculate interface quality scores
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
        distance_cutoff: Interface distance cutoff (Å)
    
    Returns:
        Dictionary of interface scores
    """
    
    scores = {}
    
    # Get interface residues
    interface_residues = _get_interface_residues(
        complex_structure,
        receptor_chain,
        ligand_chain,
        distance_cutoff
    )
    
    receptor_interface, ligand_interface = interface_residues
    
    # Interface size
    scores['interface_area'] = len(receptor_interface) * 50  # Approximate
    scores['num_interface_residues'] = len(receptor_interface) + len(ligand_interface)
    
    # Contact density
    num_contacts = _count_interface_contacts(
        complex_structure,
        receptor_chain,
        ligand_chain,
        distance_cutoff
    )
    scores['contact_density'] = num_contacts / max(1, scores['num_interface_residues'])
    
    # Hydrophobicity
    scores['interface_hydrophobicity'] = _calculate_interface_hydrophobicity(
        complex_structure,
        receptor_interface,
        ligand_interface,
        receptor_chain,
        ligand_chain
    )
    
    # Complementarity scores
    scores['shape_complementarity'] = calculate_shape_complementarity(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    scores['electrostatic_complementarity'] = calculate_electrostatic_complementarity(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    return scores

def calculate_shape_complementarity(
    complex_structure: PDB.Structure.Structure,
    chain1: str,
    chain2: str,
    probe_radius: float = 1.4
) -> float:
    """
    Calculate shape complementarity between two chains
    
    Args:
        complex_structure: Complex structure
        chain1: First chain ID
        chain2: Second chain ID
        probe_radius: Probe radius for surface calculation (Å)
    
    Returns:
        Shape complementarity score (0-1, higher is better)
    """
    
    model = complex_structure[0]
    
    # Get interface atoms
    interface1, interface2 = _get_interface_residues(
        complex_structure,
        chain1,
        chain2,
        5.0
    )
    
    if not interface1 or not interface2:
        return 0.0
    
    # Get coordinates of interface atoms
    coords1 = []
    coords2 = []
    
    for res_num in interface1:
        try:
            residue = model[chain1][res_num]
            for atom in residue:
                coords1.append(atom.coord)
        except:
            continue
    
    for res_num in interface2:
        try:
            residue = model[chain2][res_num]
            for atom in residue:
                coords2.append(atom.coord)
        except:
            continue
    
    if not coords1 or not coords2:
        return 0.0
    
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    # Calculate shape complementarity using distance distribution
    min_distances = []
    
    for c1 in coords1:
        distances = np.linalg.norm(coords2 - c1, axis=1)
        min_distances.append(np.min(distances))
    
    # Good complementarity has consistent small distances
    mean_dist = np.mean(min_distances)
    std_dist = np.std(min_distances)
    
    # Score based on how close and uniform the distances are
    if mean_dist > 10:
        score = 0.0
    else:
        closeness_score = np.exp(-mean_dist / 3.0)
        uniformity_score = np.exp(-std_dist / 2.0)
        score = (closeness_score + uniformity_score) / 2
    
    return min(1.0, score)

def calculate_electrostatic_complementarity(
    complex_structure: PDB.Structure.Structure,
    chain1: str,
    chain2: str
) -> float:
    """
    Calculate electrostatic complementarity between chains
    
    Args:
        complex_structure: Complex structure
        chain1: First chain ID
        chain2: Second chain ID
    
    Returns:
        Electrostatic complementarity score (0-1, higher is better)
    """
    
    model = complex_structure[0]
    
    # Get interface residues
    interface1, interface2 = _get_interface_residues(
        complex_structure,
        chain1,
        chain2,
        6.0  # Slightly larger cutoff for electrostatics
    )
    
    if not interface1 or not interface2:
        return 0.5  # Neutral score if no interface
    
    # Classify residues by charge
    positive = {'ARG', 'LYS', 'HIS'}
    negative = {'ASP', 'GLU'}
    
    charges1 = []
    charges2 = []
    
    # Get charges for interface residues
    for res_num in interface1:
        try:
            residue = model[chain1][res_num]
            res_name = residue.get_resname()
            if res_name in positive:
                charges1.append(1)
            elif res_name in negative:
                charges1.append(-1)
            else:
                charges1.append(0)
        except:
            charges1.append(0)
    
    for res_num in interface2:
        try:
            residue = model[chain2][res_num]
            res_name = residue.get_resname()
            if res_name in positive:
                charges2.append(1)
            elif res_name in negative:
                charges2.append(-1)
            else:
                charges2.append(0)
        except:
            charges2.append(0)
    
    # Calculate complementarity
    # Good complementarity has opposite charges
    sum1 = sum(charges1)
    sum2 = sum(charges2)
    
    # If charges balance (opposite), score is high
    charge_balance = 1.0 - abs(sum1 + sum2) / max(abs(sum1), abs(sum2), 1)
    
    # Also check for favorable pairs
    favorable_pairs = 0
    unfavorable_pairs = 0
    
    for c1 in charges1:
        for c2 in charges2:
            if c1 * c2 < 0:  # Opposite charges
                favorable_pairs += 1
            elif c1 * c2 > 0:  # Same charges
                unfavorable_pairs += 1
    
    if favorable_pairs + unfavorable_pairs > 0:
        pair_score = favorable_pairs / (favorable_pairs + unfavorable_pairs)
    else:
        pair_score = 0.5
    
    # Combine scores
    score = (charge_balance + pair_score) / 2
    
    return min(1.0, max(0.0, score))

def score_protein_protein_interaction(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str
) -> Dict[str, float]:
    """
    Comprehensive scoring of protein-protein interaction
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
    
    Returns:
        Dictionary of interaction scores
    """
    
    scores = {}
    
    # Interface scores
    interface_scores = calculate_interface_score(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    scores.update(interface_scores)
    
    # Binding energy estimate
    scores['binding_energy'] = calculate_binding_energy(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Buried surface area
    scores['buried_surface_area'] = calculate_buried_surface_area(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Hotspot residues
    scores['hotspot_score'] = score_hotspot_residues(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Packing density
    scores['packing_density'] = calculate_packing_density(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Overall quality
    scores['complex_quality'] = evaluate_complex_quality(scores)
    
    return scores

def calculate_binding_energy(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str
) -> float:
    """
    Estimate binding energy (simplified)
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
    
    Returns:
        Estimated binding energy (kcal/mol, more negative is better)
    """
    
    # Simplified binding energy calculation
    # Based on buried surface area and interface composition
    
    bsa = calculate_buried_surface_area(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Get interface composition
    interface1, interface2 = _get_interface_residues(
        complex_structure,
        receptor_chain,
        ligand_chain,
        5.0
    )
    
    model = complex_structure[0]
    
    # Count favorable interactions
    h_bonds = _count_hydrogen_bonds(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    salt_bridges = _count_salt_bridges(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Simplified energy calculation
    # Each buried Ų contributes ~-0.01 kcal/mol
    # Each H-bond contributes ~-1 kcal/mol
    # Each salt bridge contributes ~-3 kcal/mol
    
    energy = (
        -0.01 * bsa +
        -1.0 * h_bonds +
        -3.0 * salt_bridges
    )
    
    return energy

def calculate_buried_surface_area(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str
) -> float:
    """
    Calculate buried surface area upon complex formation
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
    
    Returns:
        Buried surface area (Ų)
    """
    
    # Simplified BSA calculation
    # Count interface atoms and multiply by average surface area
    
    interface1, interface2 = _get_interface_residues(
        complex_structure,
        receptor_chain,
        ligand_chain,
        5.0
    )
    
    # Approximate BSA
    # Each interface residue buries ~50-100 Ų
    bsa = (len(interface1) + len(interface2)) * 75
    
    return bsa

def score_hotspot_residues(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str
) -> float:
    """
    Score presence of hotspot residues at interface
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
    
    Returns:
        Hotspot score (0-1, higher is better)
    """
    
    # Hotspot residues typically include Trp, Tyr, Arg
    hotspot_types = {'TRP', 'TYR', 'ARG', 'PHE', 'LEU'}
    
    interface1, interface2 = _get_interface_residues(
        complex_structure,
        receptor_chain,
        ligand_chain,
        5.0
    )
    
    model = complex_structure[0]
    
    hotspot_count = 0
    total_interface = len(interface1) + len(interface2)
    
    # Count hotspot residues in interface
    for res_num in interface1:
        try:
            residue = model[receptor_chain][res_num]
            if residue.get_resname() in hotspot_types:
                hotspot_count += 1
        except:
            continue
    
    for res_num in interface2:
        try:
            residue = model[ligand_chain][res_num]
            if residue.get_resname() in hotspot_types:
                hotspot_count += 1
        except:
            continue
    
    if total_interface == 0:
        return 0.0
    
    # Score based on fraction of hotspot residues
    hotspot_fraction = hotspot_count / total_interface
    
    # Ideal is 20-40% hotspot residues
    if hotspot_fraction < 0.2:
        score = hotspot_fraction / 0.2
    elif hotspot_fraction > 0.4:
        score = 1.0 - (hotspot_fraction - 0.4) / 0.6
    else:
        score = 1.0
    
    return max(0.0, min(1.0, score))

def calculate_packing_density(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str,
    radius: float = 6.0
) -> float:
    """
    Calculate packing density at interface
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
        radius: Radius for density calculation (Å)
    
    Returns:
        Packing density score (0-1, higher is better)
    """
    
    model = complex_structure[0]
    
    # Get interface atoms
    interface_atoms = []
    
    interface1, interface2 = _get_interface_residues(
        complex_structure,
        receptor_chain,
        ligand_chain,
        5.0
    )
    
    for res_num in interface1:
        try:
            residue = model[receptor_chain][res_num]
            interface_atoms.extend(list(residue.get_atoms()))
        except:
            continue
    
    for res_num in interface2:
        try:
            residue = model[ligand_chain][res_num]
            interface_atoms.extend(list(residue.get_atoms()))
        except:
            continue
    
    if len(interface_atoms) < 10:
        return 0.0
    
    # Calculate packing density
    ns = NeighborSearch(interface_atoms)
    
    densities = []
    for atom in interface_atoms:
        neighbors = ns.search(atom.coord, radius)
        density = len(neighbors) / (4/3 * np.pi * radius**3)
        densities.append(density)
    
    # Average density
    avg_density = np.mean(densities)
    
    # Normalize (typical protein density is ~0.01-0.02 atoms/Ų)
    normalized_density = min(1.0, avg_density / 0.015)
    
    return normalized_density

def evaluate_complex_quality(scores: Dict[str, float]) -> float:
    """
    Evaluate overall complex quality from individual scores
    
    Args:
        scores: Dictionary of individual scores
    
    Returns:
        Overall quality score (0-1)
    """
    
    # Weight different components
    weights = {
        'interface_area': 0.15,
        'shape_complementarity': 0.20,
        'electrostatic_complementarity': 0.15,
        'contact_density': 0.10,
        'hotspot_score': 0.15,
        'packing_density': 0.15,
        'interface_hydrophobicity': 0.10
    }
    
    quality_score = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in scores:
            # Normalize scores
            if metric == 'interface_area':
                # Normalize to 0-1 (typical interface 600-2000 Ų)
                normalized = min(1.0, scores[metric] / 1500)
            elif metric == 'contact_density':
                # Already normalized
                normalized = min(1.0, scores[metric] / 5)
            elif metric == 'interface_hydrophobicity':
                # Invert (lower is better)
                normalized = 1.0 - min(1.0, scores[metric])
            else:
                # Already 0-1
                normalized = scores[metric]
            
            quality_score += normalized * weight
            total_weight += weight
    
    if total_weight > 0:
        quality_score /= total_weight
    
    return min(1.0, max(0.0, quality_score))

def composite_scoring_function(
    complex_structure: PDB.Structure.Structure,
    receptor_chain: str,
    ligand_chain: str
) -> float:
    """
    Composite scoring function for ranking complexes
    
    Args:
        complex_structure: Complex structure
        receptor_chain: Receptor chain ID
        ligand_chain: Ligand chain ID
    
    Returns:
        Composite score (higher is better)
    """
    
    # Get all scores
    scores = score_protein_protein_interaction(
        complex_structure,
        receptor_chain,
        ligand_chain
    )
    
    # Return overall quality
    return scores.get('complex_quality', 0.0)

# Helper functions

def _get_interface_residues(
    complex_structure: PDB.Structure.Structure,
    chain1: str,
    chain2: str,
    distance_cutoff: float
) -> Tuple[List[int], List[int]]:
    """Get interface residues between two chains"""
    
    model = complex_structure[0]
    
    chain1_atoms = []
    chain2_atoms = []
    
    for residue in model[chain1]:
        if is_aa(residue):
            chain1_atoms.extend(list(residue.get_atoms()))
    
    for residue in model[chain2]:
        if is_aa(residue):
            chain2_atoms.extend(list(residue.get_atoms()))
    
    ns1 = NeighborSearch(chain1_atoms)
    ns2 = NeighborSearch(chain2_atoms)
    
    interface1 = set()
    interface2 = set()
    
    for atom in chain2_atoms:
        close_atoms = ns1.search(atom.coord, distance_cutoff)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            interface1.add(residue.id[1])
    
    for atom in chain1_atoms:
        close_atoms = ns2.search(atom.coord, distance_cutoff)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            interface2.add(residue.id[1])
    
    return sorted(list(interface1)), sorted(list(interface2))

def _count_interface_contacts(
    complex_structure: PDB.Structure.Structure,
    chain1: str,
    chain2: str,
    distance_cutoff: float
) -> int:
    """Count atomic contacts between chains"""
    
    model = complex_structure[0]
    
    chain1_atoms = [atom for res in model[chain1] if is_aa(res) for atom in res]
    chain2_atoms = [atom for res in model[chain2] if is_aa(res) for atom in res]
    
    ns2 = NeighborSearch(chain2_atoms)
    
    contact_count = 0
    for atom in chain1_atoms:
        contacts = ns2.search(atom.coord, distance_cutoff)
        contact_count += len(contacts)
    
    return contact_count

def _calculate_interface_hydrophobicity(
    complex_structure: PDB.Structure.Structure,
    interface1: List[int],
    interface2: List[int],
    chain1: str,
    chain2: str
) -> float:
    """Calculate hydrophobicity of interface"""
    
    hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
    
    model = complex_structure[0]
    
    hydro_count = 0
    total_count = 0
    
    for res_num in interface1:
        try:
            residue = model[chain1][res_num]
            total_count += 1
            if residue.get_resname() in hydrophobic:
                hydro_count += 1
        except:
            continue
    
    for res_num in interface2:
        try:
            residue = model[chain2][res_num]
            total_count += 1
            if residue.get_resname() in hydrophobic:
                hydro_count += 1
        except:
            continue
    
    if total_count == 0:
        return 0.0
    
    return hydro_count / total_count

def _count_hydrogen_bonds(
    complex_structure: PDB.Structure.Structure,
    chain1: str,
    chain2: str
) -> int:
    """Count approximate hydrogen bonds between chains"""
    
    model = complex_structure[0]
    
    donors = ['N']
    acceptors = ['O']
    
    h_bond_count = 0
    
    for res1 in model[chain1]:
        if not is_aa(res1):
            continue
        for res2 in model[chain2]:
            if not is_aa(res2):
                continue
            
            for atom1 in res1:
                if atom1.element in donors:
                    for atom2 in res2:
                        if atom2.element in acceptors:
                            dist = np.linalg.norm(atom1.coord - atom2.coord)
                            if dist <= 3.5:
                                h_bond_count += 1
    
    return h_bond_count

def _count_salt_bridges(
    complex_structure: PDB.Structure.Structure,
    chain1: str,
    chain2: str
) -> int:
    """Count salt bridges between chains"""
    
    model = complex_structure[0]
    
    positive = {'ARG', 'LYS'}
    negative = {'ASP', 'GLU'}
    
    salt_bridge_count = 0
    
    for res1 in model[chain1]:
        if not is_aa(res1):
            continue
        if res1.get_resname() in positive:
            for res2 in model[chain2]:
                if not is_aa(res2):
                    continue
                if res2.get_resname() in negative:
                    # Check distance between charged groups
                    for atom1 in res1:
                        if atom1.element == 'N':
                            for atom2 in res2:
                                if atom2.element == 'O':
                                    dist = np.linalg.norm(atom1.coord - atom2.coord)
                                    if dist <= 4.0:
                                        salt_bridge_count += 1
                                        break
    
    return salt_bridge_count
