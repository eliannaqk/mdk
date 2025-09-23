"""
Target preparation module for MDK-LRP1 nanobody design
Handles structure processing and epitope definition
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TargetProtein:
    """Container for target protein information"""
    name: str
    pdb_file: Path
    chain_id: str
    structure: Optional[PDB.Structure.Structure] = None
    sequence: Optional[str] = None
    surface_residues: Optional[List[int]] = None
    
    def __post_init__(self):
        """Load structure upon initialization"""
        if self.pdb_file.exists():
            self.load_structure()
    
    def load_structure(self):
        """Load PDB structure"""
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure(self.name, self.pdb_file)
        self.sequence = self.extract_sequence()
        self.surface_residues = self.identify_surface_residues()
    
    def extract_sequence(self) -> str:
        """Extract amino acid sequence from structure"""
        if not self.structure:
            return ""
        
        sequence = []
        for model in self.structure:
            for chain in model:
                if chain.id == self.chain_id:
                    for residue in chain:
                        if is_aa(residue):
                            sequence.append(three_to_one(residue.get_resname()))
        return ''.join(sequence)
    
    def identify_surface_residues(self, cutoff: float = 2.5) -> List[int]:
        """Identify surface-exposed residues using solvent accessibility"""
        if not self.structure:
            return []
        
        surface_residues = []
        
        # Get all atoms
        atoms = list(self.structure.get_atoms())
        ns = NeighborSearch(atoms)
        
        for model in self.structure:
            for chain in model:
                if chain.id == self.chain_id:
                    for residue in chain:
                        if not is_aa(residue):
                            continue
                        
                        # Check if any atom of the residue is surface exposed
                        is_surface = False
                        for atom in residue:
                            # Count neighbors within cutoff
                            neighbors = ns.search(atom.coord, cutoff)
                            # If few neighbors, likely surface exposed
                            if len(neighbors) < 10:
                                is_surface = True
                                break
                        
                        if is_surface:
                            surface_residues.append(residue.id[1])
        
        return sorted(surface_residues)
    
    def get_residue_coordinates(self, residue_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get CA coordinates for specified residues"""
        coords = {}
        
        for model in self.structure:
            for chain in model:
                if chain.id == self.chain_id:
                    for residue in chain:
                        if residue.id[1] in residue_ids and 'CA' in residue:
                            coords[residue.id[1]] = residue['CA'].coord
        
        return coords

@dataclass
class EpitopeDefinition:
    """Container for epitope information"""
    name: str
    description: str
    core_residues: List[int]
    expanded_residues: List[int]
    surface_area: Optional[float] = None
    center_of_mass: Optional[np.ndarray] = None
    
    def calculate_properties(self, target: TargetProtein):
        """Calculate epitope properties"""
        coords = target.get_residue_coordinates(self.expanded_residues)
        
        if coords:
            # Calculate center of mass
            positions = np.array(list(coords.values()))
            self.center_of_mass = np.mean(positions, axis=0)
            
            # Estimate surface area (simplified)
            self.surface_area = len(self.expanded_residues) * 50  # ~50 Ų per residue

class ChainSelect(Select):
    """Select specific chain from PDB"""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id

def prepare_mdk_target(
    pdb_file: Path,
    chain_id: str = 'A',
    output_dir: Optional[Path] = None
) -> TargetProtein:
    """
    Prepare MDK target structure for binder design
    
    Args:
        pdb_file: Path to MDK PDB file
        chain_id: Chain identifier
        output_dir: Optional output directory for processed structure
    
    Returns:
        TargetProtein object
    """
    
    # Create target object
    target = TargetProtein(
        name="MDK",
        pdb_file=pdb_file,
        chain_id=chain_id
    )
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean and save structure
        io = PDBIO()
        io.set_structure(target.structure)
        clean_pdb = output_dir / "mdk_clean.pdb"
        io.save(str(clean_pdb), ChainSelect(chain_id))
        
        print(f"Saved cleaned structure to {clean_pdb}")
    
    # Report statistics
    print(f"Target: {target.name}")
    print(f"  Chain {chain_id}: {len(target.sequence)} residues")
    print(f"  Surface residues: {len(target.surface_residues)}")
    
    return target

def define_epitope(
    target: TargetProtein,
    epitope_residues: List[int],
    epitope_name: str = "epitope",
    expand_radius: float = 5.0
) -> EpitopeDefinition:
    """
    Define epitope on target protein
    
    Args:
        target: TargetProtein object
        epitope_residues: Core epitope residue numbers
        epitope_name: Name for the epitope
        expand_radius: Radius for epitope expansion (Å)
    
    Returns:
        EpitopeDefinition object
    """
    
    # Get core residues that are on the surface
    core_on_surface = [r for r in epitope_residues if r in target.surface_residues]
    
    if len(core_on_surface) < len(epitope_residues):
        print(f"Warning: {len(epitope_residues) - len(core_on_surface)} core residues are not surface-exposed")
    
    # Expand epitope
    expanded = expand_epitope_region(
        target,
        core_on_surface,
        expand_radius
    )
    
    # Create epitope definition
    epitope = EpitopeDefinition(
        name=epitope_name,
        description=f"Epitope with {len(core_on_surface)} core residues",
        core_residues=core_on_surface,
        expanded_residues=expanded
    )
    
    # Calculate properties
    epitope.calculate_properties(target)
    
    print(f"Defined epitope '{epitope_name}':")
    print(f"  Core residues: {len(epitope.core_residues)}")
    print(f"  Expanded residues: {len(epitope.expanded_residues)}")
    print(f"  Estimated surface area: {epitope.surface_area:.0f} Ų")
    
    return epitope

def expand_epitope_region(
    target: TargetProtein,
    core_residues: List[int],
    radius: float = 5.0
) -> List[int]:
    """
    Expand epitope definition to include neighboring residues
    
    Args:
        target: TargetProtein object
        core_residues: Core epitope residue numbers
        radius: Expansion radius in Angstroms
    
    Returns:
        List of expanded epitope residue numbers
    """
    
    if not target.structure:
        return core_residues
    
    # Get all atoms from core residues
    core_atoms = []
    
    for model in target.structure:
        for chain in model:
            if chain.id == target.chain_id:
                for residue in chain:
                    if residue.id[1] in core_residues:
                        core_atoms.extend(list(residue.get_atoms()))
    
    # Find neighboring residues within radius
    all_atoms = list(target.structure.get_atoms())
    ns = NeighborSearch(all_atoms)
    
    expanded_residues = set(core_residues)
    
    for atom in core_atoms:
        neighbors = ns.search(atom.coord, radius)
        for neighbor in neighbors:
            residue = neighbor.get_parent()
            if is_aa(residue) and residue.get_parent().id == target.chain_id:
                expanded_residues.add(residue.id[1])
    
    return sorted(list(expanded_residues))

def identify_binding_hotspots(
    target: TargetProtein,
    epitope: EpitopeDefinition,
    conservation_data: Optional[Dict[int, float]] = None
) -> List[int]:
    """
    Identify key binding hotspot residues within epitope
    
    Args:
        target: TargetProtein object
        epitope: EpitopeDefinition object
        conservation_data: Optional conservation scores by residue
    
    Returns:
        List of hotspot residue numbers
    """
    
    hotspots = []
    
    # Identify basic residues (important for MDK-LRP1)
    basic_residues = {'R', 'K', 'H'}
    
    for model in target.structure:
        for chain in model:
            if chain.id == target.chain_id:
                for residue in chain:
                    res_id = residue.id[1]
                    
                    if res_id not in epitope.expanded_residues:
                        continue
                    
                    # Check if basic residue
                    res_name = residue.get_resname()
                    if three_to_one(res_name) in basic_residues:
                        hotspots.append(res_id)
                    
                    # Check conservation if available
                    elif conservation_data and res_id in conservation_data:
                        if conservation_data[res_id] > 0.8:  # Highly conserved
                            hotspots.append(res_id)
    
    return sorted(list(set(hotspots)))

def save_epitope_pdb(
    target: TargetProtein,
    epitope: EpitopeDefinition,
    output_file: Path
):
    """
    Save epitope residues as a separate PDB file
    
    Args:
        target: TargetProtein object
        epitope: EpitopeDefinition object
        output_file: Output PDB file path
    """
    
    class EpitopeSelect(Select):
        def __init__(self, chain_id, residue_ids):
            self.chain_id = chain_id
            self.residue_ids = residue_ids
        
        def accept_residue(self, residue):
            return (residue.get_parent().id == self.chain_id and 
                   residue.id[1] in self.residue_ids)
    
    io = PDBIO()
    io.set_structure(target.structure)
    io.save(str(output_file), EpitopeSelect(target.chain_id, epitope.expanded_residues))
    
    print(f"Saved epitope structure to {output_file}")
def three_to_one(res_name: str) -> str:
    """Convert three-letter residue codes to one-letter form (fallback to X)."""
    return protein_letters_3to1.get(res_name.upper(), 'X')
