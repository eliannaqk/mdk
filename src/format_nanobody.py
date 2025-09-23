"""
Nanobody formatting module for converting designed binders to VHH format
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from Bio import SeqIO, PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import Superimposer
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CDRDefinition:
    """Definition of CDR regions"""
    cdr1: str
    cdr1_start: int
    cdr1_end: int
    cdr2: str
    cdr2_start: int
    cdr2_end: int
    cdr3: str
    cdr3_start: int
    cdr3_end: int

@dataclass
class NanobodyFramework:
    """Standard nanobody framework regions"""
    FR1: str = "QVQLVESGGGLVQPGGSLRLSCAAS"
    FR2: str = "WVRQAPGKGLEWVS"
    FR3: str = "RFTISRDNSKNTLYLQMNSLRAEDTAVYYC"
    FR4: str = "WGQGTLVTVSS"
    
    # Canonical positions (Kabat numbering approximation)
    FR1_range: Tuple[int, int] = (1, 25)
    CDR1_range: Tuple[int, int] = (26, 35)
    FR2_range: Tuple[int, int] = (36, 49)
    CDR2_range: Tuple[int, int] = (50, 65)
    FR3_range: Tuple[int, int] = (66, 94)
    CDR3_range: Tuple[int, int] = (95, 110)
    FR4_range: Tuple[int, int] = (111, 121)

@dataclass
class Nanobody:
    """Container for nanobody sequence and properties"""
    nanobody_id: str
    sequence: str
    cdrs: CDRDefinition
    framework: NanobodyFramework
    original_design_id: str
    original_sequence: str
    paratope_residues: List[int] = field(default_factory=list)
    properties: Dict[str, float] = field(default_factory=dict)
    optimization_notes: List[str] = field(default_factory=list)
    
    def get_cdr_sequences(self) -> Dict[str, str]:
        """Extract CDR sequences from full nanobody sequence"""
        return {
            'CDR1': self.cdrs.cdr1,
            'CDR2': self.cdrs.cdr2,
            'CDR3': self.cdrs.cdr3
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'nanobody_id': self.nanobody_id,
            'sequence': self.sequence,
            'cdrs': {
                'CDR1': self.cdrs.cdr1,
                'CDR2': self.cdrs.cdr2,
                'CDR3': self.cdrs.cdr3
            },
            'original_design_id': self.original_design_id,
            'properties': self.properties,
            'optimization_notes': self.optimization_notes
        }

class NanobodyFormatter:
    """Convert designed binders to nanobody format"""
    
    def __init__(self, framework: Optional[NanobodyFramework] = None):
        self.framework = framework or NanobodyFramework()
    
    def format_to_nanobody(
        self,
        design_id: str,
        sequence: str,
        interface_residues: List[int],
        scores: Optional[Dict[str, float]] = None
    ) -> Nanobody:
        """
        Convert a designed binder to nanobody format
        
        Args:
            design_id: Original design identifier
            sequence: Designed binder sequence
            interface_residues: Residues forming the binding interface
            scores: Optional design scores
        
        Returns:
            Nanobody object
        """
        
        # Extract paratope
        paratope = extract_paratope(sequence, interface_residues)
        
        # Extract CDR-like regions
        cdrs = self._extract_cdr_regions(sequence, paratope)
        
        # Graft CDRs onto framework
        nanobody_sequence = graft_cdrs(self.framework, cdrs)
        
        # Optimize sequence
        optimized_sequence, notes = optimize_framework(nanobody_sequence)
        
        # Calculate properties
        properties = self._calculate_properties(optimized_sequence)
        
        # Create nanobody object
        nanobody = Nanobody(
            nanobody_id=f"nb_{design_id}",
            sequence=optimized_sequence,
            cdrs=cdrs,
            framework=self.framework,
            original_design_id=design_id,
            original_sequence=sequence,
            paratope_residues=interface_residues,
            properties=properties,
            optimization_notes=notes
        )
        
        return nanobody
    
    def _extract_cdr_regions(
        self,
        sequence: str,
        paratope: List[int]
    ) -> CDRDefinition:
        """
        Extract CDR-like regions from designed sequence
        
        Args:
            sequence: Designed binder sequence
            paratope: Paratope residue positions
        
        Returns:
            CDRDefinition object
        """
        
        # Find continuous stretches in paratope
        cdrs = []
        if paratope:
            current_cdr = [paratope[0]]
            
            for i in range(1, len(paratope)):
                if paratope[i] - paratope[i-1] <= 2:  # Allow small gaps
                    current_cdr.append(paratope[i])
                else:
                    if len(current_cdr) >= 3:  # Minimum CDR length
                        cdrs.append(current_cdr)
                    current_cdr = [paratope[i]]
            
            if len(current_cdr) >= 3:
                cdrs.append(current_cdr)
        
        # Map to canonical CDRs
        cdr1 = cdr2 = cdr3 = ""
        cdr1_start = cdr1_end = 26
        cdr2_start = cdr2_end = 50
        cdr3_start = cdr3_end = 95
        
        if len(cdrs) >= 1:
            cdr1_positions = cdrs[0]
            cdr1 = sequence[cdr1_positions[0]:cdr1_positions[-1]+1]
            cdr1_start = 26
            cdr1_end = 26 + len(cdr1) - 1
        
        if len(cdrs) >= 2:
            cdr2_positions = cdrs[1]
            cdr2 = sequence[cdr2_positions[0]:cdr2_positions[-1]+1]
            cdr2_start = 50
            cdr2_end = 50 + len(cdr2) - 1
        
        if len(cdrs) >= 3:
            cdr3_positions = cdrs[2]
            cdr3 = sequence[cdr3_positions[0]:cdr3_positions[-1]+1]
            cdr3_start = 95
            cdr3_end = 95 + len(cdr3) - 1
        
        # Use defaults if needed
        if not cdr1:
            cdr1 = "GFTFS"  # Common CDR1
        if not cdr2:
            cdr2 = "IS"     # Short CDR2
        if not cdr3:
            cdr3 = "AR"     # Minimal CDR3
        
        # Ensure reasonable CDR lengths
        if len(cdr1) > 10:
            cdr1 = cdr1[:10]
        if len(cdr2) > 17:
            cdr2 = cdr2[:17]
        if len(cdr3) > 25:
            cdr3 = cdr3[:25]
        
        return CDRDefinition(
            cdr1=cdr1, cdr1_start=cdr1_start, cdr1_end=cdr1_end,
            cdr2=cdr2, cdr2_start=cdr2_start, cdr2_end=cdr2_end,
            cdr3=cdr3, cdr3_start=cdr3_start, cdr3_end=cdr3_end
        )
    
    def _calculate_properties(self, sequence: str) -> Dict[str, float]:
        """Calculate nanobody properties"""
        
        from Bio.SeqUtils import ProtParam
        
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        properties = {
            'length': len(sequence),
            'molecular_weight': analyzer.molecular_weight(),
            'theoretical_pi': analyzer.isoelectric_point(),
            'instability_index': analyzer.instability_index(),
            'aliphatic_index': analyzer.aliphatic_index(),
            'gravy': analyzer.gravy(),
            'aromaticity': analyzer.aromaticity()
        }
        
        # Secondary structure fractions
        helix, turn, sheet = analyzer.secondary_structure_fraction()
        properties['helix_fraction'] = helix
        properties['turn_fraction'] = turn
        properties['sheet_fraction'] = sheet
        
        return properties

def extract_paratope(
    sequence: str,
    interface_residues: List[int],
    min_cluster_size: int = 3
) -> List[int]:
    """
    Extract paratope residues from designed sequence
    
    Args:
        sequence: Designed binder sequence
        interface_residues: Predicted interface residue positions
        min_cluster_size: Minimum size for paratope cluster
    
    Returns:
        List of paratope residue positions
    """
    
    if not interface_residues:
        # If no interface info, use middle third as default
        length = len(sequence)
        start = length // 3
        end = 2 * length // 3
        return list(range(start, end))
    
    # Filter and cluster interface residues
    paratope = []
    interface_sorted = sorted(interface_residues)
    
    current_cluster = [interface_sorted[0]]
    
    for i in range(1, len(interface_sorted)):
        if interface_sorted[i] - interface_sorted[i-1] <= 3:
            current_cluster.append(interface_sorted[i])
        else:
            if len(current_cluster) >= min_cluster_size:
                paratope.extend(current_cluster)
            current_cluster = [interface_sorted[i]]
    
    if len(current_cluster) >= min_cluster_size:
        paratope.extend(current_cluster)
    
    return sorted(paratope)

def graft_cdrs(
    framework: NanobodyFramework,
    cdrs: CDRDefinition
) -> str:
    """
    Graft CDRs onto nanobody framework
    
    Args:
        framework: Nanobody framework regions
        cdrs: CDR sequences to graft
    
    Returns:
        Complete nanobody sequence
    """
    
    # Assemble nanobody sequence
    nanobody_sequence = (
        framework.FR1 +
        cdrs.cdr1 +
        framework.FR2 +
        cdrs.cdr2 +
        framework.FR3 +
        cdrs.cdr3 +
        framework.FR4
    )
    
    return nanobody_sequence

def optimize_framework(sequence: str) -> Tuple[str, List[str]]:
    """
    Optimize nanobody sequence for stability and expression
    
    Args:
        sequence: Initial nanobody sequence
    
    Returns:
        Tuple of (optimized sequence, list of optimization notes)
    """
    
    notes = []
    optimized = sequence
    
    # Check for unpaired cysteines
    cys_positions = [i for i, aa in enumerate(sequence) if aa == 'C']
    
    # Framework cysteines should be at approximately positions 22 and 92
    expected_cys = [22, 92]
    unexpected_cys = []
    
    for pos in cys_positions:
        if not any(abs(pos - exp) < 5 for exp in expected_cys):
            unexpected_cys.append(pos)
    
    if unexpected_cys:
        notes.append(f"Unexpected cysteines at positions: {unexpected_cys}")
        # Could mutate to serine
        seq_list = list(optimized)
        for pos in unexpected_cys:
            if pos < len(seq_list):
                seq_list[pos] = 'S'
                notes.append(f"Mutated C{pos}S for stability")
        optimized = ''.join(seq_list)
    
    # Check for N-glycosylation sites (N-X-S/T where X != P)
    glyco_pattern = r'N[^P][ST]'
    glyco_sites = [(m.start(), m.group()) for m in re.finditer(glyco_pattern, optimized)]
    
    if glyco_sites:
        notes.append(f"N-glycosylation sites found: {glyco_sites}")
        # Could mutate N to Q
        seq_list = list(optimized)
        for pos, motif in glyco_sites:
            if pos < len(seq_list):
                seq_list[pos] = 'Q'
                notes.append(f"Mutated N{pos}Q to remove glycosylation site")
        optimized = ''.join(seq_list)
    
    # Check for deamidation sites (NG, NS)
    deamidation_pattern = r'N[GS]'
    deamidation_sites = [(m.start(), m.group()) for m in re.finditer(deamidation_pattern, optimized)]
    
    if deamidation_sites:
        notes.append(f"Deamidation sites found: {deamidation_sites}")
    
    # Check for oxidation-prone methionines in CDRs
    met_positions = [i for i, aa in enumerate(optimized) if aa == 'M']
    cdr_met = [pos for pos in met_positions if 26 <= pos <= 35 or 50 <= pos <= 65 or 95 <= pos <= 110]
    
    if cdr_met:
        notes.append(f"Oxidation-prone Met in CDRs at: {cdr_met}")
        # Could mutate to Leu
        seq_list = list(optimized)
        for pos in cdr_met:
            if pos < len(seq_list):
                seq_list[pos] = 'L'
                notes.append(f"Mutated M{pos}L to prevent oxidation")
        optimized = ''.join(seq_list)
    
    # Check for aggregation-prone regions
    hydrophobic_stretch = find_hydrophobic_patches(optimized)
    if hydrophobic_stretch:
        notes.append(f"Hydrophobic patches found: {hydrophobic_stretch}")
    
    # Ensure canonical disulfide bond positions
    seq_list = list(optimized)
    
    # Force canonical cysteines if missing
    if 'C' not in optimized[20:25]:
        if len(seq_list) > 22:
            seq_list[22] = 'C'
            notes.append("Added canonical Cys22")
    
    if 'C' not in optimized[90:95]:
        if len(seq_list) > 92:
            seq_list[92] = 'C'
            notes.append("Added canonical Cys92")
    
    optimized = ''.join(seq_list)
    
    return optimized, notes

def find_hydrophobic_patches(
    sequence: str,
    window_size: int = 5,
    threshold: float = 0.8
) -> List[Tuple[int, int]]:
    """
    Find hydrophobic patches in sequence
    
    Args:
        sequence: Amino acid sequence
        window_size: Size of sliding window
        threshold: Fraction of hydrophobic residues to flag
    
    Returns:
        List of (start, end) positions of hydrophobic patches
    """
    
    hydrophobic = set('FWYLIMV')
    patches = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        hydro_count = sum(1 for aa in window if aa in hydrophobic)
        
        if hydro_count / window_size >= threshold:
            patches.append((i, i+window_size))
    
    # Merge overlapping patches
    merged = []
    for start, end in patches:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    
    return merged

def create_nanobody(
    design_sequence: str,
    interface_residues: List[int],
    framework: Optional[NanobodyFramework] = None,
    optimize: bool = True
) -> Nanobody:
    """
    High-level function to create nanobody from designed binder
    
    Args:
        design_sequence: Designed binder sequence
        interface_residues: Interface residue positions
        framework: Optional framework to use
        optimize: Whether to optimize the sequence
    
    Returns:
        Nanobody object
    """
    
    formatter = NanobodyFormatter(framework)
    
    nanobody = formatter.format_to_nanobody(
        design_id="custom",
        sequence=design_sequence,
        interface_residues=interface_residues
    )
    
    if not optimize:
        # Skip optimization
        nanobody.optimization_notes = ["Optimization skipped"]
    
    return nanobody

def humanize_nanobody(
    nanobody: Nanobody,
    human_vhh_database: Optional[Path] = None
) -> Nanobody:
    """
    Humanize nanobody sequence for reduced immunogenicity
    
    Args:
        nanobody: Nanobody to humanize
        human_vhh_database: Optional database of human VHH sequences
    
    Returns:
        Humanized nanobody
    """
    
    # This would implement actual humanization
    # For now, just flag it
    humanized = nanobody
    humanized.optimization_notes.append("Humanization pending")
    
    return humanized