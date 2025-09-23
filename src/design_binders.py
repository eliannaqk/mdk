"""
Binder design module for MDK-LRP1 nanobody generation
Integrates with BindCraft for de novo binder design
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import tempfile
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Try to import BindCraft components
try:
    from bindcraft import BinderDesign
    from bindcraft.utils import prepare_target
    BINDCRAFT_AVAILABLE = True
except ImportError:
    BINDCRAFT_AVAILABLE = False
    print("Warning: BindCraft not installed. Using mock implementation.")

@dataclass
class DesignConfig:
    """Configuration for binder design"""
    num_designs: int = 100
    binder_length_range: Tuple[int, int] = (50, 80)
    
    # AlphaFold2 settings
    num_recycles: int = 3
    use_templates: bool = False
    max_msa_depth: int = 32
    
    # Optimization settings
    learning_rate: float = 0.1
    num_optimization_steps: int = 100
    
    # Loss weights
    plddt_weight: float = 1.0
    iptm_weight: float = 2.0
    pae_weight: float = 1.0
    interface_area_weight: float = 1.5
    
    # ProteinMPNN settings
    mpnn_sampling_temp: float = 0.2
    mpnn_num_sequences: int = 8
    
    # Filtering thresholds
    min_plddt: float = 80.0
    min_iptm: float = 0.7
    min_interface_area: float = 600.0
    max_interface_hydrophobicity: float = 0.3
    
    # Computational settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    seed: int = 42

@dataclass
class BinderDesign:
    """Container for a designed binder"""
    design_id: str
    sequence: str
    structure_pdb: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    interface_residues: List[int] = field(default_factory=list)
    complex_structure: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'design_id': self.design_id,
            'sequence': self.sequence,
            'structure_pdb': self.structure_pdb,
            'scores': self.scores,
            'interface_residues': self.interface_residues
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BinderDesign':
        """Create from dictionary"""
        return cls(**data)

class BindCraftRunner:
    """Wrapper for BindCraft binder design"""
    
    def __init__(self, config: DesignConfig):
        self.config = config
        self.designs = []
        
        # Set random seed
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.manual_seed(config.seed)
    
    def design_binders(
        self,
        target_pdb: Path,
        epitope_residues: List[int],
        output_dir: Path
    ) -> List[BinderDesign]:
        """
        Run BindCraft to design binders
        
        Args:
            target_pdb: Path to target PDB file
            epitope_residues: List of epitope residue numbers
            output_dir: Output directory for designs
        
        Returns:
            List of BinderDesign objects
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if BINDCRAFT_AVAILABLE:
            return self._run_bindcraft(target_pdb, epitope_residues, output_dir)
        else:
            return self._run_mock_design(target_pdb, epitope_residues, output_dir)
    
    def _run_bindcraft(
        self,
        target_pdb: Path,
        epitope_residues: List[int],
        output_dir: Path
    ) -> List[BinderDesign]:
        """Run actual BindCraft pipeline"""
        
        # Prepare BindCraft job
        job_config = {
            'target_pdb': str(target_pdb),
            'epitope_residues': epitope_residues,
            'num_designs': self.config.num_designs,
            'binder_length_range': self.config.binder_length_range,
            'device': self.config.device,
            'output_dir': str(output_dir)
        }
        
        # Initialize BindCraft
        from bindcraft import BinderDesigner
        designer = BinderDesigner(job_config)
        
        # Run design
        designs = []
        for i in range(self.config.num_designs):
            design_id = f"design_{i+1:04d}"
            
            # Generate binder
            result = designer.hallucinate_binder(
                length_range=self.config.binder_length_range,
                num_recycles=self.config.num_recycles,
                learning_rate=self.config.learning_rate,
                num_steps=self.config.num_optimization_steps
            )
            
            if result:
                # Create BinderDesign object
                design = BinderDesign(
                    design_id=design_id,
                    sequence=result['sequence'],
                    scores=result['scores'],
                    structure_pdb=result.get('pdb_file')
                )
                designs.append(design)
        
        return designs
    
    def _run_mock_design(
        self,
        target_pdb: Path,
        epitope_residues: List[int],
        output_dir: Path
    ) -> List[BinderDesign]:
        """Mock design for testing without BindCraft"""
        
        designs = []
        
        for i in range(min(10, self.config.num_designs)):  # Limit mock designs
            design_id = f"design_{i+1:04d}"
            
            # Generate random sequence
            length = np.random.randint(
                self.config.binder_length_range[0],
                self.config.binder_length_range[1]
            )
            
            # Bias towards common amino acids in nanobodies
            aa_weights = {
                'G': 0.08, 'A': 0.06, 'V': 0.06, 'L': 0.06, 'I': 0.04,
                'S': 0.08, 'T': 0.06, 'C': 0.02, 'M': 0.02, 'P': 0.04,
                'F': 0.04, 'Y': 0.06, 'W': 0.03, 'H': 0.02, 'K': 0.05,
                'R': 0.05, 'D': 0.05, 'E': 0.05, 'N': 0.04, 'Q': 0.04
            }
            
            amino_acids = list(aa_weights.keys())
            weights = list(aa_weights.values())
            
            sequence = ''.join(np.random.choice(
                amino_acids,
                size=length,
                p=weights
            ))
            
            # Generate mock scores
            scores = {
                'plddt': np.random.uniform(70, 95),
                'iptm': np.random.uniform(0.5, 0.9),
                'pae': np.random.uniform(3, 10),
                'interface_area': np.random.uniform(400, 1200),
                'interface_hydrophobicity': np.random.uniform(0.1, 0.4)
            }
            
            # Mock interface residues (middle third of sequence)
            start = length // 3
            end = 2 * length // 3
            interface_residues = list(range(start, end))
            
            design = BinderDesign(
                design_id=design_id,
                sequence=sequence,
                scores=scores,
                interface_residues=interface_residues
            )
            
            designs.append(design)
        
        print(f"Generated {len(designs)} mock designs")
        return designs
    
    def optimize_sequences(
        self,
        designs: List[BinderDesign],
        target_pdb: Path,
        output_dir: Path
    ) -> List[BinderDesign]:
        """
        Optimize sequences using ProteinMPNN
        
        Args:
            designs: List of initial designs
            target_pdb: Path to target structure
            output_dir: Output directory
        
        Returns:
            List of optimized designs
        """
        
        optimized = []
        
        for design in designs:
            # Run ProteinMPNN (or mock)
            if self._check_proteinmpnn():
                opt_sequences = self._run_proteinmpnn(
                    design,
                    target_pdb,
                    output_dir
                )
            else:
                # Mock optimization - slight mutations
                opt_sequences = self._mock_sequence_optimization(design.sequence)
            
            # Score each optimized sequence
            for seq in opt_sequences[:self.config.mpnn_num_sequences]:
                opt_design = BinderDesign(
                    design_id=f"{design.design_id}_opt",
                    sequence=seq,
                    scores=design.scores.copy(),  # Inherit scores
                    interface_residues=design.interface_residues
                )
                optimized.append(opt_design)
        
        return optimized
    
    def _check_proteinmpnn(self) -> bool:
        """Check if ProteinMPNN is available"""
        try:
            import proteinmpnn
            return True
        except ImportError:
            return False
    
    def _run_proteinmpnn(
        self,
        design: BinderDesign,
        target_pdb: Path,
        output_dir: Path
    ) -> List[str]:
        """Run ProteinMPNN for sequence optimization"""
        
        # This would call actual ProteinMPNN
        # For now, return original sequence
        return [design.sequence]
    
    def _mock_sequence_optimization(self, sequence: str, num_variants: int = 3) -> List[str]:
        """Generate sequence variants through mock optimization"""
        
        variants = [sequence]  # Include original
        
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        for _ in range(num_variants - 1):
            # Make 5-10% mutations
            seq_list = list(sequence)
            num_mutations = int(len(sequence) * np.random.uniform(0.05, 0.1))
            
            positions = np.random.choice(len(sequence), num_mutations, replace=False)
            
            for pos in positions:
                seq_list[pos] = np.random.choice(list(amino_acids))
            
            variants.append(''.join(seq_list))
        
        return variants

def design_de_novo_binders(
    target_pdb: Path,
    epitope_residues: List[int],
    config: DesignConfig,
    output_dir: Path
) -> List[BinderDesign]:
    """
    Main function to design de novo binders
    
    Args:
        target_pdb: Path to target PDB file
        epitope_residues: List of epitope residue numbers
        config: Design configuration
        output_dir: Output directory
    
    Returns:
        List of designed binders
    """
    
    print(f"Designing {config.num_designs} binders...")
    print(f"  Target: {target_pdb}")
    print(f"  Epitope: {len(epitope_residues)} residues")
    print(f"  Length range: {config.binder_length_range}")
    
    # Initialize runner
    runner = BindCraftRunner(config)
    
    # Design binders
    designs = runner.design_binders(
        target_pdb,
        epitope_residues,
        output_dir
    )
    
    print(f"Generated {len(designs)} initial designs")
    
    return designs

def optimize_sequences(
    designs: List[BinderDesign],
    target_pdb: Path,
    config: DesignConfig,
    output_dir: Path
) -> List[BinderDesign]:
    """
    Optimize designed sequences
    
    Args:
        designs: List of initial designs
        target_pdb: Path to target structure
        config: Design configuration
        output_dir: Output directory
    
    Returns:
        List of optimized designs
    """
    
    print(f"Optimizing {len(designs)} designs...")
    
    runner = BindCraftRunner(config)
    optimized = runner.optimize_sequences(designs, target_pdb, output_dir)
    
    print(f"Generated {len(optimized)} optimized sequences")
    
    return optimized

def filter_designs(
    designs: List[BinderDesign],
    config: DesignConfig
) -> List[BinderDesign]:
    """
    Filter designs based on quality metrics
    
    Args:
        designs: List of designs to filter
        config: Design configuration with thresholds
    
    Returns:
        List of filtered designs
    """
    
    print(f"Filtering {len(designs)} designs...")
    
    filtered = []
    filter_stats = {
        'plddt': 0,
        'iptm': 0,
        'interface_area': 0,
        'hydrophobicity': 0
    }
    
    for design in designs:
        # Check filters
        if design.scores.get('plddt', 0) < config.min_plddt:
            filter_stats['plddt'] += 1
            continue
        
        if design.scores.get('iptm', 0) < config.min_iptm:
            filter_stats['iptm'] += 1
            continue
        
        if design.scores.get('interface_area', 0) < config.min_interface_area:
            filter_stats['interface_area'] += 1
            continue
        
        if design.scores.get('interface_hydrophobicity', 1) > config.max_interface_hydrophobicity:
            filter_stats['hydrophobicity'] += 1
            continue
        
        filtered.append(design)
    
    print(f"Retained {len(filtered)} designs after filtering")
    print("Filter statistics:")
    for metric, count in filter_stats.items():
        print(f"  Failed {metric}: {count}")
    
    return filtered

def rank_designs(designs: List[BinderDesign]) -> List[BinderDesign]:
    """
    Rank designs by composite score
    
    Args:
        designs: List of designs to rank
    
    Returns:
        Sorted list of designs
    """
    
    for design in designs:
        # Calculate composite score
        scores = design.scores
        
        composite = (
            scores.get('plddt', 0) / 100.0 * 0.3 +
            scores.get('iptm', 0) * 0.4 +
            (1.0 - scores.get('pae', 30) / 30.0) * 0.2 +
            min(scores.get('interface_area', 0) / 1000.0, 1.0) * 0.1
        )
        
        design.scores['composite'] = composite
    
    # Sort by composite score
    ranked = sorted(designs, key=lambda x: x.scores['composite'], reverse=True)
    
    return ranked

def save_designs(
    designs: List[BinderDesign],
    output_dir: Path
):
    """
    Save designs to files
    
    Args:
        designs: List of designs to save
        output_dir: Output directory
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sequences in FASTA
    sequences = []
    for design in designs:
        record = SeqRecord(
            Seq(design.sequence),
            id=design.design_id,
            description=f"score={design.scores.get('composite', 0):.3f}"
        )
        sequences.append(record)
    
    fasta_file = output_dir / "designs.fasta"
    SeqIO.write(sequences, fasta_file, "fasta")
    
    # Save detailed data in JSON
    json_data = [design.to_dict() for design in designs]
    
    json_file = output_dir / "designs.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved {len(designs)} designs to {output_dir}")