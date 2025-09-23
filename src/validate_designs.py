"""
Design validation module for assessing nanobody quality
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from Bio.SeqUtils import ProtParam
from Bio.PDB import PDBParser, NeighborSearch
from Bio import pairwise2
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ValidationResult:
    """Container for validation results"""
    design_id: str
    passed: bool
    scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall validation score"""
        if not self.scores:
            return 0.0
        
        # Weighted average of different metrics
        weights = {
            'developability': 0.3,
            'expression': 0.2,
            'specificity': 0.2,
            'stability': 0.2,
            'immunogenicity': 0.1
        }
        
        total = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.scores:
                total += self.scores[metric] * weight
                total_weight += weight
        
        return total / total_weight if total_weight > 0 else 0.0

@dataclass
class DevelopabilityMetrics:
    """Developability assessment metrics"""
    aggregation_propensity: float
    solubility_score: float
    charge_distribution: str
    hydrophobic_patches: List[Tuple[int, int]]
    post_translational_modifications: List[str]
    manufacturability_score: float

class DesignValidator:
    """Comprehensive validator for nanobody designs"""
    
    def __init__(self):
        self.ptn_sequence = None  # Load PTN sequence for cross-reactivity
        self.load_reference_data()
    
    def load_reference_data(self):
        """Load reference sequences and data"""
        # PTN sequence (MDK paralog) for specificity checking
        self.ptn_sequence = """
        MGASALLAFALLAALSWQVQADKPPNPYEDKLEVQLLKDKNYTKHGEKCKETVAYF
        GSKCEWQAAKKCTPKTKKPTPTPKAEDKVRKKPTTAEEKKPTPKPPGKPTPGKEESG
        KKGATNKNRQKKKNKKAPKKPESKKPVAKKPTTAEEKPKTAKDKKAKARQAAKKAKT
        KSKKK
        """.replace('\n', '').replace(' ', '')
    
    def validate_design(
        self,
        sequence: str,
        design_id: str = "design",
        comprehensive: bool = True
    ) -> ValidationResult:
        """
        Validate a single design
        
        Args:
            sequence: Amino acid sequence
            design_id: Design identifier
            comprehensive: Whether to run all checks
        
        Returns:
            ValidationResult object
        """
        
        result = ValidationResult(design_id=design_id, passed=True)
        
        # Basic checks
        self._check_sequence_validity(sequence, result)
        
        # Developability
        dev_score = calculate_developability(sequence)
        result.scores['developability'] = dev_score
        
        # Expression prediction
        expr_score = predict_expression(sequence)
        result.scores['expression'] = expr_score
        
        # Specificity assessment
        spec_score = assess_specificity(sequence, self.ptn_sequence)
        result.scores['specificity'] = spec_score
        
        # Stability prediction
        stab_score = self._predict_stability(sequence)
        result.scores['stability'] = stab_score
        
        if comprehensive:
            # Immunogenicity
            immuno_score = self._assess_immunogenicity(sequence)
            result.scores['immunogenicity'] = 1.0 - immuno_score  # Lower is better
            
            # Aggregation propensity
            agg_score = self._calculate_aggregation_propensity(sequence)
            result.scores['aggregation'] = 1.0 - agg_score  # Lower is better
        
        # Determine if passed
        if result.scores['developability'] < 0.5:
            result.passed = False
            result.errors.append("Poor developability score")
        
        if result.scores['expression'] < 0.4:
            result.passed = False
            result.errors.append("Low predicted expression")
        
        if result.scores['specificity'] < 0.6:
            result.warnings.append("Potential cross-reactivity concern")
        
        return result
    
    def _check_sequence_validity(self, sequence: str, result: ValidationResult):
        """Check basic sequence validity"""
        
        # Check for valid amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        invalid = set(sequence) - valid_aa
        
        if invalid:
            result.errors.append(f"Invalid amino acids: {invalid}")
            result.passed = False
        
        # Check length
        if len(sequence) < 100:
            result.warnings.append("Unusually short for nanobody")
        elif len(sequence) > 150:
            result.warnings.append("Unusually long for nanobody")
        
        # Check for stop codons (*)
        if '*' in sequence:
            result.errors.append("Stop codon found in sequence")
            result.passed = False
        
        # Check for required cysteines (disulfide bond)
        cys_count = sequence.count('C')
        if cys_count < 2:
            result.warnings.append("Missing cysteines for disulfide bond")
        elif cys_count > 4:
            result.warnings.append("Excess cysteines may cause misfolding")
    
    def _predict_stability(self, sequence: str) -> float:
        """Predict thermal and chemical stability"""
        
        analyzer = ProtParam.ProteinAnalysis(sequence)
        
        # Instability index (< 40 is stable)
        instability = analyzer.instability_index()
        instab_score = max(0, (60 - instability) / 60)
        
        # Aliphatic index (higher is more stable)
        aliphatic = analyzer.aliphatic_index()
        aliph_score = min(aliphatic / 100, 1.0)
        
        # Check for stabilizing features
        features_score = 0
        
        # Canonical disulfide
        if 'C' in sequence[20:25] and 'C' in sequence[90:95]:
            features_score += 0.5
        
        # Proline in turns
        if 'P' in sequence[35:40] or 'P' in sequence[65:70]:
            features_score += 0.25
        
        # No free cysteines in CDRs
        cdr_positions = list(range(26, 36)) + list(range(50, 66)) + list(range(95, 111))
        cdr_cys = sum(1 for i in cdr_positions if i < len(sequence) and sequence[i] == 'C')
        if cdr_cys == 0:
            features_score += 0.25
        
        # Combine scores
        stability = (instab_score * 0.4 + aliph_score * 0.3 + features_score * 0.3)
        
        return min(stability, 1.0)
    
    def _assess_immunogenicity(self, sequence: str) -> float:
        """Assess potential immunogenicity"""
        
        # Simplified T-cell epitope prediction
        # Count 9-mer peptides with high MHC binding potential
        
        immunogenic_motifs = [
            'FLLLLLLLL',  # Highly hydrophobic
            'WWWWWWWWW',  # Aromatic
            'KKKKKKKK',   # Highly charged
            'RRRRRRRR'    # Highly charged
        ]
        
        score = 0.0
        
        # Check for highly repetitive sequences
        for i in range(len(sequence) - 8):
            peptide = sequence[i:i+9]
            
            # Check hydrophobicity
            hydro_count = sum(1 for aa in peptide if aa in 'FWYLIMV')
            if hydro_count >= 7:
                score += 0.1
            
            # Check charge clustering
            charge_count = sum(1 for aa in peptide if aa in 'RKDE')
            if charge_count >= 6:
                score += 0.1
        
        # Normalize to 0-1
        immunogenicity = min(score / (len(sequence) / 9), 1.0)
        
        return immunogenicity
    
    def _calculate_aggregation_propensity(self, sequence: str) -> float:
        """Calculate aggregation propensity"""
        
        # Based on hydrophobicity and β-sheet propensity
        hydrophobic = 'FWYLIMV'
        beta_prone = 'VFIYT'
        
        agg_score = 0.0
        
        # Sliding window analysis
        window = 7
        for i in range(len(sequence) - window + 1):
            segment = sequence[i:i+window]
            
            hydro_frac = sum(1 for aa in segment if aa in hydrophobic) / window
            beta_frac = sum(1 for aa in segment if aa in beta_prone) / window
            
            if hydro_frac > 0.7 and beta_frac > 0.5:
                agg_score += 1.0
            elif hydro_frac > 0.5 and beta_frac > 0.3:
                agg_score += 0.5
        
        # Normalize
        aggregation = min(agg_score / (len(sequence) / window), 1.0)
        
        return aggregation

def calculate_developability(sequence: str) -> float:
    """
    Calculate developability score
    
    Args:
        sequence: Amino acid sequence
    
    Returns:
        Developability score (0-1, higher is better)
    """
    
    analyzer = ProtParam.ProteinAnalysis(sequence)
    
    scores = []
    
    # pI in favorable range (6.5-9.0)
    pi = analyzer.isoelectric_point()
    if 6.5 <= pi <= 9.0:
        pi_score = 1.0
    else:
        pi_score = max(0, 1.0 - abs(pi - 7.5) / 3.5)
    scores.append(pi_score)
    
    # Solubility (negative GRAVY is good)
    gravy = analyzer.gravy()
    solubility_score = max(0, min(1.0, (0.5 - gravy) / 1.0))
    scores.append(solubility_score)
    
    # Low instability index
    instability = analyzer.instability_index()
    instab_score = max(0, (50 - instability) / 50)
    scores.append(instab_score)
    
    # Check for problematic motifs
    motif_score = 1.0
    
    # N-glycosylation sites
    import re
    if re.search(r'N[^P][ST]', sequence):
        motif_score -= 0.2
    
    # Deamidation sites
    if re.search(r'N[GS]', sequence):
        motif_score -= 0.1
    
    # Oxidation-prone Met
    if sequence.count('M') > 2:
        motif_score -= 0.1
    
    # Free cysteines
    cys_count = sequence.count('C')
    if cys_count != 2 and cys_count != 4:
        motif_score -= 0.2
    
    scores.append(max(0, motif_score))
    
    # Average all scores
    return np.mean(scores)

def predict_expression(sequence: str) -> float:
    """
    Predict expression level in E. coli
    
    Args:
        sequence: Amino acid sequence
    
    Returns:
        Expression score (0-1, higher is better)
    """
    
    # Factors affecting expression
    expr_score = 1.0
    
    # Rare codons (approximated by rare amino acids)
    rare_aa = 'WCM'
    rare_fraction = sum(1 for aa in sequence if aa in rare_aa) / len(sequence)
    expr_score -= rare_fraction * 2  # Heavily penalize rare amino acids
    
    # Hydrophobicity (can cause inclusion bodies)
    hydrophobic = 'FWYLIMV'
    hydro_fraction = sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)
    if hydro_fraction > 0.4:
        expr_score -= (hydro_fraction - 0.4) * 2
    
    # Size penalty
    if len(sequence) > 130:
        expr_score -= (len(sequence) - 130) / 100
    
    # Charge distribution (extreme pI bad for expression)
    analyzer = ProtParam.ProteinAnalysis(sequence)
    pi = analyzer.isoelectric_point()
    if pi < 5.0 or pi > 10.0:
        expr_score -= 0.3
    
    # Proline content (can affect folding)
    pro_fraction = sequence.count('P') / len(sequence)
    if pro_fraction > 0.1:
        expr_score -= (pro_fraction - 0.1) * 2
    
    return max(0, min(1.0, expr_score))

def assess_specificity(sequence: str, off_target_sequence: Optional[str] = None) -> float:
    """
    Assess specificity against off-targets
    
    Args:
        sequence: Nanobody sequence
        off_target_sequence: Off-target sequence (e.g., PTN)
    
    Returns:
        Specificity score (0-1, higher is better)
    """
    
    if not off_target_sequence:
        return 1.0  # No off-target to check
    
    # Extract likely binding regions (CDRs)
    # Approximate CDR positions
    cdr1 = sequence[26:36] if len(sequence) > 36 else ""
    cdr2 = sequence[50:66] if len(sequence) > 66 else ""
    cdr3 = sequence[95:111] if len(sequence) > 111 else ""
    
    cdr_combined = cdr1 + cdr2 + cdr3
    
    if not cdr_combined:
        return 0.5  # Can't assess
    
    # Check for MDK-specific motifs
    mdk_specific = ['RGR', 'KKK', 'RKR']
    specificity_score = 0.5  # Baseline
    
    for motif in mdk_specific:
        if motif in cdr_combined:
            specificity_score += 0.15
    
    # Check similarity to PTN
    if off_target_sequence:
        # Simple sequence similarity check
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, cdr_combined, off_target_sequence).ratio()
        
        # Lower similarity to PTN is better
        specificity_score *= (1.0 - similarity)
    
    return min(1.0, specificity_score)

def rank_designs(validation_results: List[ValidationResult]) -> List[ValidationResult]:
    """
    Rank designs based on validation results
    
    Args:
        validation_results: List of validation results
    
    Returns:
        Sorted list of validation results
    """
    
    # Filter passed designs
    passed = [r for r in validation_results if r.passed]
    failed = [r for r in validation_results if not r.passed]
    
    # Sort by overall score
    passed_sorted = sorted(passed, key=lambda x: x.overall_score, reverse=True)
    failed_sorted = sorted(failed, key=lambda x: x.overall_score, reverse=True)
    
    return passed_sorted + failed_sorted

def generate_validation_report(
    validation_results: List[ValidationResult],
    output_file: Path
):
    """
    Generate validation report
    
    Args:
        validation_results: List of validation results
        output_file: Output file path
    """
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Nanobody Design Validation Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary statistics
        total = len(validation_results)
        passed = sum(1 for r in validation_results if r.passed)
        
        f.write(f"Total designs: {total}\n")
        f.write(f"Passed: {passed} ({passed/total*100:.1f}%)\n")
        f.write(f"Failed: {total-passed} ({(total-passed)/total*100:.1f}%)\n\n")
        
        # Score distributions
        metrics = ['developability', 'expression', 'specificity', 'stability']
        
        f.write("Score distributions:\n")
        for metric in metrics:
            scores = [r.scores.get(metric, 0) for r in validation_results]
            if scores:
                f.write(f"  {metric}: {np.mean(scores):.2f} ± {np.std(scores):.2f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("Individual Results:\n")
        f.write("-" * 40 + "\n\n")
        
        # Sort by overall score
        sorted_results = sorted(
            validation_results,
            key=lambda x: x.overall_score,
            reverse=True
        )
        
        for i, result in enumerate(sorted_results[:20], 1):
            f.write(f"{i}. {result.design_id}\n")
            f.write(f"   Overall score: {result.overall_score:.3f}\n")
            f.write(f"   Status: {'PASSED' if result.passed else 'FAILED'}\n")
            
            f.write("   Scores:\n")
            for metric, score in result.scores.items():
                f.write(f"     {metric}: {score:.3f}\n")
            
            if result.warnings:
                f.write("   Warnings:\n")
                for warning in result.warnings:
                    f.write(f"     - {warning}\n")
            
            if result.errors:
                f.write("   Errors:\n")
                for error in result.errors:
                    f.write(f"     - {error}\n")
            
            f.write("\n")
    
    print(f"Validation report saved to {output_file}")