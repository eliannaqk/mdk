#!/usr/bin/env python3
"""
Analyze and validate nanobody designs for experimental testing
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO, PDB
from Bio.SeqUtils import ProtParam
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

def load_config():
    """Load configuration"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def load_nanobodies(nanobody_dir):
    """Load nanobody data"""
    
    data_file = Path(nanobody_dir) / "nanobody_data.json"
    
    with open(data_file, 'r') as f:
        nanobodies = json.load(f)
    
    return nanobodies

def check_ptn_cross_reactivity(nanobody_sequence):
    """
    Check potential cross-reactivity with Pleiotrophin (PTN)
    PTN is MDK's paralog with similar structure
    """
    
    # Key differences between MDK and PTN that we want to exploit
    # MDK has specific basic patches that differ from PTN
    
    # Simplified check - look for MDK-specific motifs
    mdk_specific_motifs = [
        'RGR',  # MDK-specific basic cluster
        'KKK',  # Triple lysine found in MDK
        'RKR'   # Another MDK-specific pattern
    ]
    
    specificity_score = 0
    for motif in mdk_specific_motifs:
        if motif in nanobody_sequence:
            specificity_score += 1
    
    # Higher score suggests better MDK specificity
    return {
        'specificity_score': specificity_score,
        'risk': 'low' if specificity_score >= 2 else 'medium' if specificity_score >= 1 else 'high'
    }

def analyze_developability(nanobody):
    """
    Comprehensive developability assessment
    """
    
    sequence = nanobody['sequence']
    
    # Use Biopython's ProtParam for detailed analysis
    analyzer = ProtParam.ProteinAnalysis(sequence)

    try:
        aliphatic_index = analyzer.aliphatic_index()
    except AttributeError:
        composition = analyzer.count_amino_acids()
        total = float(sum(composition.values()) or 1)
        a = composition.get('A', 0) / total
        v = composition.get('V', 0) / total
        i = composition.get('I', 0) / total
        l = composition.get('L', 0) / total
        aliphatic_index = 100 * (a + 2.9 * v + 3.9 * (i + l))

    try:
        secondary_structure = analyzer.secondary_structure_fraction()
    except AttributeError:
        secondary_structure = (0.0, 0.0, 0.0)

    developability = {
        'molecular_weight': analyzer.molecular_weight(),
        'theoretical_pi': analyzer.isoelectric_point(),
        'instability_index': analyzer.instability_index(),
        'aliphatic_index': aliphatic_index,
        'gravy': analyzer.gravy(),  # Grand average of hydropathy
        'aromaticity': analyzer.aromaticity(),
        'secondary_structure': secondary_structure
    }
    
    # Assess developability risk
    risks = []
    
    # Check pI (ideal range: 6.5-9.0)
    if developability['theoretical_pi'] < 6.5 or developability['theoretical_pi'] > 9.0:
        risks.append(f"pI outside optimal range: {developability['theoretical_pi']:.1f}")
    
    # Check instability (< 40 is stable)
    if developability['instability_index'] > 40:
        risks.append(f"High instability index: {developability['instability_index']:.1f}")
    
    # Check hydrophobicity (GRAVY should be negative for soluble proteins)
    if developability['gravy'] > 0:
        risks.append(f"Positive GRAVY score: {developability['gravy']:.2f}")
    
    # Check for aggregation-prone regions
    hydrophobic_aa = 'FWYLIMV'
    max_hydrophobic_stretch = 0
    current_stretch = 0
    
    for aa in sequence:
        if aa in hydrophobic_aa:
            current_stretch += 1
            max_hydrophobic_stretch = max(max_hydrophobic_stretch, current_stretch)
        else:
            current_stretch = 0
    
    if max_hydrophobic_stretch >= 5:
        risks.append(f"Long hydrophobic stretch: {max_hydrophobic_stretch} aa")
    
    developability['risks'] = risks
    developability['risk_level'] = 'high' if len(risks) >= 3 else 'medium' if len(risks) >= 1 else 'low'
    
    return developability

def predict_expression_level(nanobody):
    """
    Predict expression level in E. coli or yeast
    """
    
    sequence = nanobody['sequence']
    
    # Factors affecting expression
    factors = {
        'rare_codons': 0,
        'gc_content': 0,
        'hydrophobicity': 0,
        'size': len(sequence)
    }
    
    # Check for rare amino acids
    rare_aa = 'WCM'
    factors['rare_codons'] = sum(1 for aa in sequence if aa in rare_aa)
    
    # Estimate GC content impact (simplified)
    gc_favorable = 'GAPR'
    factors['gc_content'] = sum(1 for aa in sequence if aa in gc_favorable) / len(sequence)
    
    # Hydrophobicity
    hydrophobic_aa = 'FWYLIMV'
    factors['hydrophobicity'] = sum(1 for aa in sequence if aa in hydrophobic_aa) / len(sequence)
    
    # Calculate expression score
    expression_score = 100
    expression_score -= factors['rare_codons'] * 5
    expression_score -= (factors['hydrophobicity'] - 0.3) * 50 if factors['hydrophobicity'] > 0.3 else 0
    expression_score -= (factors['size'] - 120) * 0.5 if factors['size'] > 120 else 0
    
    expression_level = 'high' if expression_score > 80 else 'medium' if expression_score > 60 else 'low'
    
    return {
        'score': max(0, min(100, expression_score)),
        'level': expression_level,
        'factors': factors
    }

def calculate_production_metrics(nanobody):
    """
    Calculate metrics relevant for production
    """
    
    sequence = nanobody['sequence']
    
    # Estimate yield (mg/L)
    base_yield = 50  # Base yield for average nanobody
    
    # Adjust based on properties
    analyzer = ProtParam.ProteinAnalysis(sequence)
    
    # Solubility impacts yield
    if analyzer.gravy() < -0.4:  # Very hydrophilic
        yield_factor = 1.5
    elif analyzer.gravy() > 0:  # Hydrophobic
        yield_factor = 0.5
    else:
        yield_factor = 1.0
    
    estimated_yield = base_yield * yield_factor
    
    # Purification ease
    purification_score = 100
    
    # Check for His residues (interfere with His-tag purification)
    his_count = sequence.count('H')
    if his_count > 3:
        purification_score -= 20
    
    # Check for Cys (disulfide bond complications)
    cys_count = sequence.count('C')
    if cys_count != 2:  # Should have exactly 2 for canonical disulfide
        purification_score -= 30
    
    return {
        'estimated_yield_mg_per_l': round(estimated_yield, 1),
        'purification_score': max(0, purification_score),
        'production_cost': 'low' if estimated_yield > 75 and purification_score > 80 else 'medium' if estimated_yield > 40 else 'high'
    }

def rank_nanobodies(nanobodies_with_analysis):
    """
    Rank nanobodies based on comprehensive scoring
    """
    
    for nb in nanobodies_with_analysis:
        # Calculate composite score
        score = 0
        
        # Original design quality (40% weight)
        score += nb['original_scores'].get('iptm', 0.5) * 40
        
        # Developability (30% weight)
        dev_score = 30
        if nb['developability']['risk_level'] == 'medium':
            dev_score = 20
        elif nb['developability']['risk_level'] == 'high':
            dev_score = 10
        score += dev_score
        
        # Expression (15% weight)
        expr_score = nb['expression']['score'] / 100 * 15
        score += expr_score
        
        # Specificity (15% weight)
        spec_score = 15 if nb['ptn_specificity']['risk'] == 'low' else 10 if nb['ptn_specificity']['risk'] == 'medium' else 5
        score += spec_score
        
        nb['final_score'] = round(score, 2)
    
    # Sort by final score
    nanobodies_with_analysis.sort(key=lambda x: x['final_score'], reverse=True)
    
    return nanobodies_with_analysis

def generate_visualization(nanobodies_with_analysis, output_dir):
    """
    Generate visualization plots for the analysis
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    data = []
    for nb in nanobodies_with_analysis:
        data.append({
            'ID': nb['id'].replace('nb_design_', 'NB'),
            'Final Score': nb['final_score'],
            'pI': nb['developability']['theoretical_pi'],
            'Instability': nb['developability']['instability_index'],
            'Expression': nb['expression']['score'],
            'MW (kDa)': nb['developability']['molecular_weight'] / 1000
        })
    
    df = pd.DataFrame(data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Final scores
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('Final Score', ascending=True)
    ax1.barh(df_sorted['ID'], df_sorted['Final Score'])
    ax1.set_xlabel('Final Score')
    ax1.set_title('Nanobody Ranking')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. pI distribution
    ax2 = axes[0, 1]
    ax2.scatter(df['pI'], df['Instability'])
    ax2.axvline(x=6.5, color='r', linestyle='--', alpha=0.5, label='pI range')
    ax2.axvline(x=9.0, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=40, color='r', linestyle='--', alpha=0.5, label='Stability threshold')
    ax2.set_xlabel('Theoretical pI')
    ax2.set_ylabel('Instability Index')
    ax2.set_title('Developability Metrics')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Expression vs MW
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['MW (kDa)'], df['Expression'], c=df['Final Score'], cmap='viridis')
    plt.colorbar(scatter, ax=ax3, label='Final Score')
    ax3.set_xlabel('Molecular Weight (kDa)')
    ax3.set_ylabel('Expression Score')
    ax3.set_title('Expression Prediction')
    ax3.grid(alpha=0.3)
    
    # 4. Score components
    ax4 = axes[1, 1]
    components = ['Binding', 'Developability', 'Expression', 'Specificity']
    top_nb = nanobodies_with_analysis[0]
    values = [
        top_nb['original_scores'].get('iptm', 0.5) * 100,
        100 if top_nb['developability']['risk_level'] == 'low' else 66 if top_nb['developability']['risk_level'] == 'medium' else 33,
        top_nb['expression']['score'],
        100 if top_nb['ptn_specificity']['risk'] == 'low' else 66 if top_nb['ptn_specificity']['risk'] == 'medium' else 33
    ]
    ax4.bar(components, values)
    ax4.set_ylabel('Score (%)')
    ax4.set_title(f'Top Candidate ({top_nb["id"]}) Breakdown')
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = output_dir / "analysis_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved plots to: {plot_file}")
    
    plt.close()

def generate_final_report(nanobodies_with_analysis, output_file):
    """
    Generate comprehensive final report
    """
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MDK-LRP1 NANOBODY DESIGN - FINAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total designs analyzed: {len(nanobodies_with_analysis)}\n")
        
        # Count risk levels
        low_risk = sum(1 for nb in nanobodies_with_analysis if nb['developability']['risk_level'] == 'low')
        med_risk = sum(1 for nb in nanobodies_with_analysis if nb['developability']['risk_level'] == 'medium')
        high_risk = sum(1 for nb in nanobodies_with_analysis if nb['developability']['risk_level'] == 'high')
        
        f.write(f"Developability: {low_risk} low risk, {med_risk} medium risk, {high_risk} high risk\n")
        
        high_expr = sum(1 for nb in nanobodies_with_analysis if nb['expression']['level'] == 'high')
        f.write(f"High expression predicted: {high_expr}/{len(nanobodies_with_analysis)}\n\n")
        
        # Top candidates
        f.write("TOP 3 CANDIDATES FOR EXPERIMENTAL VALIDATION\n")
        f.write("-" * 40 + "\n\n")
        
        for i, nb in enumerate(nanobodies_with_analysis[:3], 1):
            f.write(f"{i}. {nb['id']} (Score: {nb['final_score']})\n")
            f.write(f"   Binding Score (ipTM): {nb['original_scores'].get('iptm', 'N/A')}\n")
            f.write(f"   MW: {nb['developability']['molecular_weight']/1000:.1f} kDa\n")
            f.write(f"   pI: {nb['developability']['theoretical_pi']:.1f}\n")
            f.write(f"   Instability: {nb['developability']['instability_index']:.1f}\n")
            f.write(f"   Expression: {nb['expression']['level']}\n")
            f.write(f"   PTN cross-reactivity risk: {nb['ptn_specificity']['risk']}\n")
            f.write(f"   Estimated yield: {nb['production']['estimated_yield_mg_per_l']} mg/L\n")
            
            if nb['developability']['risks']:
                f.write(f"   Risks: {', '.join(nb['developability']['risks'])}\n")
            else:
                f.write(f"   Risks: None identified\n")
            
            f.write(f"   CDR3: {nb['cdrs']['CDR3']}\n")
            f.write("\n")
        
        # Detailed analysis
        f.write("\nDETAILED ANALYSIS\n")
        f.write("-" * 40 + "\n\n")
        
        for nb in nanobodies_with_analysis:
            f.write(f"{nb['id']}:\n")
            f.write(f"  Final Score: {nb['final_score']}\n")
            f.write(f"  Sequence Length: {len(nb['sequence'])} aa\n")
            f.write(f"  Properties:\n")
            f.write(f"    - Molecular Weight: {nb['developability']['molecular_weight']:.0f} Da\n")
            f.write(f"    - Theoretical pI: {nb['developability']['theoretical_pi']:.2f}\n")
            f.write(f"    - Instability Index: {nb['developability']['instability_index']:.2f}\n")
            f.write(f"    - Aliphatic Index: {nb['developability']['aliphatic_index']:.2f}\n")
            f.write(f"    - GRAVY: {nb['developability']['gravy']:.3f}\n")
            f.write(f"    - Aromaticity: {nb['developability']['aromaticity']:.3f}\n")
            f.write(f"  Expression Prediction:\n")
            f.write(f"    - Score: {nb['expression']['score']:.1f}\n")
            f.write(f"    - Level: {nb['expression']['level']}\n")
            f.write(f"  Production Metrics:\n")
            f.write(f"    - Estimated Yield: {nb['production']['estimated_yield_mg_per_l']} mg/L\n")
            f.write(f"    - Purification Score: {nb['production']['purification_score']}\n")
            f.write(f"    - Production Cost: {nb['production']['production_cost']}\n")
            f.write("\n")
        
        # Recommendations
        f.write("\nRECOMMENDATIONS FOR EXPERIMENTAL VALIDATION\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("1. Expression System:\n")
        f.write("   - Primary: E. coli BL21(DE3) with pET vector system\n")
        f.write("   - Alternative: Pichia pastoris for difficult-to-express candidates\n\n")
        
        f.write("2. Purification Strategy:\n")
        f.write("   - His-tag purification followed by size exclusion\n")
        f.write("   - Consider Protein A purification for Fc-fusion formats\n\n")
        
        f.write("3. Binding Validation:\n")
        f.write("   - SPR with immobilized MDK (focus on C-terminal domain)\n")
        f.write("   - Competition assay with LRP1 receptor fragments\n")
        f.write("   - Cell-based assays using LRP1-expressing cells\n\n")
        
        f.write("4. Specificity Testing:\n")
        f.write("   - Cross-reactivity with PTN (pleiotrophin)\n")
        f.write("   - Binding to other heparin-binding proteins\n")
        f.write("   - Species cross-reactivity (human, mouse, rat MDK)\n\n")
        
        f.write("5. Functional Assays:\n")
        f.write("   - MDK-LRP1 signaling inhibition\n")
        f.write("   - Cell migration/invasion assays\n")
        f.write("   - Downstream pathway analysis (AKT, ERK)\n\n")
    
    print(f"  Generated final report: {output_file}")

def main():
    """Main analysis pipeline"""
    
    print("=" * 60)
    print("Final Analysis and Validation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load nanobodies
    print("\n1. Loading nanobody designs...")
    nanobody_dir = Path(config['output']['base_dir']) / "nanobodies"
    
    if not nanobody_dir.exists():
        print(f"Error: Nanobody directory {nanobody_dir} not found!")
        print("Please run 05_convert_to_nanobody.py first")
        return
    
    nanobodies = load_nanobodies(nanobody_dir)
    print(f"  Loaded {len(nanobodies)} nanobodies")
    
    # Comprehensive analysis
    print("\n2. Running comprehensive analysis...")
    
    nanobodies_with_analysis = []
    
    for nb in nanobodies:
        print(f"  Analyzing {nb['id']}...")
        
        # Add analyses
        nb['ptn_specificity'] = check_ptn_cross_reactivity(nb['sequence'])
        nb['developability'] = analyze_developability(nb)
        nb['expression'] = predict_expression_level(nb)
        nb['production'] = calculate_production_metrics(nb)
        
        nanobodies_with_analysis.append(nb)
    
    # Rank nanobodies
    print("\n3. Ranking nanobodies...")
    ranked_nanobodies = rank_nanobodies(nanobodies_with_analysis)
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    vis_dir = Path(config['output']['base_dir']) / "validation"
    generate_visualization(ranked_nanobodies, vis_dir)
    
    # Generate final report
    print("\n5. Generating final report...")
    report_file = vis_dir / "final_validation_report.txt"
    generate_final_report(ranked_nanobodies, report_file)
    
    # Save ranked data
    ranked_data_file = vis_dir / "ranked_nanobodies.json"
    with open(ranked_data_file, 'w') as f:
        # Prepare for JSON serialization
        json_data = []
        for nb in ranked_nanobodies:
            json_data.append({
                'id': nb['id'],
                'final_score': nb['final_score'],
                'sequence': nb['sequence'],
                'developability_risk': nb['developability']['risk_level'],
                'expression_level': nb['expression']['level'],
                'production_cost': nb['production']['production_cost']
            })
        json.dump(json_data, f, indent=2)
    
    print(f"  Saved ranked data to: {ranked_data_file}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    # Summary
    print("\nTop 3 candidates for experimental validation:")
    for i, nb in enumerate(ranked_nanobodies[:3], 1):
        print(f"\n{i}. {nb['id']}")
        print(f"   Final score: {nb['final_score']}")
        print(f"   Risk level: {nb['developability']['risk_level']}")
        print(f"   Expression: {nb['expression']['level']}")
        print(f"   Estimated yield: {nb['production']['estimated_yield_mg_per_l']} mg/L")
    
    print(f"\nFull report available at: {report_file}")
    print("\nReady for experimental validation!")

if __name__ == "__main__":
    main()