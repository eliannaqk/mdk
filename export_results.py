#!/usr/bin/env python3
"""
Export MDK-LRP1 nanobody design results in various formats
for publication, presentation, and experimental validation
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import argparse
import zipfile
import csv

class ResultsExporter:
    """Export pipeline results in multiple formats"""
    
    def __init__(self, results_dir: Path, output_dir: Path = None):
        self.results_dir = Path(results_dir)
        self.output_dir = output_dir or Path("exports") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_all(self):
        """Export results in all available formats"""
        
        print("=" * 60)
        print("MDK-LRP1 Nanobody Results Export")
        print("=" * 60)
        
        # Load results
        print("\n1. Loading results...")
        results = self.load_results()
        
        if not results:
            print("No results found to export!")
            return
        
        # Export formats
        print("\n2. Exporting results...")
        
        # Scientific formats
        self.export_genbank(results)
        self.export_csv_tables(results)
        self.export_excel_workbook(results)
        
        # Visualization formats
        self.export_publication_figures(results)
        self.export_pymol_sessions(results)
        
        # Documentation formats
        self.export_latex_tables(results)
        self.export_markdown_report(results)
        
        # Lab formats
        self.export_benchling_format(results)
        self.export_plate_layout(results)
        
        # Archive everything
        self.create_archive()
        
        print("\n" + "=" * 60)
        print("Export Complete!")
        print("=" * 60)
        print(f"\nAll exports saved to: {self.output_dir}")
    
    def load_results(self) -> Dict:
        """Load all results from pipeline"""
        
        results = {}
        
        # Load nanobody data
        nb_file = self.results_dir / "nanobodies/nanobody_data.json"
        if nb_file.exists():
            with open(nb_file, 'r') as f:
                results['nanobodies'] = json.load(f)
        
        # Load validation data
        val_file = self.results_dir / "validation/ranked_nanobodies.json"
        if val_file.exists():
            with open(val_file, 'r') as f:
                results['validation'] = json.load(f)
        
        # Load epitope data
        epitope_file = Path("data/epitopes/lrp1_patch.json")
        if epitope_file.exists():
            with open(epitope_file, 'r') as f:
                results['epitope'] = json.load(f)
        
        print(f"  Loaded {len(results.get('nanobodies', []))} nanobody designs")
        
        return results
    
    def export_genbank(self, results: Dict):
        """Export sequences in GenBank format"""
        
        print("  Exporting GenBank format...")
        
        genbank_dir = self.output_dir / "genbank"
        genbank_dir.mkdir(exist_ok=True)
        
        for nb in results.get('nanobodies', [])[:10]:  # Top 10
            record = SeqRecord(
                Seq(nb['sequence']),
                id=nb['id'],
                name=nb['id'],
                description=f"Anti-MDK nanobody targeting LRP1 epitope",
                annotations={
                    'molecule_type': 'protein',
                    'organism': 'synthetic construct',
                    'note': f"Designed using BindCraft; MW={nb['properties']['molecular_weight_kda']}kDa, pI={nb['properties']['theoretical_pi']}"
                }
            )
            
            # Add CDR annotations
            if 'cdrs' in nb:
                # Find CDR positions in sequence
                seq = nb['sequence']
                for cdr_name, cdr_seq in nb['cdrs'].items():
                    start = seq.find(cdr_seq)
                    if start >= 0:
                        from Bio.SeqFeature import SeqFeature, FeatureLocation
                        feature = SeqFeature(
                            FeatureLocation(start, start + len(cdr_seq)),
                            type="misc_feature",
                            qualifiers={'label': cdr_name, 'note': f'{cdr_name} region'}
                        )
                        record.features.append(feature)
            
            # Save individual GenBank file
            gb_file = genbank_dir / f"{nb['id']}.gb"
            SeqIO.write([record], gb_file, "genbank")
        
        print(f"    Saved GenBank files to {genbank_dir}")
    
    def export_csv_tables(self, results: Dict):
        """Export structured CSV tables"""
        
        print("  Exporting CSV tables...")
        
        csv_dir = self.output_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Main results table
        main_data = []
        for nb in results.get('nanobodies', []):
            main_data.append({
                'ID': nb['id'],
                'Sequence': nb['sequence'],
                'Length': len(nb['sequence']),
                'MW_kDa': nb['properties']['molecular_weight_kda'],
                'pI': nb['properties']['theoretical_pi'],
                'CDR1': nb['cdrs'].get('CDR1', ''),
                'CDR2': nb['cdrs'].get('CDR2', ''),
                'CDR3': nb['cdrs'].get('CDR3', ''),
                'Original_ipTM': nb['original_scores'].get('iptm', ''),
                'Original_pLDDT': nb['original_scores'].get('plddt', '')
            })
        
        df_main = pd.DataFrame(main_data)
        df_main.to_csv(csv_dir / "nanobody_sequences.csv", index=False)
        
        # Properties table
        props_data = []
        for nb in results.get('nanobodies', []):
            if 'properties' in nb:
                props = nb['properties'].copy()
                props['ID'] = nb['id']
                props_data.append(props)
        
        if props_data:
            df_props = pd.DataFrame(props_data)
            df_props.to_csv(csv_dir / "nanobody_properties.csv", index=False)
        
        # Epitope information
        if 'epitope' in results:
            epitope_df = pd.DataFrame({
                'Residue': results['epitope']['expanded_residues'],
                'Type': ['Core' if r in results['epitope']['core_residues'] else 'Expanded' 
                        for r in results['epitope']['expanded_residues']]
            })
            epitope_df.to_csv(csv_dir / "epitope_residues.csv", index=False)
        
        print(f"    Saved CSV files to {csv_dir}")
    
    def export_excel_workbook(self, results: Dict):
        """Export comprehensive Excel workbook"""
        
        print("  Exporting Excel workbook...")
        
        excel_file = self.output_dir / "mdk_nanobody_results.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Overview sheet
            overview_data = {
                'Metric': ['Total Designs', 'Top Score', 'Average MW (kDa)', 'Average pI'],
                'Value': [
                    len(results.get('nanobodies', [])),
                    results['nanobodies'][0]['original_scores'].get('iptm', 'N/A') if results.get('nanobodies') else 'N/A',
                    np.mean([nb['properties']['molecular_weight_kda'] for nb in results.get('nanobodies', [])]) if results.get('nanobodies') else 0,
                    np.mean([nb['properties']['theoretical_pi'] for nb in results.get('nanobodies', [])]) if results.get('nanobodies') else 0
                ]
            }
            pd.DataFrame(overview_data).to_excel(writer, sheet_name='Overview', index=False)
            
            # Sequences sheet
            seq_data = []
            for nb in results.get('nanobodies', []):
                seq_data.append({
                    'ID': nb['id'],
                    'Sequence': nb['sequence'],
                    'Length': len(nb['sequence'])
                })
            pd.DataFrame(seq_data).to_excel(writer, sheet_name='Sequences', index=False)
            
            # Properties sheet
            props_data = []
            for nb in results.get('nanobodies', []):
                props_data.append({
                    'ID': nb['id'],
                    'MW (kDa)': nb['properties']['molecular_weight_kda'],
                    'pI': nb['properties']['theoretical_pi'],
                    'Net Charge': nb['properties'].get('net_charge', 'N/A')
                })
            pd.DataFrame(props_data).to_excel(writer, sheet_name='Properties', index=False)
            
            # CDRs sheet
            cdr_data = []
            for nb in results.get('nanobodies', []):
                cdr_data.append({
                    'ID': nb['id'],
                    'CDR1': nb['cdrs'].get('CDR1', ''),
                    'CDR1_Length': len(nb['cdrs'].get('CDR1', '')),
                    'CDR2': nb['cdrs'].get('CDR2', ''),
                    'CDR2_Length': len(nb['cdrs'].get('CDR2', '')),
                    'CDR3': nb['cdrs'].get('CDR3', ''),
                    'CDR3_Length': len(nb['cdrs'].get('CDR3', ''))
                })
            pd.DataFrame(cdr_data).to_excel(writer, sheet_name='CDRs', index=False)
        
        print(f"    Saved Excel workbook: {excel_file}")
    
    def export_publication_figures(self, results: Dict):
        """Generate publication-quality figures"""
        
        print("  Generating publication figures...")
        
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Figure 1: Properties distribution
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # MW distribution
        mws = [nb['properties']['molecular_weight_kda'] for nb in results.get('nanobodies', [])]
        axes[0, 0].hist(mws, bins=15, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Molecular Weight (kDa)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('MW Distribution')
        
        # pI distribution
        pis = [nb['properties']['theoretical_pi'] for nb in results.get('nanobodies', [])]
        axes[0, 1].hist(pis, bins=15, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=7.4, color='r', linestyle='--', label='Physiological pH')
        axes[0, 1].set_xlabel('Theoretical pI')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('pI Distribution')
        axes[0, 1].legend()
        
        # Length distribution
        lengths = [len(nb['sequence']) for nb in results.get('nanobodies', [])]
        axes[1, 0].hist(lengths, bins=15, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Sequence Length (aa)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Length Distribution')
        
        # CDR3 length distribution
        cdr3_lengths = [len(nb['cdrs'].get('CDR3', '')) for nb in results.get('nanobodies', [])]
        axes[1, 1].hist(cdr3_lengths, bins=10, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('CDR3 Length (aa)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('CDR3 Length Distribution')
        
        plt.tight_layout()
        plt.savefig(fig_dir / "properties_distribution.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(fig_dir / "properties_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved figures to {fig_dir}")
    
    def export_latex_tables(self, results: Dict):
        """Export LaTeX tables for manuscripts"""
        
        print("  Exporting LaTeX tables...")
        
        latex_dir = self.output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)
        
        # Top candidates table
        latex_file = latex_dir / "top_candidates.tex"
        
        with open(latex_file, 'w') as f:
            f.write("% Top MDK-LRP1 Nanobody Candidates\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Top nanobody candidates targeting MDK LRP1 epitope}\n")
            f.write("\\begin{tabular}{llccc}\n")
            f.write("\\hline\n")
            f.write("ID & CDR3 Sequence & MW (kDa) & pI & Score \\\\\n")
            f.write("\\hline\n")
            
            for nb in results.get('nanobodies', [])[:5]:
                f.write(f"{nb['id']} & {nb['cdrs'].get('CDR3', 'N/A')[:10]}... & ")
                f.write(f"{nb['properties']['molecular_weight_kda']:.1f} & ")
                f.write(f"{nb['properties']['theoretical_pi']:.1f} & ")
                f.write(f"{nb['original_scores'].get('iptm', 'N/A')} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:nanobodies}\n")
            f.write("\\end{table}\n")
        
        print(f"    Saved LaTeX tables to {latex_dir}")
    
    def export_markdown_report(self, results: Dict):
        """Export markdown report for GitHub/documentation"""
        
        print("  Exporting Markdown report...")
        
        md_file = self.output_dir / "results_report.md"
        
        with open(md_file, 'w') as f:
            f.write("# MDK-LRP1 Nanobody Design Results\n\n")
            
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total designs: {len(results.get('nanobodies', []))}\n")
            f.write(f"- Target epitope: LRP1 binding patch\n")
            f.write(f"- Epitope size: {len(results.get('epitope', {}).get('expanded_residues', []))} residues\n\n")
            
            f.write("## Top 5 Candidates\n\n")
            f.write("| Rank | ID | MW (kDa) | pI | CDR3 |\n")
            f.write("|------|----|---------|----|------|\n")
            
            for i, nb in enumerate(results.get('nanobodies', [])[:5], 1):
                f.write(f"| {i} | {nb['id']} | ")
                f.write(f"{nb['properties']['molecular_weight_kda']:.1f} | ")
                f.write(f"{nb['properties']['theoretical_pi']:.1f} | ")
                f.write(f"{nb['cdrs'].get('CDR3', 'N/A')} |\n")
            
            f.write("\n## Sequences\n\n")
            for nb in results.get('nanobodies', [])[:3]:
                f.write(f"### {nb['id']}\n")
                f.write("```\n")
                f.write(f"{nb['sequence']}\n")
                f.write("```\n\n")
        
        print(f"    Saved Markdown report: {md_file}")
    
    def export_benchling_format(self, results: Dict):
        """Export for Benchling LIMS"""
        
        print("  Exporting Benchling format...")
        
        benchling_dir = self.output_dir / "benchling"
        benchling_dir.mkdir(exist_ok=True)
        
        # Benchling-compatible CSV
        benchling_data = []
        
        for nb in results.get('nanobodies', [])[:20]:
            benchling_data.append({
                'Name': nb['id'],
                'Bases': '',  # DNA sequence would go here
                'AA Sequence': nb['sequence'],
                'Type': 'Nanobody',
                'Folder': 'MDK-LRP1 Campaign',
                'Description': f"Anti-MDK nanobody, MW={nb['properties']['molecular_weight_kda']}kDa, pI={nb['properties']['theoretical_pi']}",
                'Tags': 'MDK,LRP1,Nanobody,BindCraft'
            })
        
        df = pd.DataFrame(benchling_data)
        df.to_csv(benchling_dir / "benchling_import.csv", index=False)
        
        print(f"    Saved Benchling import file")
    
    def export_plate_layout(self, results: Dict):
        """Export 96-well plate layout for experiments"""
        
        print("  Exporting plate layout...")
        
        plate_dir = self.output_dir / "plate_layouts"
        plate_dir.mkdir(exist_ok=True)
        
        # Create 96-well plate layout
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        cols = list(range(1, 13))
        
        plate_data = []
        nb_index = 0
        
        for row in rows:
            for col in cols:
                well = f"{row}{col}"
                
                if nb_index < len(results.get('nanobodies', [])):
                    nb = results['nanobodies'][nb_index]
                    plate_data.append({
                        'Well': well,
                        'Sample': nb['id'],
                        'Type': 'Nanobody',
                        'Concentration': '1 mg/mL'
                    })
                    nb_index += 1
                else:
                    plate_data.append({
                        'Well': well,
                        'Sample': 'Empty',
                        'Type': 'Control',
                        'Concentration': ''
                    })
        
        df_plate = pd.DataFrame(plate_data)
        df_plate.to_csv(plate_dir / "96well_layout.csv", index=False)
        
        # Create visual plate map
        plate_matrix = []
        for row in rows:
            row_data = []
            for col in cols:
                well = f"{row}{col}"
                sample = df_plate[df_plate['Well'] == well]['Sample'].values[0]
                row_data.append(sample if sample != 'Empty' else '')
            plate_matrix.append(row_data)
        
        df_visual = pd.DataFrame(plate_matrix, index=rows, columns=cols)
        df_visual.to_csv(plate_dir / "plate_map.csv")
        
        print(f"    Saved plate layouts to {plate_dir}")
    
    def export_pymol_sessions(self, results: Dict):
        """Export PyMOL visualization scripts"""
        
        print("  Exporting PyMOL scripts...")
        
        pymol_dir = self.output_dir / "pymol"
        pymol_dir.mkdir(exist_ok=True)
        
        # Create master visualization script
        script_file = pymol_dir / "visualize_all.pml"
        
        with open(script_file, 'w') as f:
            f.write("# PyMOL script for MDK-LRP1 nanobody visualization\n\n")
            f.write("# Load MDK structure\n")
            f.write("load data/structures/1mkc.pdb, MDK\n\n")
            
            f.write("# Style settings\n")
            f.write("hide everything\n")
            f.write("show cartoon, MDK\n")
            f.write("color grey80, MDK\n\n")
            
            f.write("# Highlight LRP1 epitope\n")
            if 'epitope' in results:
                residues = results['epitope']['expanded_residues']
                f.write(f"select lrp1_epitope, MDK and resi {'+'.join(map(str, residues))}\n")
                f.write("show sticks, lrp1_epitope\n")
                f.write("color red, lrp1_epitope\n\n")
            
            f.write("# Set view\n")
            f.write("zoom MDK\n")
            f.write("orient\n")
            f.write("bg_color white\n")
        
        print(f"    Saved PyMOL scripts to {pymol_dir}")
    
    def create_archive(self):
        """Create ZIP archive of all exports"""
        
        print("  Creating archive...")
        
        archive_file = self.output_dir.parent / f"mdk_nanobody_results_{datetime.now().strftime('%Y%m%d')}.zip"
        
        with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.output_dir.parent)
                    zf.write(file_path, arcname)
        
        print(f"    Created archive: {archive_file}")

def main():
    """Main export function"""
    
    parser = argparse.ArgumentParser(
        description="Export MDK-LRP1 nanobody design results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results directory (default: results)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: exports/timestamp)'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['genbank', 'csv', 'excel', 'latex', 'markdown', 'benchling', 'all'],
        default=['all'],
        help='Export formats (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ResultsExporter(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    # Export based on selected formats
    if 'all' in args.formats:
        exporter.export_all()
    else:
        results = exporter.load_results()
        if not results:
            print("No results found to export!")
            return
        
        if 'genbank' in args.formats:
            exporter.export_genbank(results)
        if 'csv' in args.formats:
            exporter.export_csv_tables(results)
        if 'excel' in args.formats:
            exporter.export_excel_workbook(results)
        if 'latex' in args.formats:
            exporter.export_latex_tables(results)
        if 'markdown' in args.formats:
            exporter.export_markdown_report(results)
        if 'benchling' in args.formats:
            exporter.export_benchling_format(results)

if __name__ == "__main__":
    main()