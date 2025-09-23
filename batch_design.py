#!/usr/bin/env python3
"""
Batch processing script for running multiple nanobody design campaigns
targeting different MDK epitopes in parallel
"""

import os
import json
import yaml
import shutil
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import subprocess
import time
import pandas as pd

class MDKEpitopeCampaign:
    """Manages a single epitope design campaign"""
    
    def __init__(self, epitope_name: str, epitope_config: Dict, base_dir: Path):
        self.epitope_name = epitope_name
        self.epitope_config = epitope_config
        self.base_dir = base_dir
        self.campaign_dir = base_dir / epitope_name
        self.start_time = None
        self.end_time = None
        
    def setup_campaign_directory(self):
        """Create directory structure for this campaign"""
        
        # Create campaign directory
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        dirs = [
            'config', 'data/structures', 'data/epitopes', 
            'results/designs', 'results/nanobodies', 
            'results/validation', 'logs'
        ]
        
        for dir_name in dirs:
            (self.campaign_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Copy base structures
        src_structures = Path("data/structures")
        if src_structures.exists():
            for pdb_file in src_structures.glob("*.pdb"):
                shutil.copy(pdb_file, self.campaign_dir / "data/structures")
        
        print(f"  Setup directory for {self.epitope_name}")
    
    def create_epitope_config(self):
        """Create epitope-specific configuration"""
        
        # Load base config
        with open("config.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Update with epitope-specific settings
        base_config['epitope']['name'] = self.epitope_name
        base_config['epitope']['residues'] = self.epitope_config['residues']
        base_config['epitope']['description'] = self.epitope_config.get('description', '')
        
        # Adjust output directory
        base_config['output']['base_dir'] = str(self.campaign_dir / "results")
        
        # Save campaign config
        config_file = self.campaign_dir / "config" / "campaign_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f)
        
        return config_file
    
    def run_pipeline_step(self, script_name: str, timeout: int = 3600):
        """Run a single pipeline step"""
        
        script_path = Path("scripts") / script_name
        log_file = self.campaign_dir / "logs" / f"{script_name}.log"
        
        cmd = [
            "python", str(script_path)
        ]
        
        # Change to campaign directory for execution
        original_dir = os.getcwd()
        os.chdir(self.campaign_dir)
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Wait for completion with timeout
                process.wait(timeout=timeout)
                
                if process.returncode != 0:
                    print(f"    Error in {script_name} for {self.epitope_name}")
                    return False
                
                print(f"    Completed {script_name} for {self.epitope_name}")
                return True
                
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"    Timeout in {script_name} for {self.epitope_name}")
            return False
            
        finally:
            os.chdir(original_dir)
    
    def run_campaign(self):
        """Run complete design campaign"""
        
        print(f"\nStarting campaign: {self.epitope_name}")
        self.start_time = datetime.now()
        
        # Setup
        self.setup_campaign_directory()
        config_file = self.create_epitope_config()
        
        # Run pipeline steps
        steps = [
            ("03_define_epitope.py", 300),
            ("04_run_bindcraft.py", 14400),  # 4 hours
            ("05_convert_to_nanobody.py", 600),
            ("06_analyze_results.py", 600)
        ]
        
        success = True
        for script, timeout in steps:
            if not self.run_pipeline_step(script, timeout):
                success = False
                break
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() / 60
        
        status = "SUCCESS" if success else "FAILED"
        print(f"  Campaign {self.epitope_name}: {status} ({duration:.1f} min)")
        
        return {
            'epitope': self.epitope_name,
            'status': status,
            'duration_min': duration,
            'directory': str(self.campaign_dir)
        }

def load_epitope_definitions():
    """Load multiple epitope definitions"""
    
    epitopes = {
        'lrp1_patch': {
            'residues': [86, 87, 89, 94, 95, 96, 102, 103, 107],
            'description': 'LRP1 receptor binding patch'
        },
        'ptprz1_patch': {
            'residues': [78, 79, 81, 86, 87],  # Arg78-centered
            'description': 'PTPRZ1 high-affinity site'
        },
        'hspg_cluster1': {
            'residues': [81, 86, 87, 89],
            'description': 'Heparin-binding cluster I'
        },
        'hspg_cluster2': {
            'residues': [101, 102, 103, 107],
            'description': 'Heparin-binding cluster II'
        },
        'notch2_patch': {
            'residues': [94, 95, 96, 97, 98, 99, 100],
            'description': 'NOTCH2 binding region'
        }
    }
    
    return epitopes

def run_parallel_campaigns(epitopes: Dict, max_parallel: int = 2):
    """Run multiple campaigns in parallel"""
    
    base_dir = Path("batch_campaigns") / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create campaign objects
    campaigns = []
    for name, config in epitopes.items():
        campaign = MDKEpitopeCampaign(name, config, base_dir)
        campaigns.append(campaign)
    
    # Run campaigns with multiprocessing
    with mp.Pool(processes=max_parallel) as pool:
        results = pool.map(lambda c: c.run_campaign(), campaigns)
    
    return results, base_dir

def compare_campaign_results(base_dir: Path):
    """Compare results across all campaigns"""
    
    comparison_data = []
    
    for campaign_dir in base_dir.iterdir():
        if not campaign_dir.is_dir():
            continue
        
        # Load top nanobody from each campaign
        nb_file = campaign_dir / "results/validation/ranked_nanobodies.json"
        
        if nb_file.exists():
            with open(nb_file, 'r') as f:
                nanobodies = json.load(f)
                
                if nanobodies:
                    top_nb = nanobodies[0]
                    comparison_data.append({
                        'epitope': campaign_dir.name,
                        'top_score': top_nb['final_score'],
                        'risk_level': top_nb['developability_risk'],
                        'expression': top_nb['expression_level'],
                        'sequence_length': len(top_nb['sequence'])
                    })
    
    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)
    
    if not df.empty:
        df = df.sort_values('top_score', ascending=False)
        
        # Save comparison
        comparison_file = base_dir / "campaign_comparison.csv"
        df.to_csv(comparison_file, index=False)
        
        print("\nCampaign Comparison:")
        print(df.to_string(index=False))
        print(f"\nComparison saved to: {comparison_file}")
    
    return df

def generate_summary_report(results: List[Dict], base_dir: Path):
    """Generate summary report for all campaigns"""
    
    report_file = base_dir / "batch_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MDK Nanobody Design - Batch Campaign Summary\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Batch directory: {base_dir}\n")
        f.write(f"Total campaigns: {len(results)}\n")
        
        successful = [r for r in results if r['status'] == 'SUCCESS']
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(results) - len(successful)}\n\n")
        
        f.write("Individual Campaign Results:\n")
        f.write("-" * 40 + "\n")
        
        for result in results:
            f.write(f"\nEpitope: {result['epitope']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Duration: {result['duration_min']:.1f} minutes\n")
            f.write(f"  Directory: {result['directory']}\n")
        
        total_time = sum(r['duration_min'] for r in results)
        f.write(f"\nTotal computation time: {total_time:.1f} minutes\n")
        
    print(f"\nBatch summary saved to: {report_file}")

def main():
    """Main batch processing pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Run batch nanobody design campaigns for multiple MDK epitopes"
    )
    parser.add_argument(
        '--epitopes', 
        nargs='+', 
        help='Epitopes to target (default: all)',
        default=None
    )
    parser.add_argument(
        '--parallel', 
        type=int, 
        default=2,
        help='Number of parallel campaigns (default: 2)'
    )
    parser.add_argument(
        '--designs-per-epitope',
        type=int,
        default=50,
        help='Number of designs per epitope (default: 50)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MDK Nanobody Design - Batch Campaign Manager")
    print("=" * 60)
    
    # Load epitope definitions
    all_epitopes = load_epitope_definitions()
    
    # Select epitopes to run
    if args.epitopes:
        epitopes = {k: v for k, v in all_epitopes.items() if k in args.epitopes}
    else:
        epitopes = all_epitopes
    
    print(f"\nEpitopes to target: {list(epitopes.keys())}")
    print(f"Parallel campaigns: {args.parallel}")
    print(f"Designs per epitope: {args.designs_per_epitope}")
    
    # Update config for number of designs
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    config['bindcraft']['hallucination']['num_designs'] = args.designs_per_epitope
    
    with open("config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Run campaigns
    print("\nStarting batch campaigns...")
    start_time = datetime.now()
    
    results, base_dir = run_parallel_campaigns(epitopes, args.parallel)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60
    
    print(f"\nBatch processing complete in {total_duration:.1f} minutes")
    
    # Compare results
    print("\nComparing campaign results...")
    comparison_df = compare_campaign_results(base_dir)
    
    # Generate summary
    generate_summary_report(results, base_dir)
    
    # Final recommendations
    if not comparison_df.empty:
        best_epitope = comparison_df.iloc[0]['epitope']
        best_score = comparison_df.iloc[0]['top_score']
        
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print(f"\nBest performing epitope: {best_epitope}")
        print(f"Top score: {best_score}")
        print(f"\nResults directory: {base_dir}")
        print("\nNext steps:")
        print("1. Review top candidates from each epitope")
        print("2. Select 3-5 designs for experimental validation")
        print("3. Consider combining successful epitopes for biparatopic designs")

if __name__ == "__main__":
    main()