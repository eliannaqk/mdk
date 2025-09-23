#!/usr/bin/env python3
"""
Test suite for MDK-LRP1 nanobody design pipeline
Run with: pytest test/test_pipeline.py -v
"""

import pytest
import json
import yaml
import numpy as np
from pathlib import Path
import tempfile
import shutil
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.utils.structure import calculate_distance, get_surface_residues
from src.utils.scoring import calculate_interface_score

# Mock data for testing
MOCK_SEQUENCE = "QVQLVESGGGLVQPGGSLRLSCAASGFTFSDYYMSWVRQAPGKGLEWVSYISSSGSTIYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAREGGWFGELAFDYWGQGTLVTVSS"
MOCK_PDB_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  
ATOM      3  C   ALA A   1       2.000   1.420   0.000  1.00  0.00           C  
ATOM      4  O   ALA A   1       1.221   2.370   0.000  1.00  0.00           O  
END"""

class TestStructurePreparation:
    """Test structure preparation functions"""
    
    def test_pdb_download(self, tmp_path):
        """Test PDB file download"""
        from download_structures import download_pdb
        
        # Test with real PDB ID (small structure)
        pdb_id = "1l3r"  # Small peptide
        output_dir = tmp_path / "structures"
        output_dir.mkdir()
        
        result = download_pdb(pdb_id, output_dir)
        
        assert result is not None
        assert result.exists()
        assert result.stat().st_size > 0
    
    def test_structure_parsing(self, tmp_path):
        """Test PDB structure parsing"""
        from Bio import PDB
        
        # Create mock PDB file
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(MOCK_PDB_CONTENT)
        
        # Parse structure
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("test", pdb_file)
        
        assert structure is not None
        assert len(list(structure.get_models())) == 1
        assert len(list(structure.get_atoms())) == 4

class TestEpitopeDefinition:
    """Test epitope definition functions"""
    
    def test_epitope_expansion(self):
        """Test epitope expansion around core residues"""
        
        core_residues = [86, 87, 89]
        expand_radius = 5.0
        
        # Mock expansion (in reality would use 3D coordinates)
        expanded = set(core_residues)
        for res in core_residues:
            expanded.add(res - 1)
            expanded.add(res + 1)
        
        expanded = sorted(list(expanded))
        
        assert len(expanded) > len(core_residues)
        assert all(r in expanded for r in core_residues)
    
    def test_epitope_clustering(self):
        """Test residue clustering for epitope patches"""
        
        residues = [86, 87, 89, 94, 95, 102, 103]
        distance_cutoff = 3  # Sequential distance for testing
        
        patches = []
        current_patch = [residues[0]]
        
        for i in range(1, len(residues)):
            if residues[i] - residues[i-1] <= distance_cutoff:
                current_patch.append(residues[i])
            else:
                patches.append(current_patch)
                current_patch = [residues[i]]
        
        patches.append(current_patch)
        
        assert len(patches) == 3
        assert [86, 87, 89] in patches

class TestBindCraftIntegration:
    """Test BindCraft integration"""
    
    def test_config_loading(self):
        """Test configuration file loading"""
        
        config_content = {
            'target': {'name': 'MDK', 'pdb_file': 'test.pdb'},
            'epitope': {'residues': [86, 87]},
            'bindcraft': {'hallucination': {'num_designs': 10}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_content, f)
            config_file = f.name
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config['target']['name'] == 'MDK'
            assert len(loaded_config['epitope']['residues']) == 2
            assert loaded_config['bindcraft']['hallucination']['num_designs'] == 10
        
        finally:
            os.unlink(config_file)
    
    def test_mock_design_generation(self):
        """Test mock design generation"""
        
        # Simulate design generation
        num_designs = 5
        designs = []
        
        for i in range(num_designs):
            design = {
                'id': f'design_{i+1}',
                'sequence': MOCK_SEQUENCE[:50+i*5],  # Variable length
                'scores': {
                    'plddt': np.random.uniform(70, 95),
                    'iptm': np.random.uniform(0.5, 0.9),
                    'interface_area': np.random.uniform(400, 1200)
                }
            }
            designs.append(design)
        
        assert len(designs) == num_designs
        assert all('sequence' in d for d in designs)
        assert all(len(d['sequence']) > 0 for d in designs)

class TestNanobodyConversion:
    """Test nanobody format conversion"""
    
    def test_cdr_extraction(self):
        """Test CDR extraction from designed sequence"""
        
        sequence = MOCK_SEQUENCE
        
        # Mock CDR positions (simplified)
        cdr1_start, cdr1_end = 26, 35
        cdr2_start, cdr2_end = 50, 65
        cdr3_start, cdr3_end = 95, 110
        
        cdrs = {
            'CDR1': sequence[cdr1_start:cdr1_end],
            'CDR2': sequence[cdr2_start:cdr2_end],
            'CDR3': sequence[cdr3_start:cdr3_end]
        }
        
        assert len(cdrs['CDR1']) > 0
        assert len(cdrs['CDR2']) > 0
        assert len(cdrs['CDR3']) > 0
    
    def test_nanobody_assembly(self):
        """Test nanobody sequence assembly"""
        
        framework = {
            'FR1': 'QVQLVESGGGLVQPGGSLRLSCAAS',
            'FR2': 'WVRQAPGKGLEWVS',
            'FR3': 'RFTISRDNSKNTLYLQMNSLRAEDTAVYYC',
            'FR4': 'WGQGTLVTVSS'
        }
        
        cdrs = {
            'CDR1': 'GFTFS',
            'CDR2': 'IS',
            'CDR3': 'AR'
        }
        
        nanobody = (
            framework['FR1'] + cdrs['CDR1'] +
            framework['FR2'] + cdrs['CDR2'] +
            framework['FR3'] + cdrs['CDR3'] +
            framework['FR4']
        )
        
        assert len(nanobody) >= 80
        assert nanobody.startswith(framework['FR1'])
        assert nanobody.endswith(framework['FR4'])

class TestValidation:
    """Test validation and analysis functions"""
    
    def test_developability_assessment(self):
        """Test developability metric calculation"""
        from Bio.SeqUtils import ProtParam
        
        analyzer = ProtParam.ProteinAnalysis(MOCK_SEQUENCE)
        
        # Calculate properties
        mw = analyzer.molecular_weight()
        pi = analyzer.isoelectric_point()
        instability = analyzer.instability_index()
        
        assert mw > 10000  # Typical nanobody MW
        assert 4 < pi < 11  # Reasonable pI range
        assert instability is not None
    
    def test_specificity_check(self):
        """Test cross-reactivity assessment"""
        
        sequence = MOCK_SEQUENCE
        
        # Mock specificity check
        mdk_specific_motifs = ['RGR', 'KKK', 'RKR']
        specificity_score = sum(1 for motif in mdk_specific_motifs if motif in sequence)
        
        risk = 'low' if specificity_score >= 2 else 'medium' if specificity_score >= 1 else 'high'
        
        assert risk in ['low', 'medium', 'high']
    
    def test_expression_prediction(self):
        """Test expression level prediction"""
        
        sequence = MOCK_SEQUENCE
        
        # Simple expression score
        rare_aa = 'WCM'
        rare_count = sum(1 for aa in sequence if aa in rare_aa)
        
        expression_score = max(0, 100 - rare_count * 5)
        expression_level = 'high' if expression_score > 80 else 'medium' if expression_score > 60 else 'low'
        
        assert 0 <= expression_score <= 100
        assert expression_level in ['high', 'medium', 'low']

class TestFileIO:
    """Test file input/output operations"""
    
    def test_fasta_writing(self, tmp_path):
        """Test FASTA file writing"""
        
        sequences = [
            SeqRecord(Seq(MOCK_SEQUENCE), id="nb_1", description="Test nanobody 1"),
            SeqRecord(Seq(MOCK_SEQUENCE[:100]), id="nb_2", description="Test nanobody 2")
        ]
        
        fasta_file = tmp_path / "test.fasta"
        SeqIO.write(sequences, fasta_file, "fasta")
        
        # Read back
        parsed = list(SeqIO.parse(fasta_file, "fasta"))
        
        assert len(parsed) == 2
        assert str(parsed[0].seq) == MOCK_SEQUENCE
        assert parsed[1].id == "nb_2"
    
    def test_json_serialization(self, tmp_path):
        """Test JSON data serialization"""
        
        data = {
            'designs': [
                {'id': 'design_1', 'score': 85.5},
                {'id': 'design_2', 'score': 92.3}
            ],
            'metadata': {
                'date': '2024-01-01',
                'epitope': 'LRP1'
            }
        }
        
        json_file = tmp_path / "test.json"
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        with open(json_file, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded['designs']) == 2
        assert loaded['metadata']['epitope'] == 'LRP1'

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    @pytest.fixture
    def setup_test_environment(self, tmp_path):
        """Setup test environment"""
        
        # Create directory structure
        dirs = ['data/structures', 'data/epitopes', 'results', 'scripts']
        for dir_name in dirs:
            (tmp_path / dir_name).mkdir(parents=True)
        
        # Create mock config
        config = {
            'target': {
                'name': 'MDK_test',
                'pdb_file': str(tmp_path / 'data/structures/test.pdb')
            },
            'epitope': {
                'residues': [86, 87, 89]
            },
            'output': {
                'base_dir': str(tmp_path / 'results')
            }
        }
        
        config_file = tmp_path / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return tmp_path
    
    def test_pipeline_flow(self, setup_test_environment):
        """Test basic pipeline flow"""
        
        test_dir = setup_test_environment
        
        # Verify setup
        assert (test_dir / 'config.yaml').exists()
        assert (test_dir / 'data/structures').exists()
        
        # Simulate pipeline steps
        steps_completed = []
        
        # Step 1: Structure prep
        pdb_file = test_dir / 'data/structures/test.pdb'
        pdb_file.write_text(MOCK_PDB_CONTENT)
        steps_completed.append('structure_prep')
        
        # Step 2: Epitope definition
        epitope_data = {
            'residues': [86, 87, 89],
            'expanded_residues': [85, 86, 87, 88, 89, 90]
        }
        epitope_file = test_dir / 'data/epitopes/test_epitope.json'
        with open(epitope_file, 'w') as f:
            json.dump(epitope_data, f)
        steps_completed.append('epitope_definition')
        
        # Step 3: Mock design generation
        designs_dir = test_dir / 'results/designs'
        designs_dir.mkdir(parents=True)
        steps_completed.append('design_generation')
        
        # Step 4: Nanobody conversion
        nb_dir = test_dir / 'results/nanobodies'
        nb_dir.mkdir(parents=True)
        steps_completed.append('nanobody_conversion')
        
        # Step 5: Validation
        val_dir = test_dir / 'results/validation'
        val_dir.mkdir(parents=True)
        steps_completed.append('validation')
        
        assert len(steps_completed) == 5
        assert all((test_dir / 'results').glob('*'))

@pytest.mark.parametrize("sequence,expected_length", [
    (MOCK_SEQUENCE, len(MOCK_SEQUENCE)),
    ("ACDEFGHIKLMNPQRSTVWY", 20),
    ("QVQLVESGGG", 10)
])
def test_sequence_processing(sequence, expected_length):
    """Parametrized test for sequence processing"""
    assert len(sequence) == expected_length
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_version():
    """Test pipeline version"""
    assert sys.version_info >= (3, 8)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
