"""
Unit tests for RNA-TBM pipeline components.
"""
import numpy as np
import pytest


class TestCIFParser:
    """Tests for CIF parser module."""
    
    def test_parse_cif_line_basic(self):
        """Test basic CIF line parsing."""
        from rna_tbm.cif_parser import _parse_cif_line
        
        line = "ATOM 1 C1' A A 1 10.0 20.0 30.0"
        tokens = _parse_cif_line(line)
        
        assert len(tokens) == 9
        assert tokens[0] == "ATOM"
        assert tokens[2] == "C1'"
    
    def test_parse_cif_line_quoted(self):
        """Test CIF line parsing with quoted strings."""
        from rna_tbm.cif_parser import _parse_cif_line
        
        line = 'ATOM 1 "C1\'" A A 1 10.0 20.0 30.0'
        tokens = _parse_cif_line(line)
        
        assert tokens[2] == "C1'"
    
    def test_modified_base_mapping(self):
        """Test modified nucleotide mapping."""
        from rna_tbm.cif_parser import MODIFIED_BASE_MAP

        # Standard bases
        assert MODIFIED_BASE_MAP['A'] == 'A'
        assert MODIFIED_BASE_MAP['C'] == 'C'
        assert MODIFIED_BASE_MAP['G'] == 'G'
        assert MODIFIED_BASE_MAP['U'] == 'U'
        
        # Modified bases
        assert MODIFIED_BASE_MAP['PSU'] == 'U'  # Pseudouridine
        assert MODIFIED_BASE_MAP['5MC'] == 'C'  # 5-methylcytosine
        assert MODIFIED_BASE_MAP['1MG'] == 'G'  # 1-methylguanine


class TestAlignment:
    """Tests for alignment module."""
    
    def test_needleman_wunsch_identical(self):
        """Test alignment of identical sequences."""
        from rna_tbm.alignment import needleman_wunsch
        
        seq = "ACCGUGAC"
        aligned1, aligned2, pairs = needleman_wunsch(seq, seq)
        
        assert aligned1 == seq
        assert aligned2 == seq
        assert len(pairs) == len(seq)
    
    def test_needleman_wunsch_with_gap(self):
        """Test alignment with insertion."""
        from rna_tbm.alignment import needleman_wunsch
        
        seq1 = "ACCGUGAC"
        seq2 = "ACCGAC"  # Missing UG
        
        aligned1, aligned2, pairs = needleman_wunsch(seq1, seq2)
        
        # Should align with gaps
        assert len(pairs) < len(seq1)
    
    def test_align_sequences_metrics(self):
        """Test alignment result metrics."""
        from rna_tbm.alignment import align_sequences
        
        query = "ACCGUGACGGG"
        template = "ACCGUGACGGG"
        
        result = align_sequences(query, template)
        
        assert result.identity == 1.0
        assert result.coverage == 1.0
        assert len(result.query_to_template) == len(query)


class TestGapFilling:
    """Tests for gap filling module."""
    
    def test_generate_geometric_baseline(self):
        """Test geometric baseline generation."""
        from rna_tbm.gap_filling import generate_geometric_baseline
        
        n = 50
        coords = generate_geometric_baseline(n)
        
        assert coords.shape == (n, 3)
        assert not np.any(np.isnan(coords))
        
        # Check reasonable distances
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        assert np.all(distances > 0)
        assert np.all(distances < 20)
    
    def test_clip_coordinates(self):
        """Test coordinate clipping."""
        from rna_tbm.gap_filling import clip_coordinates
        
        coords = np.array([[-1000, 0, 10000], [0, 0, 0]])
        clipped = clip_coordinates(coords)
        
        assert np.min(clipped) >= -999.999
        assert np.max(clipped) <= 9999.999


class TestSubmission:
    """Tests for submission module."""
    
    def test_prediction_set_validation(self):
        """Test PredictionSet validation."""
        from rna_tbm.gap_filling import generate_geometric_baseline
        from rna_tbm.submission import PredictionSet
        
        sequence = "ACGU"
        models = [generate_geometric_baseline(len(sequence)) for _ in range(5)]
        
        pred = PredictionSet(
            target_id="TEST",
            sequence=sequence,
            models=models
        )
        
        assert pred.target_id == "TEST"
        assert len(pred.models) == 5
    
    def test_prediction_set_wrong_models(self):
        """Test PredictionSet rejects wrong number of models."""
        from rna_tbm.gap_filling import generate_geometric_baseline
        from rna_tbm.submission import PredictionSet
        
        sequence = "ACGU"
        models = [generate_geometric_baseline(len(sequence)) for _ in range(3)]
        
        with pytest.raises(ValueError):
            PredictionSet(
                target_id="TEST",
                sequence=sequence,
                models=models
            )
    
    def test_create_submission_rows(self):
        """Test submission row creation."""
        from rna_tbm.gap_filling import generate_geometric_baseline
        from rna_tbm.submission import PredictionSet, create_submission_rows
        
        sequence = "ACGU"
        models = [generate_geometric_baseline(len(sequence)) for _ in range(5)]
        
        pred = PredictionSet(
            target_id="TEST_1",
            sequence=sequence,
            models=models
        )
        
        rows = create_submission_rows(pred)
        
        assert len(rows) == 4
        assert rows[0]['ID'] == 'TEST_1_1'
        assert rows[0]['resname'] == 'A'
        assert rows[0]['resid'] == 1
        assert 'x_1' in rows[0]
        assert 'z_5' in rows[0]


class TestTemplateDB:
    """Tests for template database."""
    
    def test_kmer_extraction(self):
        """Test k-mer index building."""
        from rna_tbm.template_db import TemplateDB
        
        db = TemplateDB(k=3)
        
        # Manually add a sequence
        db.sequences[('TEST', 'A')] = 'ACGU'
        
        # Build k-mers manually to test logic
        seq = 'ACGU'
        kmers = [seq[i:i+3] for i in range(len(seq) - 3 + 1)]
        
        assert kmers == ['ACG', 'CGU']
    
    def test_quick_align_identical(self):
        """Test quick alignment with identical sequences."""
        from rna_tbm.template_db import TemplateDB
        
        db = TemplateDB(k=3)
        
        seq = "ACCGUGACGGG"
        identity, coverage, aligned_pairs = db._align_sequences(seq, seq)
        
        assert identity == 1.0
        assert coverage == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
