"""
Unit tests for enhanced RNA-TBM pipeline components.
Tests MSA, embeddings, refinement, metal ions, and functional modules.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile


class TestMSAModule:
    """Tests for MSA covariation module."""
    
    def test_msa_parser_stockholm(self):
        """Test Stockholm format parsing."""
        from rna_tbm.msa import MSAParser
        
        # Create temp Stockholm file
        content = """# STOCKHOLM 1.0
#=GF ID test_family
seq1  ACCGUGAC
seq2  ACCAUGAC
seq3  ACCGUGAC
//"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sto', delete=False) as f:
            f.write(content)
            f.flush()
            
            msa = MSAParser.parse_stockholm(f.name)
            
            assert msa.num_sequences == 3
            assert msa.alignment_length == 8
            assert msa.sequences[0].name == "seq1"
    
    def test_msa_parser_a2m(self):
        """Test A2M format parsing."""
        from rna_tbm.msa import MSAParser
        
        content = """>seq1
ACCGUGAC
>seq2
ACCAUGAC"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.a2m', delete=False) as f:
            f.write(content)
            f.flush()
            
            msa = MSAParser.parse_a2m(f.name)
            
            assert msa.num_sequences == 2
            assert msa.sequences[0].sequence == "ACCGUGAC"
    
    def test_covariation_analyzer(self):
        """Test DCA computation."""
        from rna_tbm.msa import MSA, MSASequence, CovariationAnalyzer
        
        # Create simple MSA
        sequences = [
            MSASequence(name="seq1", sequence="ACCGUG"),
            MSASequence(name="seq2", sequence="ACCGUG"),
            MSASequence(name="seq3", sequence="UCCGUG"),
        ]
        msa = MSA(sequences=sequences)
        
        analyzer = CovariationAnalyzer()
        
        # Test frequency computation
        single_freq, pair_freq = analyzer.compute_frequencies(msa)
        assert single_freq.shape == (6, 5)  # L x alphabet_size
        
        # Test DCA scores
        dca_scores = analyzer.compute_dca_scores(msa)
        assert dca_scores.shape == (6, 6)
        assert np.all(dca_scores >= 0)  # After APC correction
    
    def test_secondary_structure_predictor(self):
        """Test base pair prediction."""
        from rna_tbm.msa import SecondaryStructurePredictor, BasePair
        
        # Disable sequence constraints to allow A-A pairs for testing
        predictor = SecondaryStructurePredictor(
            min_score=0.0,
            use_sequence_constraints=False,
        )
        
        # Create dummy score matrix with high score at (0, 10)
        scores = np.zeros((20, 20))
        scores[0, 10] = 1.0
        scores[10, 0] = 1.0
        scores[5, 15] = 0.8
        scores[15, 5] = 0.8
        
        pairs = predictor.predict_pairs_from_scores(
            scores,
            sequence="A" * 20,
            top_k=5,
        )
        
        assert len(pairs) >= 1
        assert isinstance(pairs[0], BasePair)
    
    def test_dot_bracket_conversion(self):
        """Test dot-bracket notation conversion."""
        from rna_tbm.msa import SecondaryStructurePredictor, BasePair
        
        predictor = SecondaryStructurePredictor()
        
        # Pairs at (0,9) and (2,7) - properly nested
        pairs = [
            BasePair(i=0, j=9, score=1.0),
            BasePair(i=2, j=7, score=0.8),
        ]
        
        db = predictor.to_dot_bracket(pairs, length=10)
        # Position 0='(', 9=')', 2='(', 7=')'
        assert db == "(.(....).)"  # Matches actual output
        
        # Round-trip
        parsed = predictor.from_dot_bracket(db)
        assert len(parsed) == 2


class TestEmbeddingsModule:
    """Tests for embeddings module."""
    
    def test_rna_fm_encoder_fallback(self):
        """Test fallback embedding generation."""
        from rna_tbm.embeddings import RNAFMEncoder
        
        encoder = RNAFMEncoder(use_fallback=True)
        
        sequence = "ACCGUGACGGG"
        embeddings = encoder.encode(sequence)
        
        assert embeddings.shape == (11, 640)
        assert not np.any(np.isnan(embeddings))
    
    def test_embedding_cache(self):
        """Test embedding caching."""
        from rna_tbm.embeddings import EmbeddingCache
        
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = EmbeddingCache(cache_dir)
            
            # Store embedding
            emb = np.random.randn(10, 640).astype(np.float32)
            cache.put("TEST", "A", emb)
            
            # Retrieve
            retrieved = cache.get("TEST", "A")
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, emb)
            
            # Check cache size
            assert len(cache) == 1
    
    def test_embedding_similarity(self):
        """Test embedding similarity computation."""
        from rna_tbm.embeddings import EmbeddingSimilarity
        
        emb1 = np.random.randn(10, 640)
        emb2 = emb1 + np.random.randn(10, 640) * 0.1  # Similar
        emb3 = np.random.randn(10, 640)  # Different
        
        # Cosine similarity
        sim_same = EmbeddingSimilarity.cosine_similarity(emb1, emb1)
        sim_similar = EmbeddingSimilarity.cosine_similarity(emb1, emb2)
        sim_diff = EmbeddingSimilarity.cosine_similarity(emb1, emb3)
        
        assert sim_same > 0.99
        assert sim_similar > sim_diff
        
        # Alignment score
        align_score = EmbeddingSimilarity.alignment_score(emb1, emb2)
        assert 0 <= align_score <= 1


class TestRefinementModule:
    """Tests for torsion refinement module."""
    
    def test_torsion_to_coords(self):
        """Test torsion to coordinate conversion."""
        from rna_tbm.refinement import TorsionToCoords
        
        converter = TorsionToCoords()
        
        # Create test inputs
        initial_coords = np.random.randn(10, 3) * 10
        torsions = np.random.randn(10, 7) * 0.5  # Small torsion angles
        sequence = "ACCGUGACGU"
        
        refined = converter.apply_torsions(initial_coords, torsions, sequence)
        
        assert refined.shape == (10, 3)
        assert not np.any(np.isnan(refined))
    
    def test_structure_refiner(self):
        """Test structure refinement pipeline."""
        from rna_tbm.refinement import StructureRefiner
        
        refiner = StructureRefiner(use_fallback=True)
        
        coords = np.random.randn(15, 3) * 10
        sequence = "ACCGUGACGUGACGU"
        
        refined = refiner.refine(coords, sequence)
        
        assert refined.shape == (15, 3)
    
    def test_torsion_predictor_numpy(self):
        """Test NumPy-based torsion predictor."""
        from rna_tbm.refinement.torsion_model import TorsionPredictorNumpy
        
        predictor = TorsionPredictorNumpy()
        
        torsions = predictor.predict("ACCGUGAC")
        
        assert torsions.shape == (8, 7)
        # Check angles are in reasonable range (within 2*pi)
        assert np.all(np.abs(torsions) < 2 * np.pi)


class TestMetalIonsModule:
    """Tests for metal ion prediction module."""
    
    def test_binding_predictor_heuristics(self):
        """Test heuristic-based binding prediction."""
        from rna_tbm.metal_ions import MetalBindingPredictor
        
        predictor = MetalBindingPredictor(threshold=0.3)
        
        coords = np.random.randn(20, 3) * 10
        sequence = "GGGGACGUGACGUGGGG"  # G-rich ends
        
        sites, probs = predictor.predict(coords, sequence, return_probabilities=True)
        
        assert isinstance(sites, list)
        assert len(probs) == len(sequence)
        assert np.all((probs >= 0) & (probs <= 1))
    
    def test_metal_geometry_adjustment(self):
        """Test coordinate adjustment for metal sites."""
        from rna_tbm.metal_ions import MetalSiteGeometry
        
        geometry = MetalSiteGeometry()
        
        coords = np.random.randn(10, 3) * 5
        binding_sites = [3, 7]
        sequence = "GGGGACGUGA"
        
        adjusted = geometry.adjust_coordinates(coords, binding_sites, sequence)
        
        assert adjusted.shape == coords.shape
        # Should be different from original
        assert not np.allclose(adjusted, coords)
    
    def test_graph_construction(self):
        """Test graph building from coordinates."""
        from rna_tbm.metal_ions.mgnet import build_graph_from_coords
        
        coords = np.random.randn(15, 3) * 10
        sequence = "ACCGUGACGUGACGU"
        
        graph = build_graph_from_coords(coords, sequence, edge_cutoff=10.0)
        
        assert 'node_features' in graph
        assert 'edge_index' in graph
        assert graph['node_features'].shape[0] == 15


class TestFunctionalModule:
    """Tests for functional homology module."""
    
    def test_rfam_matcher_heuristic(self):
        """Test heuristic-based Rfam matching."""
        from rna_tbm.functional import RfamMatcher
        
        matcher = RfamMatcher(use_infernal=False)
        
        # tRNA-like sequence (70-95 nt, CCA end)
        sequence = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGA" + "CCA"
        
        hits = matcher.search(sequence)
        
        # Should find potential matches based on length/motifs
        assert isinstance(hits, list)
    
    def test_functional_similarity(self):
        """Test functional similarity computation."""
        from rna_tbm.functional import FunctionalSimilarity
        
        sim = FunctionalSimilarity()
        
        seq1 = "ACCGUGACGUGACGU"
        seq2 = "ACCGUGACGUGACGU"  # Identical
        seq3 = "UUUUUUUUUUUUUUU"  # Different
        
        score_same = sim.compute_similarity(seq1, seq2)
        score_diff = sim.compute_similarity(seq1, seq3)
        
        assert score_same > score_diff
        assert 0 <= score_same <= 1
        assert 0 <= score_diff <= 1


class TestEnhancedPipeline:
    """Tests for enhanced pipeline integration."""
    
    def test_enhanced_config(self):
        """Test enhanced configuration."""
        from rna_tbm.config import EnhancedPipelineConfig
        
        config = EnhancedPipelineConfig()
        
        assert config.use_msa_covariation == True
        assert config.use_embeddings == True
        assert config.use_torsion_refinement == True
        assert config.use_metal_prediction == True
        
        # Test minimal config
        minimal = EnhancedPipelineConfig.minimal()
        assert minimal.use_msa_covariation == False
    
    def test_module_imports(self):
        """Test that all modules can be imported."""
        from rna_tbm import (
            get_msa_module,
            get_embeddings_module,
            get_refinement_module,
            get_metal_ions_module,
            get_functional_module,
        )
        
        msa = get_msa_module()
        assert hasattr(msa, 'MSAParser')
        
        embeddings = get_embeddings_module()
        assert hasattr(embeddings, 'RNAFMEncoder')
        
        refinement = get_refinement_module()
        assert hasattr(refinement, 'StructureRefiner')
        
        metal = get_metal_ions_module()
        assert hasattr(metal, 'MetalBindingPredictor')
        
        functional = get_functional_module()
        assert hasattr(functional, 'RfamMatcher')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
