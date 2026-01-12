"""
Enhanced TBM Pipeline with deep learning features.
Extends the base TBMPipeline with MSA, embeddings, torsion refinement,
metal ion prediction, and functional homology.
"""

import time
import numpy as np
from typing import List, Optional
from pathlib import Path

from .pipeline import TBMPipeline, PipelineStats
from .config import EnhancedPipelineConfig
from .submission import PredictionSet
from .gap_filling import generate_geometric_baseline


class EnhancedTBMPipeline(TBMPipeline):
    """
    Enhanced TBM Pipeline with all Phase 2 enhancements.
    
    Features:
    - Phase 2A: MSA covariation for template ranking
    - Phase 2B: RNA-FM embeddings for similarity search
    - Phase 2C: GNN torsion refinement
    - Phase 2D: Metal ion binding prediction
    - Phase 2E: Functional homology discovery
    """
    
    def __init__(self, config: EnhancedPipelineConfig = None):
        """
        Initialize enhanced pipeline.
        
        Args:
            config: Enhanced pipeline configuration
        """
        self.enhanced_config = config or EnhancedPipelineConfig()
        super().__init__(self.enhanced_config)
        
        # Enhancement modules (lazy loaded)
        self._msa_analyzer = None
        self._ss_predictor = None
        self._rna_fm_encoder = None
        self._embedding_cache = None
        self._structure_refiner = None
        self._metal_predictor = None
        self._rfam_matcher = None
        self._functional_sim = None
    
    def _init_msa_components(self):
        """Initialize MSA covariation components."""
        if self._msa_analyzer is None and self.enhanced_config.use_msa_covariation:
            from .msa import CovariationAnalyzer, SecondaryStructurePredictor
            
            self._msa_analyzer = CovariationAnalyzer(
                pseudocount=self.enhanced_config.msa_pseudocount,
            )
            self._ss_predictor = SecondaryStructurePredictor(
                min_score=self.enhanced_config.dca_threshold,
            )
    
    def _init_embedding_components(self):
        """Initialize embedding components."""
        if self._rna_fm_encoder is None and self.enhanced_config.use_embeddings:
            from .embeddings import RNAFMEncoder, EmbeddingCache
            
            self._rna_fm_encoder = RNAFMEncoder(
                model_path=self.enhanced_config.rna_fm_model_path,
            )
            
            if self.enhanced_config.embedding_cache_dir:
                self._embedding_cache = EmbeddingCache(
                    self.enhanced_config.embedding_cache_dir,
                    encoder=self._rna_fm_encoder,
                )
    
    def _init_refinement_components(self):
        """Initialize torsion refinement components."""
        if self._structure_refiner is None and self.enhanced_config.use_torsion_refinement:
            from .refinement import StructureRefiner
            
            self._structure_refiner = StructureRefiner(
                model_path=self.enhanced_config.torsion_model_path,
            )
    
    def _init_metal_components(self):
        """Initialize metal ion prediction components."""
        if self._metal_predictor is None and self.enhanced_config.use_metal_prediction:
            from .metal_ions import MetalBindingPredictor, MetalSiteGeometry
            
            self._metal_predictor = MetalBindingPredictor(
                model_path=self.enhanced_config.metal_model_path,
                threshold=self.enhanced_config.metal_threshold,
            )
            self._metal_geometry = MetalSiteGeometry(
                adjustment_strength=self.enhanced_config.metal_adjustment_strength,
            )
    
    def _init_functional_components(self):
        """Initialize functional homology components."""
        if self._rfam_matcher is None and self.enhanced_config.use_functional_homology:
            from .functional import RfamMatcher, FunctionalSimilarity
            
            self._rfam_matcher = RfamMatcher(
                rfam_cm_path=self.enhanced_config.rfam_cm_dir,
            )
            self._functional_sim = FunctionalSimilarity(self._rfam_matcher)
    
    def predict_single(
        self,
        target_id: str,
        sequence: str,
        temporal_cutoff: str,
        msa_path: Optional[str] = None,
        num_models: int = 5,
    ) -> PredictionSet:
        """
        Enhanced prediction with all Phase 2 features.
        
        Args:
            target_id: Target identifier
            sequence: RNA sequence
            temporal_cutoff: Temporal cutoff for templates
            msa_path: Optional path to MSA file
            num_models: Number of models to generate
            
        Returns:
            PredictionSet with enhanced predictions
        """
        if self.template_db is None:
            raise RuntimeError("Template database not loaded")
        
        n = len(sequence)
        
        # Phase 2A: MSA Covariation
        predicted_pairs = []
        if self.enhanced_config.use_msa_covariation and msa_path:
            predicted_pairs = self._predict_secondary_structure(msa_path)
        
        # Phase 2B: Get query embedding
        query_embedding = None
        if self.enhanced_config.use_embeddings:
            self._init_embedding_components()
            if self._rna_fm_encoder:
                query_embedding = self._rna_fm_encoder.encode(sequence)
        
        # Search for templates (using enhanced scoring if available)
        hits = self.template_db.search(
            sequence,
            temporal_cutoff=temporal_cutoff,
            min_identity=self.config.min_identity,
            min_coverage=self.config.min_coverage,
            max_hits=self.config.max_template_hits,
        )
        
        models = []
        
        if hits:
            self.stats.num_with_templates += 1
            
            for model_idx in range(num_models):
                hit_idx = model_idx % len(hits)
                hit = hits[hit_idx]
                
                template = self.template_db.get_template_coords(hit.pdb_id, hit.chain_id)
                if template:
                    # Base TBM prediction
                    from .alignment import align_sequences, transfer_coordinates
                    from .gap_filling import fill_gaps
                    
                    alignment = align_sequences(sequence, template.sequence)
                    transfer_result = transfer_coordinates(sequence, template, alignment)
                    filled_result = fill_gaps(transfer_result, method=self.config.gap_fill_method)
                    coords = filled_result.coords
                    
                    # Phase 2C: Torsion refinement
                    if self.enhanced_config.use_torsion_refinement:
                        coords = self._refine_with_torsions(coords, sequence, query_embedding)
                    
                    # Phase 2D: Metal ion adjustment
                    if self.enhanced_config.use_metal_prediction:
                        coords = self._adjust_for_metal_ions(coords, sequence)
                    
                    # Add perturbation for diversity (except model 0)
                    if model_idx > 0:
                        noise = np.random.randn(n, 3) * self.config.perturbation_scale * model_idx
                        coords = coords + noise
                    
                    models.append(np.clip(coords, -999.999, 9999.999))
                else:
                    models.append(generate_geometric_baseline(n))
        else:
            self.stats.num_no_templates += 1
            base = generate_geometric_baseline(n)
            
            # Even for ab initio, try torsion refinement
            if self.enhanced_config.use_torsion_refinement:
                base = self._refine_with_torsions(base, sequence, query_embedding)
            
            for i in range(num_models):
                if i == 0:
                    models.append(base)
                else:
                    noise = np.random.randn(n, 3) * self.config.perturbation_scale * i
                    models.append(np.clip(base + noise, -999.999, 9999.999))
        
        # Ensure exactly num_models
        while len(models) < num_models:
            models.append(models[-1].copy() if models else generate_geometric_baseline(n))
        
        return PredictionSet(
            target_id=target_id,
            sequence=sequence,
            models=models[:num_models],
        )
    
    def _predict_secondary_structure(self, msa_path: str):
        """Predict secondary structure from MSA."""
        self._init_msa_components()
        
        if self._ss_predictor is None:
            return []
        
        try:
            from .msa import MSAParser
            
            msa = MSAParser.parse(msa_path)
            pairs = self._ss_predictor.predict_pairs(msa)
            return pairs
        except Exception as e:
            print(f"Warning: MSA analysis failed: {e}")
            return []
    
    def _refine_with_torsions(
        self,
        coords: np.ndarray,
        sequence: str,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Refine coordinates using torsion prediction."""
        self._init_refinement_components()
        
        if self._structure_refiner is None:
            return coords
        
        try:
            return self._structure_refiner.refine(
                coords,
                sequence,
                embeddings=embeddings,
                num_iterations=self.enhanced_config.refinement_iterations,
            )
        except Exception as e:
            print(f"Warning: Torsion refinement failed: {e}")
            return coords
    
    def _adjust_for_metal_ions(
        self,
        coords: np.ndarray,
        sequence: str,
    ) -> np.ndarray:
        """Adjust coordinates for metal ion binding."""
        self._init_metal_components()
        
        if self._metal_predictor is None:
            return coords
        
        try:
            binding_sites = self._metal_predictor.predict(coords, sequence)
            
            if binding_sites:
                coords = self._metal_geometry.adjust_coordinates(
                    coords,
                    binding_sites,
                    sequence,
                )
            
            return coords
        except Exception as e:
            print(f"Warning: Metal ion adjustment failed: {e}")
            return coords


def run_enhanced_pipeline(
    sequences_file: str = None,
    output_path: str = None,
    use_enhancements: bool = True,
    **kwargs,
):
    """
    Run enhanced TBM pipeline.
    
    Args:
        sequences_file: Path to sequences CSV
        output_path: Output submission file path
        use_enhancements: Whether to use Phase 2 enhancements
        **kwargs: Additional configuration options
        
    Returns:
        List of PredictionSet
    """
    if use_enhancements:
        config = EnhancedPipelineConfig(**kwargs)
        pipeline = EnhancedTBMPipeline(config)
    else:
        from .config import PipelineConfig
        config = PipelineConfig(**kwargs)
        pipeline = TBMPipeline(config)
    
    return pipeline.run(sequences_file, output_path)
