"""
Hybrid TBM + Deep Learning Pipeline.
Combines template-based modeling with learned distance predictions.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from .pipeline import TBMPipeline, PipelineStats
from .config import EnhancedPipelineConfig, PipelineConfig
from .submission import PredictionSet
from .gap_filling import generate_geometric_baseline
from .models.distance_predictor import (
    DistancePredictorNumpy,
    build_pairwise_features,
)
from .models.distance_to_coords import (
    DistanceToCoords,
    HybridCoordinateGenerator,
)


class HybridPipeline:
    """
    Hybrid pipeline combining template-based modeling with deep learning.
    
    Workflow:
    1. Search templates (like TBM)
    2. Predict inter-residue distances from sequence/MSA
    3. Combine template coords with distance constraints
    4. Refine with gradient-based optimization
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        template_weight: float = 0.6,
        distance_weight: float = 0.4,
        use_covariation: bool = True,
        use_embeddings: bool = True,
    ):
        """
        Args:
            config: Pipeline configuration
            template_weight: Weight for template-derived constraints
            distance_weight: Weight for predicted distances
            use_covariation: Use MSA covariation features
            use_embeddings: Use sequence embeddings
        """
        self.config = config or PipelineConfig()
        self.template_weight = template_weight
        self.distance_weight = distance_weight
        self.use_covariation = use_covariation
        self.use_embeddings = use_embeddings
        
        # Initialize components
        self.template_db = None
        self.distance_predictor = DistancePredictorNumpy()
        self.coord_generator = HybridCoordinateGenerator(
            template_weight=template_weight,
            distance_weight=distance_weight,
        )
        
        # Embeddings encoder
        self.embedder = None
        if use_embeddings:
            from .embeddings.rna_fm import RNAFMEncoder
            self.embedder = RNAFMEncoder(use_fallback=True)
        
        # Covariation analyzer
        self.covariation_analyzer = None
        if use_covariation:
            try:
                from .msa.covariation import CovariationAnalyzer
                self.covariation_analyzer = CovariationAnalyzer()
            except ImportError:
                pass
        
        # Statistics
        self.stats = PipelineStats()
    
    def load_or_build_database(self):
        """Load or build template database."""
        from .template_db import TemplateDB
        
        db_path = self.config.paths.template_db_file
        pdb_dir = self.config.paths.pdb_rna_dir
        release_file = self.config.paths.release_dates_file
        
        if db_path and db_path.exists():
            print(f"Loading template database from {db_path}...")
            start = time.time()
            self.template_db = TemplateDB.load(db_path)
            elapsed = time.time() - start
            print(f"Loaded {len(self.template_db.templates)} templates in {elapsed:.1f}s")
        else:
            print("Building template database...")
            self.template_db = TemplateDB()
            start = time.time()
            self.template_db.build_from_directory(
                pdb_dir,
                release_dates_file=release_file,
                max_files=self.config.max_files,
                max_file_size_mb=self.config.max_file_size_mb,
            )
            self.stats.database_build_time = time.time() - start
            
            if db_path:
                self.template_db.save(db_path)
    
    def predict_single(
        self,
        target_id: str,
        sequence: str,
        temporal_cutoff: str,
        msa_path: Optional[str] = None,
        num_models: int = 5,
    ) -> PredictionSet:
        """
        Predict structure for a single target using hybrid approach.
        
        Args:
            target_id: Target identifier
            sequence: RNA sequence
            temporal_cutoff: Templates before this date only
            msa_path: Optional path to MSA file
            num_models: Number of models to generate
            
        Returns:
            PredictionSet with predicted models
        """
        if self.template_db is None:
            self.load_or_build_database()
        
        n = len(sequence)
        
        # Step 1: Get embeddings (use enhanced mode if available)
        embeddings = None
        if self.embedder is not None:
            # Use enhanced embeddings for better features
            if hasattr(self.embedder, 'encode_enhanced'):
                embeddings = self.embedder.encode_enhanced(sequence)
            else:
                embeddings = self.embedder.encode(sequence)
        
        # Step 2: Get covariation scores (if MSA available)
        covariation_scores = None
        if msa_path and self.covariation_analyzer is not None:
            try:
                from .msa.parser import parse_msa
                msa = parse_msa(msa_path)
                if msa is not None and len(msa.sequences) > 1:
                    covariation_scores = self.covariation_analyzer.compute_dca_scores(msa)
            except Exception:
                pass
        
        # Step 3: Predict distances (pass embeddings for enhanced prediction)
        predicted_distances, distance_confidences = self.distance_predictor.predict_from_sequence(
            sequence,
            covariation_scores=covariation_scores,
            embeddings=embeddings,  # Pass embeddings for enhanced distance prediction
        )
        
        # Step 4: Search for templates
        template_coords = None
        template_mask = None
        
        hits = self.template_db.search(
            sequence,
            temporal_cutoff=temporal_cutoff,
            min_identity=self.config.min_identity,
            min_coverage=self.config.min_coverage,
            max_hits=self.config.max_template_hits,
        )
        
        if hits:
            self.stats.num_with_templates += 1
            
            # Use best template
            best_hit = hits[0]
            template = self.template_db.get_template_coords(
                best_hit.pdb_id, best_hit.chain_id
            )
            
            if template:
                from .alignment import align_sequences, transfer_coordinates
                
                alignment = align_sequences(sequence, template.sequence)
                transfer_result = transfer_coordinates(sequence, template, alignment)
                
                template_coords = transfer_result.coords
                template_mask = ~np.isnan(template_coords[:, 0])
        else:
            self.stats.num_no_templates += 1
        
        # Step 5: Generate coordinates using hybrid approach
        models = []
        
        for model_idx in range(num_models):
            # Vary weights slightly for diversity
            weight_noise = np.random.uniform(-0.1, 0.1)
            
            self.coord_generator.template_weight = max(0.1, self.template_weight + weight_noise)
            self.coord_generator.distance_weight = max(0.1, self.distance_weight - weight_noise)
            
            coords = self.coord_generator.generate(
                sequence,
                template_coords=template_coords,
                template_mask=template_mask,
                predicted_distances=predicted_distances,
                distance_confidences=distance_confidences,
            )
            
            # Add small perturbation for model diversity
            if model_idx > 0:
                noise = np.random.randn(n, 3) * 0.2 * model_idx
                coords = coords + noise
            
            models.append(np.clip(coords, -999.999, 9999.999))
        
        return PredictionSet(
            target_id=target_id,
            sequence=sequence,
            models=models,
        )
    
    def predict_all(
        self,
        sequences_df,
        output_path: Optional[str] = None,
        verbose: bool = True,
    ) -> List[PredictionSet]:
        """
        Predict structures for all sequences in dataframe.
        
        Args:
            sequences_df: DataFrame with target_id, sequence, temporal_cutoff
            output_path: Optional path to save submission CSV
            verbose: Print progress
            
        Returns:
            List of PredictionSets
        """
        from .submission import write_submission
        
        predictions = []
        total = len(sequences_df)
        
        for idx, row in sequences_df.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            temporal_cutoff = row.get('temporal_cutoff', '2099-12-31')
            
            if verbose:
                print(f"[{idx + 1}/{total}] {target_id}: {len(sequence)}nt")
            
            pred = self.predict_single(
                target_id,
                sequence,
                temporal_cutoff,
                num_models=5,
            )
            predictions.append(pred)
        
        if output_path:
            write_submission(predictions, output_path)
        
        return predictions


def run_hybrid_pipeline(
    sequences_file: str = None,
    output_path: str = None,
    max_templates: int = 2000,
    **kwargs,
) -> List[PredictionSet]:
    """
    Convenience function to run hybrid pipeline.
    
    Args:
        sequences_file: Path to sequences CSV
        output_path: Output submission file path
        max_templates: Maximum templates to load
        **kwargs: Additional configuration
        
    Returns:
        List of PredictionSet
    """
    import pandas as pd
    from .config import PathConfig, PipelineConfig
    
    # Set up paths
    paths = PathConfig()
    
    if sequences_file is None:
        sequences_file = paths.validation_sequences_file or paths.test_sequences_file
    
    if sequences_file is None:
        raise ValueError("No sequences file found")
    
    if output_path is None:
        output_path = str(paths.output_dir / "hybrid_submission.csv")
    
    # Create config
    config = PipelineConfig(
        paths=paths,
        max_files=max_templates,
    )
    
    # Create and run pipeline
    pipeline = HybridPipeline(config, **kwargs)
    pipeline.load_or_build_database()
    
    # Load sequences
    df = pd.read_csv(sequences_file)
    
    print(f"\nProcessing {len(df)} sequences...")
    predictions = pipeline.predict_all(df, output_path, verbose=True)
    
    print(f"\nResults saved to {output_path}")
    print(pipeline.stats.summary())
    
    return predictions
