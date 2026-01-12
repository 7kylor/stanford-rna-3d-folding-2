"""
Main TBM inference pipeline for Stanford RNA 3D Folding Part 2.
This module orchestrates the full prediction workflow.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .alignment import align_sequences, transfer_coordinates
from .config import PipelineConfig, get_config
from .gap_filling import fill_gaps, generate_geometric_baseline
from .submission import PredictionSet, write_submission
from .template_db import TemplateDB, build_template_database


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""
    num_targets: int = 0
    num_with_templates: int = 0
    num_no_templates: int = 0
    total_time_seconds: float = 0.0
    db_build_time_seconds: float = 0.0
    prediction_time_seconds: float = 0.0
    
    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"Processed {self.num_targets} targets:\n"
            f"  - With templates: {self.num_with_templates}\n"
            f"  - No templates: {self.num_no_templates}\n"
            f"  - DB build time: {self.db_build_time_seconds:.1f}s\n"
            f"  - Prediction time: {self.prediction_time_seconds:.1f}s\n"
            f"  - Total time: {self.total_time_seconds:.1f}s"
        )


class TBMPipeline:
    """
    Template-Based Modeling pipeline for RNA structure prediction.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or get_config()
        self.template_db: Optional[TemplateDB] = None
        self.stats = PipelineStats()
    
    def load_or_build_database(self, force_rebuild: bool = False) -> TemplateDB:
        """
        Load existing database or build new one.
        
        Args:
            force_rebuild: Force rebuilding even if cache exists
            
        Returns:
            Loaded or built TemplateDB
        """
        paths = self.config.paths
        db_path = paths.template_db_file
        
        # Ensure output directory exists
        paths.ensure_output_dir()
        
        if not force_rebuild and db_path and db_path.exists():
            print(f"Loading template database from {db_path}...")
            start = time.time()
            self.template_db = TemplateDB.load(str(db_path))
            self.stats.db_build_time_seconds = time.time() - start
            print(f"Loaded {len(self.template_db)} templates in {self.stats.db_build_time_seconds:.1f}s")
        else:
            print("Building template database...")
            start = time.time()
            
            pdb_dir = paths.pdb_rna_dir
            release_file = paths.release_dates_file
            
            if not pdb_dir or not pdb_dir.exists():
                raise FileNotFoundError(f"PDB_RNA directory not found: {pdb_dir}")
            
            self.template_db = build_template_database(
                pdb_rna_dir=str(pdb_dir),
                release_dates_file=str(release_file) if release_file else None,
                output_path=str(db_path) if db_path else None,
                k=self.config.kmer_size,
                max_files=self.config.max_files,
                max_file_size_mb=self.config.max_file_size_mb
            )
            self.stats.db_build_time_seconds = time.time() - start
        
        return self.template_db
    
    def predict_single(
        self,
        target_id: str,
        sequence: str,
        temporal_cutoff: str,
        num_models: int = 5
    ) -> PredictionSet:
        """
        Predict structure for a single target.
        
        Args:
            target_id: Target identifier
            sequence: RNA sequence
            temporal_cutoff: Only use templates before this date
            num_models: Number of models to generate
            
        Returns:
            PredictionSet with 5 models
        """
        if self.template_db is None:
            raise RuntimeError("Template database not loaded. Call load_or_build_database() first.")
        
        n = len(sequence)
        
        # Search for templates
        hits = self.template_db.search(
            sequence,
            temporal_cutoff=temporal_cutoff,
            min_identity=self.config.min_identity,
            min_coverage=self.config.min_coverage,
            max_hits=self.config.max_template_hits
        )
        
        models = []
        
        if hits:
            self.stats.num_with_templates += 1
            
            # Use top templates for diversity
            for model_idx in range(num_models):
                hit_idx = model_idx % len(hits)
                hit = hits[hit_idx]
                
                template = self.template_db.get_template_coords(hit.pdb_id, hit.chain_id)
                if template:
                    # Align and transfer
                    alignment = align_sequences(sequence, template.sequence)
                    transfer_result = transfer_coordinates(sequence, template, alignment)
                    
                    # Fill gaps
                    filled_result = fill_gaps(transfer_result, method=self.config.gap_fill_method)
                    coords = filled_result.coords
                    
                    # Add perturbation for diversity (except model 0)
                    if model_idx > 0:
                        noise = np.random.randn(n, 3) * self.config.perturbation_scale * model_idx
                        coords = coords + noise
                    
                    models.append(np.clip(coords, -999.999, 9999.999))
                else:
                    models.append(generate_geometric_baseline(n))
        else:
            self.stats.num_no_templates += 1
            # No templates - use geometric baseline
            base = generate_geometric_baseline(n)
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
            models=models[:num_models]
        )
    
    def predict_all(
        self,
        sequences_df: pd.DataFrame,
        output_path: str = None,
        verbose: bool = True
    ) -> List[PredictionSet]:
        """
        Predict structures for all targets in a DataFrame.
        
        Args:
            sequences_df: DataFrame with columns: target_id, sequence, temporal_cutoff
            output_path: Path to write submission CSV (optional)
            verbose: Print progress
            
        Returns:
            List of PredictionSet
        """
        start_time = time.time()
        self.stats.num_targets = len(sequences_df)
        
        predictions = []
        
        for idx, row in sequences_df.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            temporal_cutoff = row['temporal_cutoff']
            
            pred_start = time.time()
            pred = self.predict_single(
                target_id,
                sequence,
                temporal_cutoff,
                num_models=self.config.num_models
            )
            pred_time = time.time() - pred_start
            
            predictions.append(pred)
            
            if verbose:
                print(f"[{idx+1}/{len(sequences_df)}] {target_id}: {len(sequence)}nt, {pred_time:.2f}s")
        
        self.stats.prediction_time_seconds = time.time() - start_time - self.stats.db_build_time_seconds
        self.stats.total_time_seconds = time.time() - start_time
        
        # Write submission if path provided
        if output_path:
            write_submission(predictions, output_path)
        
        return predictions
    
    def run(
        self,
        sequences_file: str = None,
        output_path: str = None,
        force_rebuild_db: bool = False
    ) -> List[PredictionSet]:
        """
        Run the complete pipeline.
        
        Args:
            sequences_file: Path to sequences CSV (uses config default if None)
            output_path: Path for output CSV (uses config default if None)
            force_rebuild_db: Force database rebuild
            
        Returns:
            List of PredictionSet
        """
        print("=" * 60)
        print("Stanford RNA 3D Folding - TBM Pipeline")
        print("=" * 60)
        
        # Load or build database
        print("\n1. Preparing template database...")
        self.load_or_build_database(force_rebuild=force_rebuild_db)
        
        # Load sequences
        if sequences_file is None:
            sequences_file = self.config.paths.test_sequences_file
        
        print(f"\n2. Loading sequences from {sequences_file}...")
        sequences_df = pd.read_csv(sequences_file)
        print(f"   Found {len(sequences_df)} targets")
        
        # Run predictions
        print("\n3. Running predictions...")
        if output_path is None:
            output_path = str(self.config.paths.output_dir / "submission.csv")
        
        predictions = self.predict_all(sequences_df, output_path, verbose=True)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(self.stats.summary())
        
        return predictions


def run_pipeline(
    sequences_file: str = None,
    output_path: str = None,
    config: PipelineConfig = None
) -> List[PredictionSet]:
    """
    Convenience function to run the pipeline.
    
    Args:
        sequences_file: Path to sequences CSV
        output_path: Path for output CSV
        config: Pipeline configuration
        
    Returns:
        List of PredictionSet
    """
    pipeline = TBMPipeline(config)
    return pipeline.run(sequences_file, output_path)
