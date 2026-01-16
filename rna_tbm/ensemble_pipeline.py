"""
Ensemble Pipeline for High-Accuracy RNA 3D Structure Prediction.

Combines multiple prediction sources:
1. Template-Based Modeling (TBM)
2. ML Distance Prediction
3. MSA Covariation Constraints
4. Iterative Refinement

Uses confidence-based ensemble selection for final model output.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from .config import PipelineConfig
from .template_db import TemplateDB
from .alignment import needleman_wunsch
from .gap_filling import generate_geometric_baseline
from .submission import PredictionSet, write_submission
from .confidence import ConfidenceScorer, compute_tm_score
from .models.distance_geometry import DistanceGeometrySolver, HybridCoordinateGenerator
from .models.transformer_distance import predict_distances as ml_predict_distances

# Constants
AVERAGE_C1_DISTANCE = 5.9  # Angstroms


def fill_gaps_array(coords: np.ndarray, avg_distance: float = AVERAGE_C1_DISTANCE) -> np.ndarray:
    """
    Fill gaps in coordinate array using interpolation.
    
    Simple array-based gap filling for use with raw numpy arrays.
    
    Args:
        coords: (L, 3) coordinate array, may contain NaN
        avg_distance: Average C1'-C1' distance for extrapolation
        
    Returns:
        Filled coordinate array
    """
    n = len(coords)
    if n == 0:
        return coords
    
    filled = coords.copy()
    
    # Find mapped (non-NaN) positions
    mapped_indices = [i for i in range(n) if not np.isnan(coords[i, 0])]
    
    if len(mapped_indices) == 0:
        # No coverage - generate helix
        return generate_geometric_baseline(n)
    
    if len(mapped_indices) == 1:
        # Single anchor - extend linearly
        anchor_idx = mapped_indices[0]
        direction = np.array([avg_distance, 0, 0])
        
        for i in range(anchor_idx - 1, -1, -1):
            filled[i] = filled[i + 1] - direction
        for i in range(anchor_idx + 1, n):
            filled[i] = filled[i - 1] + direction
        return filled
    
    # Multiple anchors - interpolate
    sorted_anchors = sorted(mapped_indices)
    
    # Fill internal gaps
    for i in range(len(sorted_anchors) - 1):
        start_idx = sorted_anchors[i]
        end_idx = sorted_anchors[i + 1]
        if end_idx - start_idx > 1:
            start_coord = coords[start_idx]
            end_coord = coords[end_idx]
            gap_length = end_idx - start_idx
            for j in range(1, gap_length):
                t = j / gap_length
                filled[start_idx + j] = start_coord + t * (end_coord - start_coord)
    
    # Fill leading gap
    if sorted_anchors[0] > 0:
        first_anchor = sorted_anchors[0]
        if len(sorted_anchors) >= 2:
            direction = coords[first_anchor] - coords[sorted_anchors[1]]
            norm = np.linalg.norm(direction)
            direction = direction / norm * avg_distance if norm > 0 else np.array([-avg_distance, 0, 0])
        else:
            direction = np.array([-avg_distance, 0, 0])
        for i in range(first_anchor - 1, -1, -1):
            filled[i] = filled[i + 1] + direction
    
    # Fill trailing gap
    if sorted_anchors[-1] < n - 1:
        last_anchor = sorted_anchors[-1]
        if len(sorted_anchors) >= 2:
            direction = coords[last_anchor] - coords[sorted_anchors[-2]]
            norm = np.linalg.norm(direction)
            direction = direction / norm * avg_distance if norm > 0 else np.array([avg_distance, 0, 0])
        else:
            direction = np.array([avg_distance, 0, 0])
        for i in range(last_anchor + 1, n):
            filled[i] = filled[i - 1] + direction
    
    return filled


@dataclass
class EnsembleConfig:
    """Configuration for ensemble pipeline."""
    # Template parameters
    max_templates: int = 10
    min_template_identity: float = 0.25
    
    # Distance prediction
    use_ml_distances: bool = True
    
    # Refinement
    num_refinement_steps: int = 200
    refinement_lr: float = 0.1
    
    # Ensemble
    num_candidates: int = 20
    num_output_models: int = 5
    diversity_threshold: float = 2.0  # Min RMSD between selected
    
    # Weights for combining constraints
    template_weight: float = 0.6
    ml_distance_weight: float = 0.3
    covariation_weight: float = 0.1


@dataclass  
class EnsemblePredictionSet:
    """
    Extended prediction set with confidence and TM-score tracking.
    
    Used internally by the ensemble pipeline for scoring and selection.
    Can be converted to standard PredictionSet for submission.
    """
    target_id: str
    sequence: str
    models: List[np.ndarray]
    confidences: List[float] = None
    tm_scores: List[float] = None
    
    def __post_init__(self):
        if self.confidences is None:
            self.confidences = [0.5] * len(self.models)
        if self.tm_scores is None:
            self.tm_scores = []
    
    def to_prediction_set(self) -> PredictionSet:
        """Convert to standard PredictionSet for submission."""
        # Ensure exactly 5 models
        models = self.models[:5]
        while len(models) < 5:
            if models:
                models.append(models[-1].copy())
            else:
                models.append(generate_geometric_baseline(len(self.sequence)))
        return PredictionSet(
            target_id=self.target_id,
            sequence=self.sequence,
            models=models,
        )


class EnsemblePipeline:
    """
    Multi-source ensemble pipeline for 80%+ TM-score RNA structure prediction.
    
    Architecture:
    1. Generate diverse candidates from multiple sources:
       - Template-based models (different templates)
       - Distance geometry solutions (different random seeds)
       - Hybrid template + distance models
    
    2. Score each candidate by structural quality:
       - Backbone geometry validity
       - Clash score
       - Distance constraint satisfaction
       
    3. Iteratively refine top candidates
    
    4. Select best 5 diverse models by confidence
    """
    
    def __init__(
        self,
        config: PipelineConfig = None,
        ensemble_config: EnsembleConfig = None,
    ):
        """
        Initialize ensemble pipeline.
        
        Args:
            config: Base pipeline configuration
            ensemble_config: Ensemble-specific configuration
        """
        self.config = config or PipelineConfig()
        self.ensemble_config = ensemble_config or EnsembleConfig()
        
        # Core components
        self.template_db: Optional[TemplateDB] = None
        self.scorer = ConfidenceScorer()
        self.dg_solver = DistanceGeometrySolver(
            max_iterations=self.ensemble_config.num_refinement_steps,
            learning_rate=self.ensemble_config.refinement_lr,
        )
        self.hybrid_generator = HybridCoordinateGenerator()
        
        # Statistics
        self.stats: Dict[str, Any] = {
            'targets_processed': 0,
            'total_candidates': 0,
            'avg_best_confidence': 0.0,
            'avg_template_identity': 0.0,
        }
    
    def load_template_database(
        self,
        pdb_dir: Optional[str] = None,
        metadata_file: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        """Load or build template database.
        
        Prefers training template DB (5,716 structures) over PDB-RNA (242 structures).
        """
        # First try training template database (24x more structures)
        if cache_path:
            cache_path_obj = Path(cache_path)
            training_db_path = cache_path_obj.parent / "training_template_db.pkl"
            
            if training_db_path.exists():
                print(f"Loading expanded training template DB from {training_db_path}")
                self.template_db = TemplateDB.load_from_training(str(training_db_path))
                print(f"Template database ready: {len(self.template_db)} structures (from training data)")
                return
        
        # Fall back to original PDB-RNA database
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached template database from {cache_path}")
            self.template_db = TemplateDB.load(cache_path)
        else:
            print("Building template database...")
            self.template_db = TemplateDB(k=self.config.kmer_size)
            
            pdb_dir = pdb_dir or str(self.config.paths.pdb_rna_dir)
            metadata_file = metadata_file or str(self.config.paths.release_dates_file)
            
            self.template_db.build_from_directory(
                pdb_dir, metadata_file, self.config.max_files
            )
            
            if cache_path:
                self.template_db.save(cache_path)
        
        print(f"Template database ready: {len(self.template_db.templates)} templates")
    
    def predict_single(
        self,
        target_id: str,
        sequence: str,
        temporal_cutoff: str,
        msa_path: Optional[str] = None,
        ground_truth: Optional[np.ndarray] = None,
    ) -> PredictionSet:
        """
        Predict structure for single target using ensemble approach.
        
        Args:
            target_id: Target identifier
            sequence: RNA sequence
            temporal_cutoff: Templates before this date only
            msa_path: Optional path to MSA file
            ground_truth: Optional ground truth for validation
            
        Returns:
            PredictionSet with top 5 models
        """
        L = len(sequence)
        candidates = []
        
        # --- Source 1: Template-based models ---
        template_candidates = self._generate_template_candidates(
            sequence, temporal_cutoff
        )
        candidates.extend(template_candidates)
        
        # --- Source 2: ML distance-based models ---
        if self.ensemble_config.use_ml_distances:
            ml_candidates = self._generate_ml_candidates(sequence, msa_path)
            candidates.extend(ml_candidates)
        
        # --- Source 3: Hybrid models ---
        if template_candidates and self.ensemble_config.use_ml_distances:
            hybrid_candidates = self._generate_hybrid_candidates(
                sequence, template_candidates[0], msa_path
            )
            candidates.extend(hybrid_candidates)
        
        # Fallback if no candidates
        if not candidates:
            baseline = generate_geometric_baseline(L)
            candidates = [baseline]
        
        # --- Predict distance constraints for scoring ---
        if self.ensemble_config.use_ml_distances:
            distances, confidence = ml_predict_distances(sequence)
        else:
            distances, confidence = None, None
        
        # --- Score and refine candidates ---
        scored_candidates = []
        for coords in candidates:
            # Refine if we have distance constraints
            if distances is not None:
                coords, _ = self.dg_solver.refine_with_constraints(
                    coords, distances, confidence,
                    num_steps=self.ensemble_config.num_refinement_steps // 2
                )
            
            # Score
            score = self.scorer.score(coords, distances, confidence)
            scored_candidates.append((coords, score))
        
        # --- Select diverse top-k ---
        scored_candidates.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        selected_indices = self.scorer.select_diverse_top_k(
            [c[0] for c in scored_candidates],
            k=self.ensemble_config.num_output_models,
            distance_constraints=distances,
            constraint_confidence=confidence,
            diversity_threshold=self.ensemble_config.diversity_threshold,
        )
        
        models = [scored_candidates[i][0] for i in selected_indices]
        confidences = [scored_candidates[i][1].overall_score for i in selected_indices]
        
        # Pad if needed
        while len(models) < self.ensemble_config.num_output_models:
            if models:
                perturbed = models[-1] + np.random.randn(L, 3) * 0.5
                models.append(perturbed)
                confidences.append(confidences[-1] * 0.9)
            else:
                models.append(generate_geometric_baseline(L))
                confidences.append(0.1)
        
        # Update stats
        self.stats['targets_processed'] += 1
        self.stats['total_candidates'] += len(candidates)
        self.stats['avg_best_confidence'] = (
            (self.stats['avg_best_confidence'] * (self.stats['targets_processed'] - 1) +
             confidences[0]) / self.stats['targets_processed']
        )
        
        # Validation if ground truth available
        if ground_truth is not None:
            tm_scores = [compute_tm_score(m, ground_truth) for m in models]
            return EnsemblePredictionSet(
                target_id=target_id,
                sequence=sequence,
                models=models[:self.ensemble_config.num_output_models],
                confidences=confidences[:self.ensemble_config.num_output_models],
                tm_scores=tm_scores[:self.ensemble_config.num_output_models],
            )
        
        return EnsemblePredictionSet(
            target_id=target_id,
            sequence=sequence,
            models=models[:self.ensemble_config.num_output_models],
            confidences=confidences[:self.ensemble_config.num_output_models],
        )
    
    def _generate_template_candidates(
        self,
        sequence: str,
        temporal_cutoff: str,
    ) -> List[np.ndarray]:
        """Generate candidate models from template hits."""
        if self.template_db is None:
            return []
        
        hits = self.template_db.search(
            sequence,
            temporal_cutoff=temporal_cutoff,
            max_hits=self.ensemble_config.max_templates,
            min_identity=self.ensemble_config.min_template_identity,
        )
        
        candidates = []
        if not hits:
            print(" No templates found.", end="")
            return candidates
            
        print(f" Found {len(hits)} templates.", end="")
        for hit in hits:
            # Get template coordinates from database
            template_coords_obj = self.template_db.get_template_coords(hit.pdb_id, hit.chain_id)
            if template_coords_obj is None:
                continue
                
            # Extract coordinates as array from Residue objects
            template_coords = np.array([(r.x, r.y, r.z) for r in template_coords_obj.residues])
                
            # Align sequences
            from .alignment import align_sequences
            alignment = align_sequences(sequence, hit.sequence)
            
            # Transfer coordinates - segment-based to prevent disparate template regions
            raw_coords = np.full((len(sequence), 3), np.nan)
            
            # 1. Extract all mapped coordinates first
            for i in range(len(sequence)):
                t_idx = alignment.query_to_template[i]
                if t_idx >= 0 and t_idx < len(template_coords):
                    raw_coords[i] = template_coords[t_idx]
            
            # 2. Identify continuous segments based on spatial connectivity
            non_nan_indices = [i for i in range(len(sequence)) if not np.isnan(raw_coords[i, 0])]
            
            if not non_nan_indices:
                continue  # No mapped residues, skip this template
            
            segments = []
            current_segment = [non_nan_indices[0]]
            
            for k in range(1, len(non_nan_indices)):
                curr_idx = non_nan_indices[k]
                prev_idx = non_nan_indices[k-1]
                
                # Check spatial distance
                dist = np.linalg.norm(raw_coords[curr_idx] - raw_coords[prev_idx])
                
                # Allow tolerance proportional to sequence separation
                seq_dist = curr_idx - prev_idx
                allowed_dist = 8.0 * seq_dist + 10.0  # ~8A per residue gap + buffer
                
                if dist < allowed_dist:
                    current_segment.append(curr_idx)
                else:
                    segments.append(current_segment)
                    current_segment = [curr_idx]
            segments.append(current_segment)
            
            # 3. Keep longest segment only
            longest_segment = max(segments, key=len)
            
            # 4. Fill output coords if segment is significant (at least 3 residues)
            coords = np.full((len(sequence), 3), np.nan)
            if len(longest_segment) < 3:
                continue  # Skip if no significant contiguous segment
                
            for idx in longest_segment:
                coords[idx] = raw_coords[idx]
            
            # Fill gaps - use vectorized fill_gaps_array
            coords = fill_gaps_array(coords)
            
            if not np.any(np.isnan(coords)):
                # Center the candidate immediately to prevent large floating point values
                coords -= np.mean(coords, axis=0)
                candidates.append(coords)
                
                # Update stats
                if not hasattr(self, '_template_identities'):
                    self._template_identities = []
                self._template_identities.append(hit.identity)
        
        return candidates
    
    def _generate_ml_candidates(
        self,
        sequence: str,
        msa_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate candidates from ML distance prediction."""
        candidates = []
        
        # Get predicted distances
        try:
            # Try to get covariation if MSA available
            covariation = None
            if msa_path:
                try:
                    from .msa.parser import parse_msa
                    from .msa.covariation import CovariationAnalyzer
                    msa = parse_msa(msa_path)
                    analyzer = CovariationAnalyzer()
                    covariation = analyzer.compute_dca_scores(msa)
                except Exception:
                    pass
            
            distances, confidence = ml_predict_distances(
                sequence,
                covariation_scores=covariation,
            )
            
            # Generate multiple solutions
            solutions = self.dg_solver.sample_diverse(
                distances, confidence,
                num_samples=5,
                noise_scale=3.0,
            )
            
            candidates.extend([sol[0] for sol in solutions])
            
        except Exception as e:
            # Fallback to geometric baseline variants
            L = len(sequence)
            base = generate_geometric_baseline(L)
            for i in range(3):
                perturbed = base + np.random.randn(L, 3) * (i + 1) * 2
                candidates.append(perturbed)
        
        return candidates
    
    def _generate_hybrid_candidates(
        self,
        sequence: str,
        template_coords: np.ndarray,
        msa_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate hybrid template + distance geometry candidates."""
        candidates = []
        
        try:
            # Get ML distances
            distances, confidence = ml_predict_distances(sequence)
            
            L = len(sequence)
            
            # Create template mask (non-NaN positions)
            template_mask = ~np.isnan(template_coords).any(axis=1)
            
            # Generate hybrid with different weights
            for template_weight in [0.8, 0.5, 0.3]:
                hybrid = self.hybrid_generator.generate(
                    L,
                    template_coords=template_coords,
                    template_mask=template_mask,
                    predicted_distances=distances,
                    distance_confidence=confidence * (1 - template_weight),
                )
                candidates.append(hybrid)
            
        except Exception:
            pass
        
        return candidates
    
    def _transfer_coordinates(
        self,
        query_seq: str,
        template_coords,
    ) -> np.ndarray:
        """Transfer coordinates from template to query."""
        from .alignment import align_sequences, transfer_coordinates
        
        # First align sequences
        alignment = align_sequences(query_seq, template_coords.sequence)
        
        # Then transfer coordinates
        result = transfer_coordinates(query_seq, template_coords, alignment)
        
        return result.coords
    
    def predict_all(
        self,
        sequences_df,
        output_path: Optional[str] = None,
        ground_truth_df=None,
        verbose: bool = True,
    ) -> List[PredictionSet]:
        """
        Predict structures for all sequences.
        
        Args:
            sequences_df: DataFrame with target_id, sequence, temporal_cutoff
            output_path: Optional path to save submission
            ground_truth_df: Optional DataFrame with ground truth
            verbose: Print progress
            
        Returns:
            List of PredictionSet
        """
        predictions = []
        total = len(sequences_df)
        
        # Load ground truth if available
        ground_truth = {}
        if ground_truth_df is not None:
            ground_truth = self._load_ground_truth(ground_truth_df)
        
        start_time = time.time()
        
        for idx, row in sequences_df.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']
            temporal_cutoff = row.get('temporal_cutoff', '2099-12-31')
            
            # Get ground truth if available
            gt = ground_truth.get(target_id)
            
            if verbose:
                print(f"[{idx+1}/{total}] {target_id} ({len(sequence)}nt)...", end=" ")
            
            pred = self.predict_single(
                target_id, sequence, temporal_cutoff,
                ground_truth=gt,
            )
            predictions.append(pred)
            
            if verbose:
                if pred.tm_scores:
                    print(f"TM={pred.tm_scores[0]:.3f}, conf={pred.confidences[0]:.3f}")
                else:
                    print(f"conf={pred.confidences[0]:.3f}")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nProcessed {total} targets in {elapsed:.1f}s")
            print(f"Avg candidates per target: {self.stats['total_candidates'] / max(1, total):.1f}")
            print(f"Avg best confidence: {self.stats['avg_best_confidence']:.3f}")
        
        # Write submission
        if output_path:
            write_submission(predictions, output_path)
            if verbose:
                print(f"Submission saved to {output_path}")
        
        return predictions
    
    def _load_ground_truth(self, df) -> Dict[str, np.ndarray]:
        """Load ground truth coordinates from DataFrame."""
        ground_truth = {}
        # Ensure ID column is present
        if 'ID' not in df.columns:
            return {}
            
        grouped = df.groupby(df['ID'].str.rsplit('_', n=1).str[0])
        
        for target_id, group in grouped:
            coords_list = []
            for _, row in group.iterrows():
                x, y, z = row.get('x_1', np.nan), row.get('y_1', np.nan), row.get('z_1', np.nan)
                
                # Handle outliers (-1e+18 is used for missing coordinates in Kaggle data)
                if x < -1e10 or y < -1e10 or z < -1e10:
                    coords_list.append([np.nan, np.nan, np.nan])
                else:
                    coords_list.append([x, y, z])
                    
            ground_truth[target_id] = np.array(coords_list)
        
        return ground_truth


def run_ensemble_pipeline(
    sequences_file: str = None,
    output_path: str = None,
    ground_truth_file: str = None,
    cache_path: str = None,
    **kwargs,
) -> List[PredictionSet]:
    """
    Convenience function to run ensemble pipeline.
    
    Args:
        sequences_file: Path to sequences CSV
        output_path: Output submission file path
        ground_truth_file: Optional path to ground truth CSV
        cache_path: Path to template database cache
        **kwargs: Additional configuration
        
    Returns:
        List of PredictionSet
    """
    import pandas as pd
    
    # Load configuration
    config = PipelineConfig()
    ensemble_config = EnsembleConfig(**{
        k: v for k, v in kwargs.items()
        if hasattr(EnsembleConfig, k)
    })
    
    # Create pipeline
    pipeline = EnsemblePipeline(config, ensemble_config)
    
    # Load template database
    cache_path = cache_path or str(config.paths.output_dir / "template_db.pkl")
    pipeline.load_template_database(cache_path=cache_path)
    
    # Load sequences
    if sequences_file is None:
        sequences_file = str(config.paths.test_sequences_file)
    
    sequences_df = pd.read_csv(sequences_file)
    
    # Load ground truth if available
    ground_truth_df = None
    if ground_truth_file:
        ground_truth_df = pd.read_csv(ground_truth_file)
    
    # Run predictions
    if output_path is None:
        output_path = str(config.paths.output_dir / "submission.csv")
    
    predictions = pipeline.predict_all(
        sequences_df,
        output_path=output_path,
        ground_truth_df=ground_truth_df,
        verbose=True,
    )
    
    # Print summary if we have TM-scores
    if predictions and predictions[0].tm_scores:
        print("\n" + "=" * 60)
        print("ENSEMBLE PIPELINE RESULTS")
        print("=" * 60)
        
        tm_scores = [p.tm_scores[0] for p in predictions if p.tm_scores]
        avg_tm = np.mean(tm_scores)
        
        print(f"Average TM-score: {avg_tm:.3f}")
        print(f"Targets ≥0.8: {sum(1 for t in tm_scores if t >= 0.8)}/{len(tm_scores)}")
        print(f"Targets ≥0.5: {sum(1 for t in tm_scores if t >= 0.5)}/{len(tm_scores)}")
    
    return predictions
