"""
Distance Geometry Solver.

Converts inter-residue distance matrices to 3D coordinates using:
- Metric Matrix Embedding (MDS)
- Iterative refinement with constraint satisfaction
- Bounds smoothing for distance bounds
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import minimize


class DistanceGeometrySolver:
    """
    Convert distance constraints to 3D coordinates.
    
    Uses Multi-Dimensional Scaling (MDS) for initial embedding,
    followed by gradient-based refinement to satisfy constraints.
    
    Methods:
        mds_embed: Classical MDS for initial coordinates
        refine_with_constraints: Gradient descent refinement
        solve: Full pipeline from distances to coordinates
        sample_diverse: Generate multiple solutions
    """
    
    def __init__(
        self,
        max_iterations: int = 500,
        learning_rate: float = 0.1,
        convergence_threshold: float = 1e-4,
        use_bounds: bool = True,
    ):
        """
        Initialize solver.
        
        Args:
            max_iterations: Max refinement iterations
            learning_rate: Step size for gradient descent
            convergence_threshold: Stop if loss change < this
            use_bounds: Apply distance bounds smoothing
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.use_bounds = use_bounds
        
        # RNA-specific parameters
        self.min_c1_distance = 3.0  # Minimum C1'-C1' distance
        self.max_c1_distance = 50.0  # Maximum reasonable distance
        self.consecutive_distance = 5.9  # Adjacent C1' distance
        self.consecutive_tolerance = 1.0
    
    def solve(
        self,
        distance_matrix: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        initial_coords: Optional[np.ndarray] = None,
        num_refinement_steps: int = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Solve for 3D coordinates from distance matrix.
        
        Args:
            distance_matrix: (L, L) target distances
            confidence: (L, L) constraint confidence (0-1)
            initial_coords: Optional starting coordinates
            num_refinement_steps: Override max_iterations
            
        Returns:
            coords: (L, 3) optimized coordinates
            final_loss: Final constraint violation loss
        """
        L = len(distance_matrix)
        
        # Apply bounds smoothing if enabled
        if self.use_bounds:
            distance_matrix = self._smooth_bounds(distance_matrix)
        
        # Get initial coordinates
        if initial_coords is not None:
            coords = initial_coords.copy()
        else:
            coords = self.mds_embed(distance_matrix)
        
        # Refine
        if num_refinement_steps is None:
            num_refinement_steps = self.max_iterations
            
        coords, loss = self.refine_with_constraints(
            coords, distance_matrix, confidence, num_refinement_steps
        )
        
        return coords, loss
    
    def mds_embed(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Classical Multi-Dimensional Scaling embedding.
        
        Converts distance matrix to coordinates using eigendecomposition
        of the doubly-centered Gram matrix.
        
        Args:
            distance_matrix: (L, L) symmetric distance matrix
            
        Returns:
            coords: (L, 3) coordinates
        """
        L = len(distance_matrix)
        
        # Ensure symmetry
        D = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(D, 0)
        
        # Squared distances
        D_sq = D ** 2
        
        # Double centering to get Gram matrix: G = -0.5 * J * D^2 * J
        # where J = I - (1/L) * 1 * 1^T is the centering matrix
        row_mean = D_sq.mean(axis=1, keepdims=True)
        col_mean = D_sq.mean(axis=0, keepdims=True)
        total_mean = D_sq.mean()
        
        G = -0.5 * (D_sq - row_mean - col_mean + total_mean)
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(G)
        except np.linalg.LinAlgError:
            # Fallback: random initialization
            return np.random.randn(L, 3) * 10
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top 3 dimensions
        # Handle negative eigenvalues (non-Euclidean distances)
        eigenvalues_pos = np.maximum(eigenvalues[:3], 0)
        
        if np.sum(eigenvalues_pos) < 1e-10:
            # Degenerate case
            return np.random.randn(L, 3) * 10
        
        # Coordinates: X = V * sqrt(Lambda)
        coords = eigenvectors[:, :3] * np.sqrt(eigenvalues_pos)
        
        return coords
    
    def refine_with_constraints(
        self,
        coords: np.ndarray,
        target_distances: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        num_steps: int = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Refine coordinates to satisfy distance constraints.
        
        Vectorized implementation with adaptive learning rate.
        """
        L = len(coords)
        coords = coords.copy().astype(np.float64)
        
        if confidence is None:
            confidence = np.ones_like(target_distances)
        
        if num_steps is None:
            num_steps = self.max_iterations
            
        # Limit steps for very long sequences
        if L > 1000:
            num_steps = min(num_steps, 50)
        
        # Build constraint mask
        mask = (confidence > 0.05) & ~np.eye(L, dtype=bool)
        num_constraints = np.sum(mask) + (L - 1)  # pairs + consecutive
        
        # Adaptive learning rate - start smaller for stability
        lr = 0.01 / np.sqrt(L)  # Scale with sequence length
        
        best_coords = coords.copy()
        best_loss = float('inf')
        no_improvement_count = 0
        
        for step in range(num_steps):
            # 1. Coordinate differences
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            
            # 2. Current distances
            current_dist = np.linalg.norm(diff, axis=2)
            current_dist_safe = np.where(current_dist < 1e-6, 1e-6, current_dist)
            
            # 3. Distance errors
            dist_error = current_dist - target_distances
            
            # 4. Total Loss (normalized by number of constraints)
            weighted_error = dist_error * mask * confidence
            loss = np.sum(weighted_error ** 2) / num_constraints
            
            # Consecutive backbone constraints
            consec_diff = coords[1:] - coords[:-1]
            consec_dist = np.linalg.norm(consec_diff, axis=1)
            consec_error = consec_dist - self.consecutive_distance
            loss += 10.0 * np.mean(consec_error ** 2)
            
            # Check for NaNs
            if not np.isfinite(loss):
                coords = best_coords.copy()
                break
            
            # Track best solution
            if loss < best_loss:
                best_loss = loss
                best_coords = coords.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                # Early stopping if no improvement
                if no_improvement_count > 20:
                    coords = best_coords.copy()
                    break
                # Reduce learning rate on divergence
                lr *= 0.9
            
            # Check convergence
            if loss < self.convergence_threshold:
                break
            
            # 5. Gradients (normalized)
            grad_scalars = 2 * dist_error * mask * confidence / current_dist_safe / num_constraints
            grad_dist = np.sum(grad_scalars[:, :, np.newaxis] * diff, axis=1)
            
            # Gradient from consecutive constraints
            grad_consec = np.zeros_like(coords)
            consec_dist_safe = np.where(consec_dist < 1e-6, 1e-6, consec_dist)
            consec_grad_scalars = 20.0 * consec_error / consec_dist_safe / (L - 1)
            
            consec_vecs = consec_diff * consec_grad_scalars[:, np.newaxis]
            grad_consec[:-1] += consec_vecs
            grad_consec[1:] -= consec_vecs
            
            total_gradient = grad_dist + grad_consec
            
            # Gradient clipping
            grad_norm = np.linalg.norm(total_gradient)
            if grad_norm > 10.0:
                total_gradient = total_gradient * (10.0 / grad_norm)
            
            # Update
            coords -= lr * total_gradient
            
            # Clamp coordinates to prevent extreme values
            coords = np.clip(coords, -200, 200)
            
            # Centering
            coords -= coords.mean(axis=0)
        
        return best_coords, best_loss
    
    def sample_diverse(
        self,
        distance_matrix: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        num_samples: int = 10,
        noise_scale: float = 2.0,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Generate diverse coordinate solutions.
        
        Starts from different random perturbations of MDS solution
        to sample multiple local minima.
        
        Args:
            distance_matrix: (L, L) target distances
            confidence: (L, L) constraint confidence
            num_samples: Number of solutions to generate
            noise_scale: Scale of initial perturbations
            
        Returns:
            List of (coordinates, loss) tuples sorted by loss
        """
        L = len(distance_matrix)
        
        # Base MDS solution
        base_coords = self.mds_embed(distance_matrix)
        
        solutions = []
        
        # First solution: unperturbed
        coords, loss = self.solve(distance_matrix, confidence, base_coords)
        solutions.append((coords, loss))
        
        # Additional solutions with perturbations
        for i in range(num_samples - 1):
            perturbed = base_coords + np.random.randn(L, 3) * noise_scale
            coords, loss = self.solve(distance_matrix, confidence, perturbed)
            solutions.append((coords, loss))
        
        # Sort by loss (ascending)
        solutions.sort(key=lambda x: x[1])
        
        return solutions
    
    def _smooth_bounds(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply triangle inequality bounds smoothing.
        
        Optimized for large matrices.
        """
        L = len(distances)
        if L > 500:
            # Skip expensive smoothing for very large structures
            return distances
            
        D = distances.copy()
        
        # Ensure symmetry
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        
        # Clip to valid range
        D = np.clip(D, self.min_c1_distance, self.max_c1_distance)
        
        # Enforce consecutive distance constraints
        for i in range(L - 1):
            D[i, i+1] = self.consecutive_distance
            D[i+1, i] = self.consecutive_distance
        
        # Triangle inequality smoothing (Floyd-Warshall style)
        # Optimized with vectorized row updates
        max_smoothing_iterations = 2
        
        for _ in range(max_smoothing_iterations):
            for k in range(L):
                # D[i,j] = min(D[i,j], D[i,k] + D[k,j])
                # We can broadcast D[:, k] (L, 1) and D[k, :] (1, L)
                D_k = D[:, k:k+1] + D[k:k+1, :]
                D = np.minimum(D, D_k)
        
        return D


class HybridCoordinateGenerator:
    """
    Generate coordinates using hybrid template + distance geometry.
    
    Combines template-derived coordinates with distance predictions
    for regions without template coverage.
    """
    
    def __init__(
        self,
        template_weight: float = 0.7,
        distance_weight: float = 0.3,
    ):
        """
        Initialize generator.
        
        Args:
            template_weight: Weight for template-derived constraints
            distance_weight: Weight for predicted distances
        """
        self.template_weight = template_weight
        self.distance_weight = distance_weight
        self.solver = DistanceGeometrySolver()
    
    def generate(
        self,
        sequence_length: int,
        template_coords: Optional[np.ndarray] = None,
        template_mask: Optional[np.ndarray] = None,
        predicted_distances: Optional[np.ndarray] = None,
        distance_confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate coordinates from hybrid constraints.
        
        Args:
            sequence_length: Length of target sequence
            template_coords: (L, 3) template coordinates (may have NaN)
            template_mask: (L,) boolean mask for valid template positions
            predicted_distances: (L, L) ML-predicted distances
            distance_confidence: (L, L) prediction confidence
            
        Returns:
            coords: (L, 3) generated coordinates
        """
        L = sequence_length
        
        # If we have good template coverage, start from template
        if template_coords is not None and template_mask is not None:
            coverage = np.sum(template_mask) / L
            if coverage > 0.5:
                # Use template as base, refine gaps
                return self._refine_from_template(
                    template_coords, template_mask,
                    predicted_distances, distance_confidence
                )
        
        # Otherwise, use pure distance geometry
        if predicted_distances is not None:
            coords, _ = self.solver.solve(
                predicted_distances,
                distance_confidence,
            )
            return coords
        
        # Fallback: generate helix
        return self._generate_helix(L)
    
    def _refine_from_template(
        self,
        template_coords: np.ndarray,
        template_mask: np.ndarray,
        predicted_distances: Optional[np.ndarray],
        distance_confidence: Optional[np.ndarray],
    ) -> np.ndarray:
        """Refine template coordinates with distance constraints."""
        L = len(template_coords)
        coords = template_coords.copy()
        
        # Fill gaps with interpolation first
        coords = self._fill_gaps(coords, template_mask)
        
        # If we have distance predictions, refine
        if predicted_distances is not None:
            # Create combined constraint matrix
            # Higher weight for template-covered regions
            combined_confidence = np.zeros((L, L))
            
            if distance_confidence is not None:
                combined_confidence = distance_confidence.copy()
            
            # Downweight constraints involving template-covered regions
            for i in range(L):
                for j in range(L):
                    if template_mask[i] and template_mask[j]:
                        combined_confidence[i, j] *= 0.3
                    elif template_mask[i] or template_mask[j]:
                        combined_confidence[i, j] *= 0.7
            
            coords, _ = self.solver.refine_with_constraints(
                coords, predicted_distances, combined_confidence, num_steps=200
            )
        
        return coords
    
    def _fill_gaps(
        self,
        coords: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill gaps in coordinates using linear interpolation."""
        L = len(coords)
        filled = coords.copy()
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return self._generate_helix(L)
        
        if len(valid_indices) == L:
            return coords
        
        # Linear interpolation between valid points
        for dim in range(3):
            valid_values = coords[valid_mask, dim]
            all_indices = np.arange(L)
            filled[:, dim] = np.interp(all_indices, valid_indices, valid_values)
        
        return filled
    
    def _generate_helix(self, length: int) -> np.ndarray:
        """Generate A-form RNA helix coordinates."""
        coords = np.zeros((length, 3))
        radius = 10.0
        rise_per_residue = 2.8
        residues_per_turn = 11
        
        for i in range(length):
            angle = 2 * np.pi * i / residues_per_turn
            coords[i, 0] = radius * np.cos(angle)
            coords[i, 1] = radius * np.sin(angle)
            coords[i, 2] = i * rise_per_residue
        
        return coords
