"""
Convert predicted distance maps to 3D coordinates.
Uses Multi-Dimensional Scaling (MDS) and gradient-based refinement.
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import minimize


class DistanceToCoords:
    """
    Convert inter-residue distance matrix to 3D coordinates.
    
    Uses classical MDS for initialization, followed by gradient-based
    refinement to satisfy distance constraints and RNA geometry.
    
    Enhanced with:
    - Backbone angle constraints
    - Improved MDS with noise injection for diversity
    - Better clash detection and resolution
    - RNA-specific geometry constraints
    """
    
    # RNA backbone geometry constraints (Angstroms)
    BOND_LENGTH_C1_C1 = 5.9      # Adjacent C1'-C1' distance
    BOND_LENGTH_TOLERANCE = 0.5  # Allowed deviation
    MIN_CONTACT_DISTANCE = 3.0   # Minimum non-bonded distance
    
    # Additional RNA geometry parameters
    BACKBONE_ANGLE_IDEAL = 155.0    # Degrees - ideal backbone angle
    BACKBONE_ANGLE_TOLERANCE = 15.0 # Allowed deviation
    BASE_PAIR_DISTANCE = 5.4        # C1'-C1' in base pair
    STACKING_DISTANCE = 3.4         # Base stacking distance
    
    def __init__(
        self,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        clash_weight: float = 1.0,
        distance_weight: float = 1.0,
        bond_weight: float = 5.0,
        angle_weight: float = 2.0,
        use_enhanced: bool = True,
    ):
        """
        Args:
            max_iterations: Max optimization iterations
            learning_rate: Gradient descent step size
            clash_weight: Weight for clash penalty
            distance_weight: Weight for distance constraints
            bond_weight: Weight for backbone bond constraints
            angle_weight: Weight for backbone angle constraints
            use_enhanced: Use enhanced geometry constraints
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.clash_weight = clash_weight
        self.distance_weight = distance_weight
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.use_enhanced = use_enhanced
    
    def convert(
        self,
        distances: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        initial_coords: Optional[np.ndarray] = None,
        sequence: Optional[str] = None,
    ) -> np.ndarray:
        """
        Convert distance matrix to 3D coordinates.
        
        Args:
            distances: (L, L) predicted distance matrix
            confidences: (L, L) confidence weights (optional)
            initial_coords: (L, 3) initial coordinates (optional)
            sequence: RNA sequence for geometry-aware refinement
            
        Returns:
            coords: (L, 3) optimized 3D coordinates
        """
        L = distances.shape[0]
        
        if confidences is None:
            confidences = np.ones_like(distances)
        
        # Initialize with MDS or provided coordinates
        if initial_coords is not None:
            coords = initial_coords.copy()
        else:
            coords = self._classical_mds(distances)
        
        # Check for NaN in coordinates
        if np.any(np.isnan(coords)):
            coords = self._helix_initialization(L)
        
        # Refine with gradient descent
        coords = self._gradient_refinement(coords, distances, confidences, sequence)
        
        return coords
    
    def convert_with_diversity(
        self,
        distances: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        num_models: int = 5,
        noise_scale: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Generate multiple coordinate sets with diversity.
        
        Args:
            distances: (L, L) predicted distance matrix
            confidences: (L, L) confidence weights
            num_models: Number of models to generate
            noise_scale: Scale of noise injection for diversity
            
        Returns:
            List of (L, 3) coordinate arrays
        """
        L = distances.shape[0]
        
        if confidences is None:
            confidences = np.ones_like(distances)
        
        models = []
        
        for i in range(num_models):
            # Different initialization strategies
            if i == 0:
                # Best MDS solution
                init_coords = self._classical_mds(distances)
            elif i == 1:
                # Helix initialization
                init_coords = self._helix_initialization(L)
            else:
                # MDS with noise
                init_coords = self._classical_mds(distances)
                noise = np.random.randn(L, 3) * noise_scale * (i - 1)
                init_coords += noise
            
            # Refine
            coords = self._gradient_refinement(
                init_coords, distances, confidences
            )
            models.append(coords)
        
        return models
    
    def _classical_mds(self, distances: np.ndarray) -> np.ndarray:
        """
        Classical Multi-Dimensional Scaling.
        
        Converts distance matrix to coordinates using eigendecomposition.
        Enhanced with better handling of invalid values.
        """
        L = distances.shape[0]
        
        # Make symmetric and handle invalid values
        D = (distances + distances.T) / 2
        D = np.nan_to_num(D, nan=15.0, posinf=15.0, neginf=0.0)
        np.fill_diagonal(D, 0)
        
        # Clip extreme values
        D = np.clip(D, 0, 100)
        
        # Squared distances
        D2 = D ** 2
        
        # Double centering
        H = np.eye(L) - np.ones((L, L)) / L
        B = -0.5 * H @ D2 @ H
        
        # Ensure symmetric
        B = (B + B.T) / 2
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(B)
        except np.linalg.LinAlgError:
            # Fallback: initialize along a helix
            return self._helix_initialization(L)
        
        # Take top 3 positive eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Handle negative eigenvalues (non-metric distances)
        eigenvalues = np.maximum(eigenvalues[:3], 1e-10)
        
        # Compute coordinates
        coords = eigenvectors[:, :3] * np.sqrt(eigenvalues[:3])
        
        # Center coordinates
        coords -= coords.mean(axis=0)
        
        return coords
    
    def _helix_initialization(self, L: int) -> np.ndarray:
        """Initialize as RNA A-form helix."""
        # A-form helix parameters
        rise_per_bp = 2.8  # Angstroms
        radius = 11.0      # Helix radius
        twist = np.radians(32.7)  # Twist per bp
        
        coords = np.zeros((L, 3))
        for i in range(L):
            angle = i * twist
            coords[i, 0] = radius * np.cos(angle)
            coords[i, 1] = radius * np.sin(angle)
            coords[i, 2] = i * rise_per_bp
        
        return coords
    
    def _gradient_refinement(
        self,
        coords: np.ndarray,
        target_distances: np.ndarray,
        confidences: np.ndarray,
        sequence: Optional[str] = None,
    ) -> np.ndarray:
        """
        Refine coordinates using gradient descent.
        
        Minimizes weighted distance errors while maintaining RNA geometry.
        Enhanced with angle constraints and better clash handling.
        """
        L = coords.shape[0]
        
        def objective(flat_coords):
            coords = flat_coords.reshape(L, 3)
            
            # Current distances
            curr_dist = squareform(pdist(coords))
            
            # Distance constraint loss (weighted by confidence)
            diff = curr_dist - target_distances
            distance_loss = np.sum(confidences * diff ** 2)
            
            # Backbone bond loss
            bond_loss = 0.0
            for i in range(L - 1):
                d = np.linalg.norm(coords[i] - coords[i + 1])
                bond_loss += (d - self.BOND_LENGTH_C1_C1) ** 2
            
            # Backbone angle loss (if enhanced mode)
            angle_loss = 0.0
            if self.use_enhanced and L >= 3:
                angle_loss = self._compute_angle_loss(coords)
            
            # Clash penalty (non-bonded atoms too close)
            clash_loss = 0.0
            for i in range(L):
                for j in range(i + 2, L):  # Skip bonded pairs
                    d = curr_dist[i, j]
                    if d < self.MIN_CONTACT_DISTANCE:
                        # Quadratic penalty below threshold
                        clash_loss += (self.MIN_CONTACT_DISTANCE - d) ** 2
            
            total_loss = (
                self.distance_weight * distance_loss +
                self.bond_weight * bond_loss +
                self.angle_weight * angle_loss +
                self.clash_weight * clash_loss
            )
            
            return total_loss
        
        # Optimize
        result = minimize(
            objective,
            coords.flatten(),
            method='L-BFGS-B',
            options={
                'maxiter': self.max_iterations,
                'ftol': 1e-6,
                'gtol': 1e-5,
            }
        )
        
        refined = result.x.reshape(L, 3)
        
        # Post-processing: resolve any remaining clashes
        if self.use_enhanced:
            refined = self._resolve_clashes(refined)
        
        return refined
    
    def _compute_angle_loss(self, coords: np.ndarray) -> float:
        """Compute backbone angle constraint loss."""
        L = coords.shape[0]
        angle_loss = 0.0
        
        ideal_angle_rad = np.radians(self.BACKBONE_ANGLE_IDEAL)
        tolerance_rad = np.radians(self.BACKBONE_ANGLE_TOLERANCE)
        
        for i in range(1, L - 1):
            # Vectors from i to neighbors
            v1 = coords[i - 1] - coords[i]
            v2 = coords[i + 1] - coords[i]
            
            # Compute angle
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                # Penalty for deviation from ideal
                angle_diff = abs(angle - ideal_angle_rad)
                if angle_diff > tolerance_rad:
                    angle_loss += (angle_diff - tolerance_rad) ** 2
        
        return angle_loss
    
    def _resolve_clashes(
        self, 
        coords: np.ndarray, 
        max_iterations: int = 100,
    ) -> np.ndarray:
        """Resolve remaining atomic clashes by local perturbation."""
        L = coords.shape[0]
        coords = coords.copy()
        
        for _ in range(max_iterations):
            clashes_found = False
            
            for i in range(L):
                for j in range(i + 2, L):
                    d = np.linalg.norm(coords[i] - coords[j])
                    
                    if d < self.MIN_CONTACT_DISTANCE and d > 0:
                        clashes_found = True
                        
                        # Push apart along the connecting vector
                        direction = coords[j] - coords[i]
                        direction = direction / (d + 1e-8)
                        
                        push = (self.MIN_CONTACT_DISTANCE - d) / 2 + 0.1
                        coords[i] -= direction * push
                        coords[j] += direction * push
            
            if not clashes_found:
                break
        
        return coords
    
    def _stacking_constraint_loss(
        self, 
        coords: np.ndarray, 
        sequence: str,
    ) -> float:
        """Penalize poor base stacking geometry."""
        L = len(sequence)
        stacking_loss = 0.0
        
        # Adjacent bases should stack (~3.4Å in helix direction)
        for i in range(L - 1):
            # Check if both nucleotides are in a helix region
            # (Simplified: just check backbone distance)
            d = np.linalg.norm(coords[i] - coords[i + 1])
            
            # If backbone is proper length, check vertical stacking
            if abs(d - self.BOND_LENGTH_C1_C1) < 1.0:
                # Estimate stacking based on z-component difference
                z_diff = abs(coords[i][2] - coords[i + 1][2])
                target_z = self.STACKING_DISTANCE
                
                if z_diff < target_z * 0.5:
                    # Bases are too flat - penalize
                    stacking_loss += (target_z - z_diff) ** 2
        
        return stacking_loss


    def compute_rmsd(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
    ) -> float:
        """Compute RMSD between two coordinate sets."""
        # Center both
        coords1 = coords1 - coords1.mean(axis=0)
        coords2 = coords2 - coords2.mean(axis=0)
        
        # Optimal rotation (Kabsch algorithm)
        H = coords1.T @ coords2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        coords1_rotated = coords1 @ R
        
        rmsd = np.sqrt(np.mean(np.sum((coords1_rotated - coords2) ** 2, axis=1)))
        return rmsd


class HybridCoordinateGenerator:
    """
    Generate coordinates using both template and distance predictions.
    
    Combines template-derived coordinates with learned distance constraints
    for improved structure prediction.
    """
    
    def __init__(
        self,
        template_weight: float = 0.5,
        distance_weight: float = 0.5,
        refinement_iterations: int = 500,
    ):
        """
        Args:
            template_weight: Weight for template coordinates
            distance_weight: Weight for distance predictions
            refinement_iterations: Optimization iterations
        """
        self.template_weight = template_weight
        self.distance_weight = distance_weight
        self.converter = DistanceToCoords(max_iterations=refinement_iterations)
    
    def generate(
        self,
        sequence: str,
        template_coords: Optional[np.ndarray] = None,
        template_mask: Optional[np.ndarray] = None,
        predicted_distances: Optional[np.ndarray] = None,
        distance_confidences: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate coordinates from templates and/or distance predictions.
        
        Args:
            sequence: RNA sequence
            template_coords: (L, 3) from template (may have gaps)
            template_mask: (L,) True where template has valid coords
            predicted_distances: (L, L) predicted distances
            distance_confidences: (L, L) confidence scores
            
        Returns:
            coords: (L, 3) final coordinates
        """
        L = len(sequence)
        
        # Case 1: Only template
        if predicted_distances is None and template_coords is not None:
            return self._fill_template_gaps(template_coords, template_mask)
        
        # Case 2: Only distance predictions
        if template_coords is None and predicted_distances is not None:
            return self.converter.convert(
                predicted_distances, 
                distance_confidences
            )
        
        # Case 3: Hybrid - combine both
        if template_coords is not None and predicted_distances is not None:
            return self._hybrid_generation(
                sequence,
                template_coords,
                template_mask,
                predicted_distances,
                distance_confidences,
            )
        
        # Case 4: No information - geometric baseline
        return self._geometric_baseline(L)
    
    def _hybrid_generation(
        self,
        sequence: str,
        template_coords: np.ndarray,
        template_mask: np.ndarray,
        predicted_distances: np.ndarray,
        distance_confidences: np.ndarray,
    ) -> np.ndarray:
        """Combine template coordinates with distance predictions."""
        L = len(sequence)
        
        if template_mask is None:
            template_mask = np.ones(L, dtype=bool)
        
        # Compute template-derived distances where available
        template_distances = np.full((L, L), np.nan)
        template_conf = np.zeros((L, L))
        
        for i in range(L):
            for j in range(L):
                if template_mask[i] and template_mask[j]:
                    d = np.linalg.norm(template_coords[i] - template_coords[j])
                    template_distances[i, j] = d
                    template_conf[i, j] = 1.0
        
        # Merge distances: weighted combination where both available
        merged_distances = np.copy(predicted_distances)
        merged_conf = np.copy(distance_confidences) if distance_confidences is not None else np.ones((L, L))
        
        for i in range(L):
            for j in range(L):
                if not np.isnan(template_distances[i, j]):
                    # Weighted average
                    w_t = self.template_weight * template_conf[i, j]
                    w_d = self.distance_weight * merged_conf[i, j]
                    total_w = w_t + w_d + 1e-8
                    
                    merged_distances[i, j] = (
                        w_t * template_distances[i, j] + 
                        w_d * predicted_distances[i, j]
                    ) / total_w
                    merged_conf[i, j] = max(template_conf[i, j], merged_conf[i, j])
        
        # Use template coords as initialization
        init_coords = np.copy(template_coords)
        
        # Fill any gaps with MDS-derived positions
        if not template_mask.all():
            gap_init = self.converter._classical_mds(merged_distances)
            for i in range(L):
                if not template_mask[i]:
                    init_coords[i] = gap_init[i]
        
        # Refine
        return self.converter.convert(
            merged_distances,
            merged_conf,
            initial_coords=init_coords,
        )
    
    def _fill_template_gaps(
        self,
        template_coords: np.ndarray,
        template_mask: np.ndarray,
    ) -> np.ndarray:
        """Fill gaps in template coordinates."""
        L = template_coords.shape[0]
        coords = np.copy(template_coords)
        
        if template_mask is None or template_mask.all():
            return coords
        
        # Linear interpolation for gaps
        for i in range(L):
            if not template_mask[i]:
                # Find nearest valid coords
                left = right = i
                while left > 0 and not template_mask[left]:
                    left -= 1
                while right < L - 1 and not template_mask[right]:
                    right += 1
                
                if template_mask[left] and template_mask[right]:
                    # Interpolate
                    t = (i - left) / (right - left + 1e-8)
                    coords[i] = (1 - t) * coords[left] + t * coords[right]
                elif template_mask[left]:
                    # Extend from left
                    coords[i] = coords[left] + (i - left) * np.array([5.9, 0, 0])
                elif template_mask[right]:
                    # Extend from right
                    coords[i] = coords[right] - (right - i) * np.array([5.9, 0, 0])
        
        return coords
    
    def _geometric_baseline(self, L: int) -> np.ndarray:
        """Generate geometric baseline coordinates."""
        return self.converter._helix_initialization(L)
