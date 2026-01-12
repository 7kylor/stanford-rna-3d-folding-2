"""
Geometry adjustment for metal binding sites.
Refines local structure around predicted Mg²⁺ coordination sites.
"""

import numpy as np
from typing import List, Tuple, Optional


class MetalSiteGeometry:
    """
    Adjust local geometry around predicted metal binding sites.
    
    Mg²⁺ typically coordinates 6 ligands in octahedral geometry.
    Common ligands: phosphate oxygens, N7 of purines, water molecules.
    """
    
    # Metal coordination parameters
    MG_COORD_DISTANCE = 2.1  # Å, typical Mg-O/Mg-N distance
    MG_COORD_DISTANCE_RANGE = (1.9, 2.4)  # Acceptable range
    COORDINATION_NUMBER = 6  # Octahedral
    
    # Interaction types and their ideal distances
    INTERACTION_DISTANCES = {
        'phosphate_O': 2.1,   # Mg-O(P)
        'N7_purine': 2.2,     # Mg-N7 of G/A
        'O6_guanine': 2.1,    # Mg-O6 of G
        'water': 2.1,         # Mg-O(water)
        'ribose_O': 2.3,      # Mg-O(ribose)
    }
    
    def __init__(
        self,
        adjustment_strength: float = 0.3,
        max_displacement: float = 2.0,
    ):
        """
        Initialize geometry adjuster.
        
        Args:
            adjustment_strength: Strength of coordinate adjustment (0-1)
            max_displacement: Maximum displacement allowed (Å)
        """
        self.adjustment_strength = adjustment_strength
        self.max_displacement = max_displacement
    
    def adjust_coordinates(
        self,
        coords: np.ndarray,
        binding_sites: List[int],
        sequence: str,
    ) -> np.ndarray:
        """
        Adjust coordinates around predicted Mg²⁺ sites.
        
        Applies constraints to create favorable metal coordination geometry:
        - Tightens distances for potential ligands
        - Adjusts local geometry for octahedral coordination
        
        Args:
            coords: (L, 3) C1' coordinates
            binding_sites: List of predicted binding residue indices
            sequence: RNA sequence
            
        Returns:
            Adjusted coordinates (L, 3)
        """
        if len(binding_sites) == 0:
            return coords
        
        adjusted = coords.copy()
        
        for site_idx in binding_sites:
            adjusted = self._adjust_around_site(
                adjusted,
                site_idx,
                sequence,
            )
        
        return adjusted
    
    def _adjust_around_site(
        self,
        coords: np.ndarray,
        site_idx: int,
        sequence: str,
    ) -> np.ndarray:
        """
        Adjust geometry around a single binding site.
        
        Args:
            coords: Coordinates
            site_idx: Binding site index
            sequence: RNA sequence
            
        Returns:
            Adjusted coordinates
        """
        L = len(coords)
        adjusted = coords.copy()
        
        # Find neighboring residues
        neighbors = self._find_coordination_neighbors(coords, site_idx)
        
        if len(neighbors) == 0:
            return adjusted
        
        # Estimate metal position (center of coordinating atoms)
        metal_pos = self._estimate_metal_position(coords, site_idx, neighbors)
        
        # Adjust neighbor positions for optimal coordination
        for neighbor_idx, _ in neighbors:
            current_pos = coords[neighbor_idx]
            current_dist = np.linalg.norm(current_pos - metal_pos)
            
            # Calculate ideal distance based on nucleotide type
            nt = sequence[neighbor_idx].upper()
            ideal_dist = self._get_ideal_distance(nt)
            
            # Adjust position
            if current_dist > 1e-8:
                direction = (current_pos - metal_pos) / current_dist
                
                # Move toward ideal distance
                dist_adjustment = (ideal_dist - current_dist) * self.adjustment_strength
                dist_adjustment = np.clip(
                    dist_adjustment,
                    -self.max_displacement,
                    self.max_displacement,
                )
                
                adjusted[neighbor_idx] = current_pos + direction * dist_adjustment
        
        return adjusted
    
    def _find_coordination_neighbors(
        self,
        coords: np.ndarray,
        site_idx: int,
        search_radius: float = 8.0,
    ) -> List[Tuple[int, float]]:
        """
        Find residues that could coordinate the metal.
        
        Args:
            coords: Coordinates
            site_idx: Binding site index
            search_radius: Search radius (Å)
            
        Returns:
            List of (residue_index, distance) tuples
        """
        site_coord = coords[site_idx]
        neighbors = []
        
        for i, coord in enumerate(coords):
            if i == site_idx:
                continue
            
            dist = np.linalg.norm(coord - site_coord)
            if dist < search_radius:
                neighbors.append((i, dist))
        
        # Sort by distance and take closest (up to coordination number)
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:self.COORDINATION_NUMBER]
    
    def _estimate_metal_position(
        self,
        coords: np.ndarray,
        site_idx: int,
        neighbors: List[Tuple[int, float]],
    ) -> np.ndarray:
        """
        Estimate metal ion position based on coordinating residues.
        
        The metal is typically positioned such that ligands are arranged
        roughly octahedrally around it.
        
        Args:
            coords: Coordinates
            site_idx: Primary binding site
            neighbors: Neighboring residues
            
        Returns:
            Estimated metal position
        """
        # Start with weighted average position
        positions = [coords[site_idx]]
        weights = [2.0]  # Weight primary site higher
        
        for neighbor_idx, dist in neighbors[:4]:  # Top 4 neighbors
            positions.append(coords[neighbor_idx])
            weights.append(1.0 / (dist + 1e-8))
        
        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        metal_pos = np.sum(positions * weights[:, np.newaxis], axis=0)
        
        # Shift slightly toward primary site
        metal_pos = 0.7 * metal_pos + 0.3 * coords[site_idx]
        
        return metal_pos
    
    def _get_ideal_distance(self, nucleotide: str) -> float:
        """
        Get ideal metal-residue distance based on nucleotide type.
        
        Purines (A, G) can coordinate through N7, so different distance.
        """
        if nucleotide in ['G', 'A']:
            # Purine: can coordinate through N7
            return self.INTERACTION_DISTANCES['N7_purine']
        else:
            # Pyrimidine or general
            return self.MG_COORD_DISTANCE
    
    def validate_coordination_geometry(
        self,
        coords: np.ndarray,
        site_idx: int,
    ) -> dict:
        """
        Validate coordination geometry around a site.
        
        Args:
            coords: Coordinates
            site_idx: Binding site index
            
        Returns:
            Validation metrics
        """
        neighbors = self._find_coordination_neighbors(coords, site_idx)
        
        if len(neighbors) < 2:
            return {
                'valid': False,
                'reason': 'Too few coordinating residues',
            }
        
        # Check distances
        distances = [d for _, d in neighbors]
        
        # Check if distances are in acceptable range
        in_range = all(
            self.MG_COORD_DISTANCE_RANGE[0] <= d <= self.MG_COORD_DISTANCE_RANGE[1] * 2
            for d in distances
        )
        
        # Check angular distribution (should be roughly octahedral)
        # Simplified: check if residues are distributed around the site
        site_coord = coords[site_idx]
        vectors = [coords[i] - site_coord for i, _ in neighbors]
        
        if len(vectors) >= 2:
            # Check angles between coordination vectors
            angles = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    v1 = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-8)
                    v2 = vectors[j] / (np.linalg.norm(vectors[j]) + 1e-8)
                    cos_angle = np.dot(v1, v2)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    angles.append(angle)
            
            # Octahedral: expect ~90° and ~180° angles
            mean_angle = np.mean(angles)
            good_geometry = 60 <= mean_angle <= 120
        else:
            good_geometry = False
        
        return {
            'valid': in_range and good_geometry,
            'num_neighbors': len(neighbors),
            'distances': distances,
            'in_distance_range': in_range,
            'good_geometry': good_geometry,
        }


def add_metal_ions_to_structure(
    coords: np.ndarray,
    binding_sites: List[int],
    sequence: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Add predicted metal ion coordinates to structure.
    
    Args:
        coords: (L, 3) RNA coordinates
        binding_sites: Predicted binding sites
        sequence: RNA sequence
        
    Returns:
        (adjusted_rna_coords, metal_coords)
        adjusted_rna_coords: Adjusted RNA coordinates
        metal_coords: List of metal ion 3D positions
    """
    geometry_adjuster = MetalSiteGeometry()
    
    # Adjust structure
    adjusted_coords = geometry_adjuster.adjust_coordinates(
        coords, binding_sites, sequence
    )
    
    # Estimate metal positions
    metal_coords = []
    for site_idx in binding_sites:
        neighbors = geometry_adjuster._find_coordination_neighbors(
            adjusted_coords, site_idx
        )
        metal_pos = geometry_adjuster._estimate_metal_position(
            adjusted_coords, site_idx, neighbors
        )
        metal_coords.append(metal_pos)
    
    return adjusted_coords, metal_coords
