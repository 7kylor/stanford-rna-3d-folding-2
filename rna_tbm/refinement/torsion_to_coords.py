"""
Convert torsion angles to 3D coordinates using NeRF algorithm.
Natural Extension Reference Frame (NeRF) builds backbone atom by atom.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


# Torsion angle names
TORSION_NAMES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']

# RNA backbone atoms in order
BACKBONE_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]

# Bond lengths in Angstroms (average values from RNA structures)
BOND_LENGTHS = {
    ("P", "O5'"): 1.593,
    ("O5'", "C5'"): 1.440,
    ("C5'", "C4'"): 1.508,
    ("C4'", "C3'"): 1.524,
    ("C3'", "O3'"): 1.422,
    ("O3'", "P"): 1.607,  # To next residue
    ("C1'", "N"): 1.47,   # Glycosidic bond
    ("C4'", "C1'"): 2.40,  # Approximate distance
}

# Bond angles in degrees
BOND_ANGLES = {
    ("P", "O5'", "C5'"): 120.0,
    ("O5'", "C5'", "C4'"): 111.0,
    ("C5'", "C4'", "C3'"): 115.4,
    ("C4'", "C3'", "O3'"): 110.0,
    ("C3'", "O3'", "P"): 119.0,  # To next residue
    ("O3'", "P", "O5'"): 104.0,  # Next residue
}


@dataclass
class RNAResidue:
    """Represents an RNA residue with backbone coordinates."""
    resid: int
    resname: str
    P: Optional[np.ndarray] = None
    O5_prime: Optional[np.ndarray] = None
    C5_prime: Optional[np.ndarray] = None
    C4_prime: Optional[np.ndarray] = None
    C3_prime: Optional[np.ndarray] = None
    O3_prime: Optional[np.ndarray] = None
    C1_prime: Optional[np.ndarray] = None  # Sugar C1'
    
    def get_c1_prime(self) -> np.ndarray:
        """Get or estimate C1' position."""
        if self.C1_prime is not None:
            return self.C1_prime
        
        # Estimate from C4' if available
        if self.C4_prime is not None and self.C3_prime is not None:
            # C1' is roughly opposite to C5' from C4'
            direction = self.C4_prime - self.C3_prime
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            # Perpendicular in sugar ring
            perp = np.cross(direction, [0, 0, 1])
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            return self.C4_prime + 2.4 * perp
        
        return self.C4_prime if self.C4_prime is not None else np.zeros(3)


class TorsionToCoords:
    """
    Convert RNA torsion angles to 3D coordinates.
    
    Uses the NeRF (Natural Extension Reference Frame) algorithm
    to build the backbone atom by atom from torsion angles.
    """
    
    def __init__(self):
        """Initialize converter."""
        self.bond_lengths = BOND_LENGTHS
        self.bond_angles = BOND_ANGLES
    
    def apply_torsions(
        self,
        initial_coords: np.ndarray,
        torsion_angles: np.ndarray,
        sequence: str,
    ) -> np.ndarray:
        """
        Refine coordinates using predicted torsion angles.
        
        This method adjusts C1' positions based on the torsion angles
        while trying to maintain the global structure from templates.
        
        Args:
            initial_coords: (L, 3) initial C1' coordinates from template
            torsion_angles: (L, 7) predicted torsion angles
            sequence: RNA sequence
            
        Returns:
            Refined coordinates (L, 3)
        """
        L = len(sequence)
        
        if L == 0:
            return initial_coords
        
        # Start with initial coordinates
        refined = initial_coords.copy()
        
        # Build backbone from torsions
        residues = self._build_backbone_from_torsions(torsion_angles, sequence)
        
        # Extract C1' positions from built backbone
        torsion_c1_coords = np.array([r.get_c1_prime() for r in residues])
        
        # Blend template and torsion coordinates
        # Use template for global structure, torsions for local refinement
        refined = self._blend_coordinates(
            initial_coords,
            torsion_c1_coords,
            blend_factor=0.3,  # 30% torsion, 70% template
        )
        
        return refined
    
    def _build_backbone_from_torsions(
        self,
        torsion_angles: np.ndarray,
        sequence: str,
    ) -> List[RNAResidue]:
        """
        Build backbone using NeRF algorithm from torsion angles.
        
        Args:
            torsion_angles: (L, 7) torsion angles [alpha, beta, gamma, delta, epsilon, zeta, chi]
            sequence: RNA sequence
            
        Returns:
            List of RNAResidue with backbone coordinates
        """
        L = len(sequence)
        residues = []
        
        # Initialize first residue at origin
        first_residue = RNAResidue(resid=1, resname=sequence[0])
        first_residue.P = np.array([0.0, 0.0, 0.0])
        first_residue.O5_prime = np.array([1.593, 0.0, 0.0])
        first_residue.C5_prime = np.array([2.5, 1.0, 0.0])
        first_residue.C4_prime = np.array([3.5, 0.5, 0.5])
        first_residue.C3_prime = np.array([4.5, 1.2, 0.3])
        first_residue.O3_prime = np.array([5.5, 0.8, 0.7])
        
        # Estimate C1' from delta torsion
        first_residue.C1_prime = self._estimate_c1_prime(first_residue, torsion_angles[0, 3])
        
        residues.append(first_residue)
        
        # Build remaining residues
        for i in range(1, L):
            prev = residues[-1]
            torsions = torsion_angles[i]
            
            # Build next residue using NeRF
            new_residue = self._build_next_residue(
                prev,
                i + 1,
                sequence[i],
                torsions,
                torsion_angles[i-1] if i > 0 else torsions,
            )
            residues.append(new_residue)
        
        return residues
    
    def _build_next_residue(
        self,
        prev_residue: RNAResidue,
        resid: int,
        resname: str,
        torsions: np.ndarray,
        prev_torsions: np.ndarray,
    ) -> RNAResidue:
        """
        Build the next residue using NeRF from previous residue.
        
        Args:
            prev_residue: Previous residue
            resid: New residue ID
            resname: Nucleotide name
            torsions: Torsion angles for new residue
            prev_torsions: Torsion angles for previous residue
            
        Returns:
            New RNAResidue
        """
        new_res = RNAResidue(resid=resid, resname=resname)
        
        # Use epsilon and zeta from previous, alpha from current for O3'-P bond
        epsilon = prev_torsions[4]  # epsilon
        zeta = prev_torsions[5]      # zeta
        alpha = torsions[0]          # alpha
        beta = torsions[1]           # beta
        gamma = torsions[2]          # gamma
        delta = torsions[3]          # delta
        
        # Build P from prev O3'
        new_res.P = self._nerf_atom(
            prev_residue.C3_prime,
            prev_residue.O3_prime,
            prev_residue.O3_prime + np.array([1.607, 0, 0]),  # Direction
            self.bond_lengths[("O3'", "P")],
            np.radians(self.bond_angles[("C3'", "O3'", "P")]),
            zeta,
        )
        
        # Build O5' from P
        new_res.O5_prime = self._nerf_atom(
            prev_residue.O3_prime,
            new_res.P,
            new_res.P + np.array([1, 0, 0]),
            self.bond_lengths[("P", "O5'")],
            np.radians(self.bond_angles[("O3'", "P", "O5'")]),
            alpha,
        )
        
        # Build C5' from O5'
        new_res.C5_prime = self._nerf_atom(
            new_res.P,
            new_res.O5_prime,
            new_res.O5_prime + np.array([1, 0, 0]),
            self.bond_lengths[("O5'", "C5'")],
            np.radians(self.bond_angles[("P", "O5'", "C5'")]),
            beta,
        )
        
        # Build C4' from C5'
        new_res.C4_prime = self._nerf_atom(
            new_res.O5_prime,
            new_res.C5_prime,
            new_res.C5_prime + np.array([1, 0, 0]),
            self.bond_lengths[("C5'", "C4'")],
            np.radians(self.bond_angles[("O5'", "C5'", "C4'")]),
            gamma,
        )
        
        # Build C3' from C4'
        new_res.C3_prime = self._nerf_atom(
            new_res.C5_prime,
            new_res.C4_prime,
            new_res.C4_prime + np.array([1, 0, 0]),
            self.bond_lengths[("C4'", "C3'")],
            np.radians(self.bond_angles[("C5'", "C4'", "C3'")]),
            delta,
        )
        
        # Build O3' from C3'
        new_res.O3_prime = self._nerf_atom(
            new_res.C4_prime,
            new_res.C3_prime,
            new_res.C3_prime + np.array([1, 0, 0]),
            self.bond_lengths[("C3'", "O3'")],
            np.radians(self.bond_angles[("C4'", "C3'", "O3'")]),
            torsions[4],  # epsilon
        )
        
        # Estimate C1' position
        new_res.C1_prime = self._estimate_c1_prime(new_res, delta)
        
        return new_res
    
    def _nerf_atom(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        bond_length: float,
        bond_angle: float,
        torsion: float,
    ) -> np.ndarray:
        """
        Place new atom using NeRF algorithm.
        
        Given atoms A-B-C, place D such that:
        - B-C-D bond length is bond_length
        - A-B-C-D bond angle is bond_angle
        - Dihedral A-B-C-D is torsion
        
        Args:
            a, b, c: Previous three atom positions
            bond_length: Length of C-D bond
            bond_angle: Angle B-C-D
            torsion: Dihedral angle A-B-C-D
            
        Returns:
            Position of new atom D
        """
        # Vector from B to C
        bc = c - b
        bc = bc / (np.linalg.norm(bc) + 1e-8)
        
        # Vector from A to B
        ab = b - a
        
        # Normal to ABC plane
        n = np.cross(ab, bc)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            # Collinear, use arbitrary perpendicular
            n = np.array([1, 0, 0]) if abs(bc[0]) < 0.9 else np.array([0, 1, 0])
            n = np.cross(bc, n)
        n = n / (np.linalg.norm(n) + 1e-8)
        
        # Vector in ABC plane, perpendicular to BC
        m = np.cross(n, bc)
        m = m / (np.linalg.norm(m) + 1e-8)
        
        # Build new atom position
        d = c + bond_length * (
            -bc * np.cos(bond_angle) +
            m * np.sin(bond_angle) * np.cos(torsion) +
            n * np.sin(bond_angle) * np.sin(torsion)
        )
        
        return d
    
    def _estimate_c1_prime(self, residue: RNAResidue, delta: float) -> np.ndarray:
        """
        Estimate C1' position from backbone atoms and delta torsion.
        
        C1' is part of the ribose ring, related to C4' and O4'.
        Delta torsion affects the sugar pucker which affects C1' position.
        """
        if residue.C4_prime is None or residue.C3_prime is None:
            return np.zeros(3)
        
        # C1' is approximately 2.4 Å from C4' in the sugar ring
        # Direction depends on sugar pucker (delta angle)
        
        # Vector from C3' to C4'
        c3_c4 = residue.C4_prime - residue.C3_prime
        c3_c4 = c3_c4 / (np.linalg.norm(c3_c4) + 1e-8)
        
        # Perpendicular in the approximate plane
        up = np.array([0, 0, 1])
        perp = np.cross(c3_c4, up)
        if np.linalg.norm(perp) < 1e-8:
            perp = np.array([1, 0, 0])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        
        # Rotate based on delta (sugar pucker)
        # C2'-endo vs C3'-endo affects C1' position
        pucker_angle = delta - np.radians(82)  # Deviation from ideal
        
        c1_direction = (
            perp * np.cos(pucker_angle) +
            np.cross(c3_c4, perp) * np.sin(pucker_angle)
        )
        
        c1_prime = residue.C4_prime + 2.4 * c1_direction
        
        return c1_prime
    
    def _blend_coordinates(
        self,
        template_coords: np.ndarray,
        torsion_coords: np.ndarray,
        blend_factor: float = 0.3,
    ) -> np.ndarray:
        """
        Blend template and torsion-derived coordinates.
        
        Uses template for global structure and torsion for local refinement.
        
        Args:
            template_coords: (L, 3) template coordinates
            torsion_coords: (L, 3) torsion-derived coordinates
            blend_factor: Weight for torsion coordinates (0-1)
            
        Returns:
            Blended coordinates (L, 3)
        """
        L = len(template_coords)
        
        if L != len(torsion_coords):
            return template_coords
        
        # Superpose torsion structure onto template using rotation
        torsion_aligned = self._superpose(torsion_coords, template_coords)
        
        # Blend
        blended = (1 - blend_factor) * template_coords + blend_factor * torsion_aligned
        
        return blended
    
    def _superpose(
        self,
        mobile: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Superpose mobile onto target using SVD.
        
        Args:
            mobile: Coordinates to move
            target: Reference coordinates
            
        Returns:
            Transformed mobile coordinates
        """
        # Center both
        mobile_center = np.mean(mobile, axis=0)
        target_center = np.mean(target, axis=0)
        
        mobile_centered = mobile - mobile_center
        target_centered = target - target_center
        
        # Compute optimal rotation using SVD
        H = mobile_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply transformation
        aligned = (mobile_centered @ R.T) + target_center
        
        return aligned


def extract_torsions_from_pdb(coords_dict: dict) -> np.ndarray:
    """
    Extract torsion angles from PDB coordinates.
    Used for generating training data.
    
    Args:
        coords_dict: Dictionary mapping atom names to coordinates per residue
        
    Returns:
        Torsion angles (L, 7)
    """
    # This would require full backbone coordinates
    # Placeholder for now
    raise NotImplementedError("Requires full backbone atom coordinates")
