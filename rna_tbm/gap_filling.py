"""
Gap filling for unmapped residues in TBM predictions.
Uses interpolation for internal gaps and extrapolation for terminal gaps.
"""
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .alignment import CoordinateTransferResult

# Average C1'-C1' distance in RNA (approximately 5.9 Å for adjacent residues)
AVERAGE_C1_DISTANCE = 5.9


def fill_gaps(
    transfer_result: CoordinateTransferResult,
    method: str = 'linear'
) -> CoordinateTransferResult:
    """
    Fill gaps in coordinate predictions.
    
    Args:
        transfer_result: CoordinateTransferResult with some NaN coordinates
        method: Interpolation method ('linear', 'cubic', 'geometric')
        
    Returns:
        New CoordinateTransferResult with gaps filled
    """
    coords = transfer_result.coords.copy()
    from_template = transfer_result.from_template.copy()
    n = len(coords)
    
    if n == 0:
        return transfer_result
    
    # Find mapped positions (anchors)
    mapped_indices = [i for i in range(n) if from_template[i]]
    
    if len(mapped_indices) == 0:
        # No template coverage - generate geometric baseline
        coords = generate_geometric_baseline(n)
        from_template = [False] * n
    elif len(mapped_indices) == 1:
        # Only one anchor - extend linearly in both directions
        anchor_idx = mapped_indices[0]
        anchor_coord = coords[anchor_idx]
        
        # Use a default direction (along x-axis)
        direction = np.array([AVERAGE_C1_DISTANCE, 0, 0])
        
        # Fill before anchor
        for i in range(anchor_idx - 1, -1, -1):
            coords[i] = coords[i + 1] - direction
            from_template[i] = False
        
        # Fill after anchor
        for i in range(anchor_idx + 1, n):
            coords[i] = coords[i - 1] + direction
            from_template[i] = False
    else:
        # Multiple anchors - interpolate/extrapolate
        if method == 'geometric':
            coords = _fill_geometric(coords, mapped_indices)
        else:
            coords = _fill_interpolation(coords, mapped_indices, method)
        
        # Update from_template flags for filled positions
        for i in range(n):
            if not from_template[i]:
                from_template[i] = False
    
    return CoordinateTransferResult(
        resids=transfer_result.resids,
        resnames=transfer_result.resnames,
        coords=coords,
        from_template=from_template,
        mapped_fraction=transfer_result.mapped_fraction,
        template_pdb_id=transfer_result.template_pdb_id,
        template_chain_id=transfer_result.template_chain_id
    )


def _fill_interpolation(
    coords: np.ndarray,
    mapped_indices: List[int],
    method: str = 'linear'
) -> np.ndarray:
    """Fill gaps using scipy interpolation."""
    n = len(coords)
    filled = coords.copy()
    
    # Get anchor points
    anchor_x = np.array(mapped_indices)
    anchor_coords = coords[mapped_indices]
    
    # Interpolate each coordinate dimension
    for dim in range(3):
        anchor_y = anchor_coords[:, dim]
        
        # Create interpolator
        kind = 'linear' if method == 'linear' else 'cubic'
        if len(anchor_x) < 4 and kind == 'cubic':
            kind = 'linear'
        
        # For positions within the anchor range
        min_anchor, max_anchor = anchor_x[0], anchor_x[-1]
        
        if max_anchor > min_anchor:
            interp_func = interp1d(anchor_x, anchor_y, kind=kind, 
                                    fill_value='extrapolate')
            
            for i in range(n):
                if np.isnan(filled[i, dim]):
                    filled[i, dim] = interp_func(i)
        else:
            # All anchors at same position (shouldn't happen)
            for i in range(n):
                if np.isnan(filled[i, dim]):
                    filled[i, dim] = anchor_y[0]
    
    # Handle extrapolation for terminal gaps more carefully
    filled = _refine_terminal_extrapolation(filled, mapped_indices)
    
    return filled


def _fill_geometric(
    coords: np.ndarray,
    mapped_indices: List[int]
) -> np.ndarray:
    """
    Fill gaps using geometric constraints.
    Maintains ~5.9 Å distance between adjacent residues.
    """
    n = len(coords)
    filled = coords.copy()
    
    # Sort mapped indices
    sorted_anchors = sorted(mapped_indices)
    
    # Fill internal gaps
    for i in range(len(sorted_anchors) - 1):
        start_idx = sorted_anchors[i]
        end_idx = sorted_anchors[i + 1]
        
        if end_idx - start_idx > 1:
            # Linear interpolation in 3D space
            start_coord = coords[start_idx]
            end_coord = coords[end_idx]
            gap_length = end_idx - start_idx
            
            for j in range(1, gap_length):
                t = j / gap_length
                filled[start_idx + j] = start_coord + t * (end_coord - start_coord)
    
    # Fill leading gap (before first anchor)
    if sorted_anchors[0] > 0:
        first_anchor = sorted_anchors[0]
        
        # Get direction from first two anchors, or use default
        if len(sorted_anchors) >= 2:
            second_anchor = sorted_anchors[1]
            direction = coords[first_anchor] - coords[second_anchor]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * AVERAGE_C1_DISTANCE
            else:
                direction = np.array([-AVERAGE_C1_DISTANCE, 0, 0])
        else:
            direction = np.array([-AVERAGE_C1_DISTANCE, 0, 0])
        
        for i in range(first_anchor - 1, -1, -1):
            filled[i] = filled[i + 1] + direction
    
    # Fill trailing gap (after last anchor)
    if sorted_anchors[-1] < n - 1:
        last_anchor = sorted_anchors[-1]
        
        # Get direction from last two anchors, or use default
        if len(sorted_anchors) >= 2:
            second_last = sorted_anchors[-2]
            direction = coords[last_anchor] - coords[second_last]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * AVERAGE_C1_DISTANCE
            else:
                direction = np.array([AVERAGE_C1_DISTANCE, 0, 0])
        else:
            direction = np.array([AVERAGE_C1_DISTANCE, 0, 0])
        
        for i in range(last_anchor + 1, n):
            filled[i] = filled[i - 1] + direction
    
    return filled


def _refine_terminal_extrapolation(
    coords: np.ndarray,
    mapped_indices: List[int]
) -> np.ndarray:
    """
    Refine terminal extrapolation to maintain reasonable distances.
    """
    n = len(coords)
    sorted_anchors = sorted(mapped_indices)
    
    # Check and fix leading gap distances
    first_anchor = sorted_anchors[0]
    for i in range(first_anchor - 1, -1, -1):
        dist = np.linalg.norm(coords[i + 1] - coords[i])
        if dist > 2 * AVERAGE_C1_DISTANCE or dist < 0.5 * AVERAGE_C1_DISTANCE:
            # Distance is unreasonable, use geometric approach
            direction = coords[i + 1] - coords[min(i + 2, first_anchor)]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction) * AVERAGE_C1_DISTANCE
            else:
                direction = np.array([-AVERAGE_C1_DISTANCE, 0, 0])
            coords[i] = coords[i + 1] + direction
    
    # Check and fix trailing gap distances
    last_anchor = sorted_anchors[-1]
    for i in range(last_anchor + 1, n):
        dist = np.linalg.norm(coords[i] - coords[i - 1])
        if dist > 2 * AVERAGE_C1_DISTANCE or dist < 0.5 * AVERAGE_C1_DISTANCE:
            direction = coords[i - 1] - coords[max(i - 2, last_anchor)]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction) * AVERAGE_C1_DISTANCE
            else:
                direction = np.array([AVERAGE_C1_DISTANCE, 0, 0])
            coords[i] = coords[i - 1] + direction
    
    return coords


def generate_geometric_baseline(n: int) -> np.ndarray:
    """
    Generate a geometric baseline structure when no template is available.
    Creates a roughly helical arrangement.
    
    Args:
        n: Number of residues
        
    Returns:
        Coordinates array of shape (n, 3)
    """
    coords = np.zeros((n, 3))
    
    # Parameters for a loose helix (A-form RNA)
    rise_per_residue = 2.8  # Å
    radius = 10.0  # Å
    residues_per_turn = 11  # Approximate for A-form RNA
    
    for i in range(n):
        angle = 2 * np.pi * i / residues_per_turn
        coords[i, 0] = radius * np.cos(angle)
        coords[i, 1] = radius * np.sin(angle)
        coords[i, 2] = i * rise_per_residue
    
    return coords


def validate_coordinates(coords: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate that coordinates are reasonable.
    
    Returns:
        (is_valid, list of warning messages)
    """
    warnings = []
    
    # Check for NaN/Inf
    if np.any(np.isnan(coords)):
        warnings.append("Contains NaN values")
    if np.any(np.isinf(coords)):
        warnings.append("Contains Inf values")
    
    # Check sequential distances
    for i in range(len(coords) - 1):
        dist = np.linalg.norm(coords[i + 1] - coords[i])
        if dist > 20:
            warnings.append(f"Large gap between residues {i+1} and {i+2}: {dist:.1f} Å")
        if dist < 2:
            warnings.append(f"Residues {i+1} and {i+2} too close: {dist:.1f} Å")
    
    # Check coordinate bounds (for PDB format compatibility)
    if np.any(coords < -999.999) or np.any(coords > 9999.999):
        warnings.append("Coordinates outside PDB format bounds")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def clip_coordinates(coords: np.ndarray) -> np.ndarray:
    """Clip coordinates to valid PDB format range."""
    return np.clip(coords, -999.999, 9999.999)
