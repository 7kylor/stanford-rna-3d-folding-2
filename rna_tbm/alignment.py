"""
Sequence alignment and coordinate transfer for TBM.
Maps template C1' coordinates to target residue indices.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cif_parser import ChainCoords, Residue


@dataclass
class AlignmentResult:
    """Result of sequence alignment with coordinate mapping."""
    query_seq: str
    template_seq: str
    identity: float
    coverage: float
    # Maps query position (0-based) -> template position (0-based), or -1 if gap
    query_to_template: List[int]
    # Maps template position (0-based) -> query position (0-based), or -1 if gap  
    template_to_query: List[int]


def needleman_wunsch(
    seq1: str,
    seq2: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2
) -> Tuple[str, str, List[Tuple[int, int]]]:
    """
    Needleman-Wunsch global alignment.
    
    Returns:
        (aligned_seq1, aligned_seq2, aligned_pairs)
        where aligned_pairs is list of (seq1_idx, seq2_idx) for matched positions
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize DP matrix
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i * gap_penalty
    for j in range(m + 1):
        dp[0, j] = j * gap_penalty
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = dp[i-1, j] + gap_penalty
            insert = dp[i, j-1] + gap_penalty
            dp[i, j] = max(match, delete, insert)
    
    # Traceback
    aligned1, aligned2 = [], []
    aligned_pairs = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = match_score if seq1[i-1] == seq2[j-1] else mismatch_score
            if dp[i, j] == dp[i-1, j-1] + score:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                aligned_pairs.append((i-1, j-1))
                i -= 1
                j -= 1
                continue
        
        if i > 0 and dp[i, j] == dp[i-1, j] + gap_penalty:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1
    
    aligned1.reverse()
    aligned2.reverse()
    aligned_pairs.reverse()
    
    return ''.join(aligned1), ''.join(aligned2), aligned_pairs


def align_sequences(query_seq: str, template_seq: str) -> AlignmentResult:
    """
    Align query sequence to template sequence.
    
    Args:
        query_seq: Target sequence to predict
        template_seq: Template sequence with known structure
        
    Returns:
        AlignmentResult with position mappings
    """
    aligned_q, aligned_t, pairs = needleman_wunsch(query_seq, template_seq)
    
    # Build position mappings
    query_to_template = [-1] * len(query_seq)
    template_to_query = [-1] * len(template_seq)
    
    matches = 0
    for q_idx, t_idx in pairs:
        query_to_template[q_idx] = t_idx
        template_to_query[t_idx] = q_idx
        if query_seq[q_idx] == template_seq[t_idx]:
            matches += 1
    
    # Calculate metrics
    aligned_count = len(pairs)
    identity = matches / len(query_seq) if query_seq else 0
    coverage = aligned_count / len(query_seq) if query_seq else 0
    
    return AlignmentResult(
        query_seq=query_seq,
        template_seq=template_seq,
        identity=identity,
        coverage=coverage,
        query_to_template=query_to_template,
        template_to_query=template_to_query
    )


@dataclass
class CoordinateTransferResult:
    """Result of coordinate transfer from template to target."""
    # Per-residue data (1-based residue IDs)
    resids: List[int]
    resnames: List[str]
    coords: np.ndarray  # Shape (N, 3)
    # Which residues came from template vs gap-fill
    from_template: List[bool]
    # Quality metrics
    mapped_fraction: float
    template_pdb_id: str
    template_chain_id: str


def transfer_coordinates(
    query_seq: str,
    template_coords: ChainCoords,
    alignment: AlignmentResult
) -> CoordinateTransferResult:
    """
    Transfer C1' coordinates from template to target based on alignment.
    
    Residues aligned to template get template coordinates.
    Gaps are marked for later interpolation.
    
    Args:
        query_seq: Target sequence
        template_coords: Template ChainCoords with C1' positions
        alignment: AlignmentResult from align_sequences
        
    Returns:
        CoordinateTransferResult with transferred coordinates
    """
    n = len(query_seq)
    coords = np.full((n, 3), np.nan, dtype=np.float64)
    from_template = [False] * n
    
    # Build template residue lookup: index -> (x, y, z)
    template_lookup: Dict[int, Tuple[float, float, float]] = {}
    for i, residue in enumerate(template_coords.residues):
        template_lookup[i] = (residue.x, residue.y, residue.z)
    
    # Transfer coordinates based on alignment
    mapped_count = 0
    for q_idx, t_idx in enumerate(alignment.query_to_template):
        if t_idx >= 0 and t_idx in template_lookup:
            x, y, z = template_lookup[t_idx]
            coords[q_idx] = [x, y, z]
            from_template[q_idx] = True
            mapped_count += 1
    
    return CoordinateTransferResult(
        resids=list(range(1, n + 1)),  # 1-based
        resnames=list(query_seq),
        coords=coords,
        from_template=from_template,
        mapped_fraction=mapped_count / n if n > 0 else 0,
        template_pdb_id=template_coords.chain_id,
        template_chain_id=template_coords.chain_id
    )


def transfer_from_multiple_templates(
    query_seq: str,
    template_hits: List[Tuple[ChainCoords, AlignmentResult]],
    min_coverage_per_residue: int = 1
) -> CoordinateTransferResult:
    """
    Transfer coordinates from multiple templates, using best match per residue.
    
    Useful when no single template covers the entire query.
    
    Args:
        query_seq: Target sequence
        template_hits: List of (ChainCoords, AlignmentResult) pairs
        min_coverage_per_residue: Minimum hits needed per residue
        
    Returns:
        Combined CoordinateTransferResult
    """
    n = len(query_seq)
    coords = np.full((n, 3), np.nan, dtype=np.float64)
    from_template = [False] * n
    coord_counts = np.zeros(n, dtype=int)
    
    for template_coords, alignment in template_hits:
        # Build template lookup
        template_lookup = {}
        for i, residue in enumerate(template_coords.residues):
            template_lookup[i] = (residue.x, residue.y, residue.z)
        
        # Transfer where aligned
        for q_idx, t_idx in enumerate(alignment.query_to_template):
            if t_idx >= 0 and t_idx in template_lookup:
                x, y, z = template_lookup[t_idx]
                if np.isnan(coords[q_idx, 0]):
                    coords[q_idx] = [x, y, z]
                else:
                    # Average with existing
                    coords[q_idx] = (coords[q_idx] * coord_counts[q_idx] + [x, y, z]) / (coord_counts[q_idx] + 1)
                coord_counts[q_idx] += 1
                from_template[q_idx] = True
    
    mapped_count = sum(from_template)
    
    return CoordinateTransferResult(
        resids=list(range(1, n + 1)),
        resnames=list(query_seq),
        coords=coords,
        from_template=from_template,
        mapped_fraction=mapped_count / n if n > 0 else 0,
        template_pdb_id="multiple",
        template_chain_id="multiple"
    )
