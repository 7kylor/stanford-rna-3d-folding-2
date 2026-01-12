"""
Embedding-based similarity computation.
Provides methods for comparing RNA sequences using embeddings.
"""

import numpy as np
from typing import List, Tuple, Optional


class EmbeddingSimilarity:
    """
    Compute structural similarity using RNA embeddings.
    """
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Works with both sequence-level (1D) and per-residue (2D) embeddings.
        For 2D embeddings, uses mean pooling.
        
        Args:
            emb1: First embedding (L1, D) or (D,)
            emb2: Second embedding (L2, D) or (D,)
            
        Returns:
            Cosine similarity in [0, 1]
        """
        # Mean pool if per-residue
        if emb1.ndim == 2:
            emb1 = np.mean(emb1, axis=0)
        if emb2.ndim == 2:
            emb2 = np.mean(emb2, axis=0)
        
        # Normalize
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        
        # Ensure in valid range
        return float(np.clip(similarity, -1.0, 1.0))
    
    @staticmethod
    def alignment_score(
        query_emb: np.ndarray,
        template_emb: np.ndarray,
        gap_penalty: float = -0.5,
    ) -> float:
        """
        Compute alignment-based similarity using dynamic programming.
        Similar to Needleman-Wunsch but in embedding space.
        
        Args:
            query_emb: Query embeddings (L1, D)
            template_emb: Template embeddings (L2, D)
            gap_penalty: Penalty for gaps
            
        Returns:
            Alignment score normalized to [0, 1]
        """
        L1, D1 = query_emb.shape
        L2, D2 = template_emb.shape
        
        # Normalize embeddings
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        template_norm = template_emb / (np.linalg.norm(template_emb, axis=1, keepdims=True) + 1e-8)
        
        # Compute pairwise similarity matrix
        sim_matrix = np.dot(query_norm, template_norm.T)  # (L1, L2)
        
        # Dynamic programming
        dp = np.zeros((L1 + 1, L2 + 1))
        
        # Initialize with gap penalties
        for i in range(1, L1 + 1):
            dp[i, 0] = dp[i-1, 0] + gap_penalty
        for j in range(1, L2 + 1):
            dp[0, j] = dp[0, j-1] + gap_penalty
        
        # Fill DP table
        for i in range(1, L1 + 1):
            for j in range(1, L2 + 1):
                match_score = dp[i-1, j-1] + sim_matrix[i-1, j-1]
                gap_query = dp[i-1, j] + gap_penalty
                gap_template = dp[i, j-1] + gap_penalty
                dp[i, j] = max(match_score, gap_query, gap_template)
        
        # Normalize score
        max_possible = min(L1, L2)  # Best case: all positions match with score 1
        raw_score = dp[L1, L2]
        
        # Transform to [0, 1] range
        normalized = (raw_score / max_possible + 1) / 2 if max_possible > 0 else 0.0
        
        return float(np.clip(normalized, 0.0, 1.0))
    
    @staticmethod
    def local_similarity(
        query_emb: np.ndarray,
        template_emb: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Compute local similarity using sliding window.
        
        Args:
            query_emb: Query embeddings (L1, D)
            template_emb: Template embeddings (L2, D)
            window_size: Window size for local comparison
            
        Returns:
            Similarity matrix (L1, L2)
        """
        L1, D = query_emb.shape
        L2, _ = template_emb.shape
        
        # Pad embeddings
        pad = window_size // 2
        query_padded = np.pad(query_emb, ((pad, pad), (0, 0)), mode='edge')
        template_padded = np.pad(template_emb, ((pad, pad), (0, 0)), mode='edge')
        
        # Compute local embeddings (mean over window)
        query_local = np.zeros((L1, D))
        template_local = np.zeros((L2, D))
        
        for i in range(L1):
            query_local[i] = np.mean(query_padded[i:i+window_size], axis=0)
        for j in range(L2):
            template_local[j] = np.mean(template_padded[j:j+window_size], axis=0)
        
        # Normalize
        query_local = query_local / (np.linalg.norm(query_local, axis=1, keepdims=True) + 1e-8)
        template_local = template_local / (np.linalg.norm(template_local, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        return np.dot(query_local, template_local.T)
    
    @staticmethod
    def find_alignable_regions(
        similarity_matrix: np.ndarray,
        threshold: float = 0.7,
        min_length: int = 10,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find alignable regions from similarity matrix.
        
        Args:
            similarity_matrix: (L1, L2) similarity matrix
            threshold: Minimum similarity threshold
            min_length: Minimum region length
            
        Returns:
            List of (query_start, query_end, template_start, template_end)
        """
        L1, L2 = similarity_matrix.shape
        regions = []
        
        # Find diagonal regions with high similarity
        for offset in range(-L1 + min_length, L2 - min_length + 1):
            # Extract diagonal
            if offset >= 0:
                diag = np.array([
                    similarity_matrix[i, i + offset]
                    for i in range(min(L1, L2 - offset))
                ])
                start_i, start_j = 0, offset
            else:
                diag = np.array([
                    similarity_matrix[i - offset, i]
                    for i in range(min(L2, L1 + offset))
                ])
                start_i, start_j = -offset, 0
            
            # Find runs above threshold
            above_threshold = diag > threshold
            
            # Find contiguous runs
            run_start = None
            for k, is_above in enumerate(above_threshold):
                if is_above and run_start is None:
                    run_start = k
                elif not is_above and run_start is not None:
                    run_length = k - run_start
                    if run_length >= min_length:
                        q_start = start_i + run_start
                        q_end = start_i + k
                        t_start = start_j + run_start
                        t_end = start_j + k
                        regions.append((q_start, q_end, t_start, t_end))
                    run_start = None
            
            # Check last run
            if run_start is not None:
                run_length = len(above_threshold) - run_start
                if run_length >= min_length:
                    q_start = start_i + run_start
                    q_end = start_i + len(above_threshold)
                    t_start = start_j + run_start
                    t_end = start_j + len(above_threshold)
                    regions.append((q_start, q_end, t_start, t_end))
        
        return regions


def hybrid_score(
    kmer_score: float,
    embedding_score: float,
    kmer_weight: float = 0.4,
    embedding_weight: float = 0.6,
) -> float:
    """
    Combine k-mer and embedding scores.
    
    Args:
        kmer_score: K-mer based similarity
        embedding_score: Embedding-based similarity
        kmer_weight: Weight for k-mer score
        embedding_weight: Weight for embedding score
        
    Returns:
        Combined score
    """
    return kmer_weight * kmer_score + embedding_weight * embedding_score
