"""
Functional similarity computation.
Computes similarity between RNA sequences based on functional annotations.
"""

import numpy as np
from typing import List, Optional, Set, Tuple
from .rfam_matcher import RfamMatcher, RfamHit


class FunctionalSimilarity:
    """
    Compute functional similarity between RNA sequences.
    
    Uses Rfam family membership and structural features.
    """
    
    def __init__(self, rfam_matcher: Optional[RfamMatcher] = None):
        """
        Initialize similarity calculator.
        
        Args:
            rfam_matcher: RfamMatcher instance
        """
        self.rfam = rfam_matcher or RfamMatcher()
        
        # Cache for sequence family assignments
        self._family_cache: dict = {}
    
    def compute_similarity(
        self,
        query_sequence: str,
        template_sequence: str,
        use_cache: bool = True,
    ) -> float:
        """
        Compute functional similarity between two sequences.
        
        Based on:
        1. Shared Rfam family membership (Jaccard)
        2. Length similarity
        3. Compositional similarity
        
        Args:
            query_sequence: Query RNA sequence
            template_sequence: Template RNA sequence
            use_cache: Whether to cache family assignments
            
        Returns:
            Similarity score in [0, 1]
        """
        # Get family assignments
        query_families = self._get_families(query_sequence, use_cache)
        template_families = self._get_families(template_sequence, use_cache)
        
        # Family similarity (Jaccard)
        family_sim = self._jaccard_similarity(query_families, template_families)
        
        # Length similarity
        length_sim = self._length_similarity(len(query_sequence), len(template_sequence))
        
        # Composition similarity
        comp_sim = self._composition_similarity(query_sequence, template_sequence)
        
        # Weighted combination
        similarity = (
            0.5 * family_sim +   # Family membership most important
            0.2 * length_sim +   # Length moderately important
            0.3 * comp_sim       # Composition somewhat important
        )
        
        return float(similarity)
    
    def _get_families(self, sequence: str, use_cache: bool) -> Set[str]:
        """Get Rfam family assignments for sequence."""
        seq_hash = hash(sequence)
        
        if use_cache and seq_hash in self._family_cache:
            return self._family_cache[seq_hash]
        
        hits = self.rfam.search(sequence)
        families = {h.family_id for h in hits}
        
        if use_cache:
            self._family_cache[seq_hash] = families
        
        return families
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _length_similarity(self, len1: int, len2: int) -> float:
        """Compute length similarity."""
        if len1 == 0 or len2 == 0:
            return 0.0
        
        ratio = min(len1, len2) / max(len1, len2)
        return ratio
    
    def _composition_similarity(self, seq1: str, seq2: str) -> float:
        """Compute nucleotide composition similarity."""
        def get_composition(seq: str) -> dict:
            seq = seq.upper().replace('T', 'U')
            total = len(seq)
            if total == 0:
                return {'A': 0, 'C': 0, 'G': 0, 'U': 0}
            return {
                'A': seq.count('A') / total,
                'C': seq.count('C') / total,
                'G': seq.count('G') / total,
                'U': seq.count('U') / total,
            }
        
        comp1 = get_composition(seq1)
        comp2 = get_composition(seq2)
        
        # Euclidean distance in composition space
        diff = sum((comp1[nt] - comp2[nt]) ** 2 for nt in 'ACGU')
        distance = np.sqrt(diff)
        
        # Convert to similarity (max distance is √2)
        similarity = 1.0 - (distance / np.sqrt(2))
        return max(0.0, similarity)
    
    def rank_templates(
        self,
        query_sequence: str,
        template_sequences: List[Tuple[str, str, str]],  # (pdb_id, chain_id, sequence)
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Rank templates by functional similarity.
        
        Args:
            query_sequence: Query sequence
            template_sequences: List of (pdb_id, chain_id, sequence) tuples
            top_k: Number of top templates to return
            
        Returns:
            List of (pdb_id, chain_id, similarity) tuples
        """
        scores = []
        
        for pdb_id, chain_id, template_seq in template_sequences:
            sim = self.compute_similarity(query_sequence, template_seq)
            scores.append((pdb_id, chain_id, sim))
        
        # Sort by similarity
        scores.sort(key=lambda x: -x[2])
        
        return scores[:top_k]
    
    def find_functional_homologs(
        self,
        query_sequence: str,
        template_db: 'TemplateDB',
        min_similarity: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """
        Find functionally homologous templates.
        
        Args:
            query_sequence: Query sequence
            template_db: Template database
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (pdb_id, chain_id, similarity) tuples
        """
        # Get query family assignments
        query_families = self._get_families(query_sequence, use_cache=True)
        
        homologs = []
        
        for (pdb_id, chain_id), template_seq in template_db.sequences.items():
            sim = self.compute_similarity(query_sequence, template_seq)
            
            if sim >= min_similarity:
                homologs.append((pdb_id, chain_id, sim))
        
        # Sort by similarity
        homologs.sort(key=lambda x: -x[2])
        
        return homologs


class FunctionalClusterer:
    """
    Cluster RNA sequences by functional similarity.
    """
    
    def __init__(self, similarity_calculator: FunctionalSimilarity):
        self.similarity = similarity_calculator
    
    def cluster(
        self,
        sequences: List[Tuple[str, str]],  # (id, sequence)
        threshold: float = 0.5,
    ) -> List[List[str]]:
        """
        Cluster sequences by functional similarity.
        Uses single-linkage clustering.
        
        Args:
            sequences: List of (id, sequence) tuples
            threshold: Similarity threshold for clustering
            
        Returns:
            List of clusters (each cluster is list of sequence IDs)
        """
        n = len(sequences)
        
        # Build similarity matrix
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.similarity.compute_similarity(
                    sequences[i][1], 
                    sequences[j][1],
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
            sim_matrix[i, i] = 1.0
        
        # Single-linkage clustering
        clusters = [[i] for i in range(n)]
        
        while True:
            # Find most similar pair of clusters
            best_sim = 0
            best_i, best_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Max similarity between clusters (single-linkage)
                    max_sim = max(
                        sim_matrix[ci, cj]
                        for ci in clusters[i]
                        for cj in clusters[j]
                    )
                    if max_sim > best_sim:
                        best_sim = max_sim
                        best_i, best_j = i, j
            
            # Merge if above threshold
            if best_sim >= threshold and best_i >= 0:
                clusters[best_i].extend(clusters[best_j])
                clusters.pop(best_j)
            else:
                break
        
        # Convert to sequence IDs
        return [[sequences[i][0] for i in cluster] for cluster in clusters]


def compute_functional_enrichment(
    families: List[str],
    background_families: List[str],
) -> dict:
    """
    Compute functional enrichment of families.
    
    Args:
        families: Observed families
        background_families: Background set of families
        
    Returns:
        Enrichment statistics
    """
    from collections import Counter
    
    obs_counts = Counter(families)
    bg_counts = Counter(background_families)
    
    total_obs = len(families)
    total_bg = len(background_families)
    
    enrichment = {}
    
    for family, obs_count in obs_counts.items():
        bg_count = bg_counts.get(family, 0)
        
        # Expected count
        expected = (bg_count / total_bg) * total_obs if total_bg > 0 else 0
        
        # Fold enrichment
        fold = obs_count / expected if expected > 0 else float('inf')
        
        enrichment[family] = {
            'observed': obs_count,
            'expected': expected,
            'fold_enrichment': fold,
        }
    
    return enrichment
