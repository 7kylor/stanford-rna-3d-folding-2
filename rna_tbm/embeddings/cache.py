"""
Embedding cache for efficient storage and retrieval.
Avoids recomputing embeddings for template sequences.
"""

import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class EmbeddingCache:
    """
    Cache for RNA sequence embeddings.
    Stores embeddings on disk to avoid recomputation.
    """
    
    def __init__(
        self,
        cache_dir: str,
        encoder: Optional['RNAFMEncoder'] = None,
    ):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
            encoder: RNAFMEncoder instance for computing new embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        
        # In-memory cache for frequently accessed embeddings
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._max_memory_items = 1000
    
    def get(
        self,
        pdb_id: str,
        chain_id: str,
        sequence: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Get embeddings from cache.
        
        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier
            sequence: Sequence (for verification)
            
        Returns:
            Cached embeddings or None if not found
        """
        cache_key = self._make_key(pdb_id, chain_id)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embeddings = np.load(cache_file)
                
                # Verify length if sequence provided
                if sequence and len(embeddings) != len(sequence):
                    return None
                
                # Add to memory cache
                self._add_to_memory_cache(cache_key, embeddings)
                
                return embeddings
            except Exception:
                return None
        
        return None
    
    def get_or_compute(
        self,
        pdb_id: str,
        chain_id: str,
        sequence: str,
    ) -> np.ndarray:
        """
        Get embeddings from cache or compute if not found.
        
        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier
            sequence: RNA sequence
            
        Returns:
            Embeddings array
        """
        # Try cache first
        cached = self.get(pdb_id, chain_id, sequence)
        if cached is not None:
            return cached
        
        # Compute new embeddings
        if self.encoder is None:
            from .rna_fm import RNAFMEncoder
            self.encoder = RNAFMEncoder()
        
        embeddings = self.encoder.encode(sequence)
        
        # Save to cache
        self.put(pdb_id, chain_id, embeddings)
        
        return embeddings
    
    def put(
        self,
        pdb_id: str,
        chain_id: str,
        embeddings: np.ndarray,
    ):
        """
        Store embeddings in cache.
        
        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier
            embeddings: Embeddings array
        """
        cache_key = self._make_key(pdb_id, chain_id)
        
        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_file, embeddings)
        
        # Add to memory cache
        self._add_to_memory_cache(cache_key, embeddings)
    
    def precompute_templates(
        self,
        template_db: 'TemplateDB',
        progress_callback=None,
    ):
        """
        Precompute embeddings for all templates in database.
        
        Args:
            template_db: Template database
            progress_callback: Optional callback(current, total, message)
        """
        if self.encoder is None:
            from .rna_fm import RNAFMEncoder
            self.encoder = RNAFMEncoder()
        
        total = len(template_db.sequences)
        computed = 0
        skipped = 0
        
        for i, ((pdb_id, chain_id), sequence) in enumerate(template_db.sequences.items()):
            # Check if already cached
            if self.get(pdb_id, chain_id) is not None:
                skipped += 1
            else:
                try:
                    embeddings = self.encoder.encode(sequence)
                    self.put(pdb_id, chain_id, embeddings)
                    computed += 1
                except Exception as e:
                    if progress_callback:
                        progress_callback(i + 1, total, f"Error: {pdb_id}_{chain_id}: {e}")
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total, f"Computed: {computed}, Skipped: {skipped}")
        
        if progress_callback:
            progress_callback(total, total, f"Done. Computed: {computed}, Skipped: {skipped}")
    
    def clear(self):
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
    
    def _make_key(self, pdb_id: str, chain_id: str) -> str:
        """Create cache key from identifiers."""
        return f"{pdb_id.upper()}_{chain_id}"
    
    def _add_to_memory_cache(self, key: str, embeddings: np.ndarray):
        """Add to memory cache with LRU-style eviction."""
        if len(self._memory_cache) >= self._max_memory_items:
            # Remove oldest item (simple approach)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = embeddings
    
    def __len__(self) -> int:
        """Number of cached embeddings on disk."""
        return len(list(self.cache_dir.glob("*.npy")))


class SequenceEmbeddingIndex:
    """
    Index for fast embedding-based sequence search.
    Uses approximate nearest neighbor search.
    """
    
    def __init__(self, cache: EmbeddingCache):
        """
        Initialize index.
        
        Args:
            cache: Embedding cache
        """
        self.cache = cache
        self._index_built = False
        self._keys: list = []
        self._embeddings: Optional[np.ndarray] = None
    
    def build_index(self, template_db: 'TemplateDB'):
        """
        Build search index from template database.
        
        Args:
            template_db: Template database
        """
        embeddings_list = []
        keys = []
        
        for (pdb_id, chain_id), sequence in template_db.sequences.items():
            emb = self.cache.get(pdb_id, chain_id)
            if emb is not None:
                # Use mean pooling for sequence-level embedding
                seq_emb = np.mean(emb, axis=0)
                embeddings_list.append(seq_emb)
                keys.append((pdb_id, chain_id))
        
        if embeddings_list:
            self._embeddings = np.stack(embeddings_list)
            self._keys = keys
            self._index_built = True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list:
        """
        Search for similar sequences.
        
        Args:
            query_embedding: Query sequence embedding (L, D) or (D,)
            top_k: Number of results
            
        Returns:
            List of (pdb_id, chain_id, similarity) tuples
        """
        if not self._index_built:
            return []
        
        # Mean pool if per-residue
        if query_embedding.ndim == 2:
            query_emb = np.mean(query_embedding, axis=0)
        else:
            query_emb = query_embedding
        
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        db_embs = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = np.dot(db_embs, query_emb)
        
        # Get top k
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            pdb_id, chain_id = self._keys[idx]
            results.append((pdb_id, chain_id, float(similarities[idx])))
        
        return results
