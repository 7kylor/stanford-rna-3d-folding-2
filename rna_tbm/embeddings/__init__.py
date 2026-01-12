"""
Structure-aware embeddings module for RNA.
Provides RNA-FM integration, caching, and embedding-based similarity.
"""

from .rna_fm import RNAFMEncoder
from .cache import EmbeddingCache
from .similarity import EmbeddingSimilarity

__all__ = [
    'RNAFMEncoder',
    'EmbeddingCache',
    'EmbeddingSimilarity',
]
