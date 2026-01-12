"""
Functional homology discovery module.
Provides Rfam-based matching and functional similarity scoring.
"""

from .rfam_matcher import RfamMatcher, RfamHit
from .functional_similarity import FunctionalSimilarity

__all__ = [
    'RfamMatcher',
    'RfamHit',
    'FunctionalSimilarity',
]
