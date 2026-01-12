"""
RNA TBM - Template-Based Modeling for RNA 3D Structure Prediction

A high-performance pipeline for predicting RNA 3D structures using
template-based modeling with k-mer indexed search.

Enhanced with:
- MSA covariation analysis (Phase 2A)
- Structure-aware embeddings (Phase 2B)
- GNN torsion refinement (Phase 2C)
- Metal ion binding prediction (Phase 2D)
- Functional homology discovery (Phase 2E)
"""

__version__ = "2.0.0"
__author__ = "Stanford RNA Folding Team"

# Core modules
from .alignment import AlignmentResult, align_sequences, transfer_coordinates
from .cif_parser import ChainCoords, Residue, parse_cif_c1prime
from .config import PathConfig, get_config, is_kaggle, set_config, EnhancedPipelineConfig
from .gap_filling import clip_coordinates, fill_gaps, generate_geometric_baseline
from .pipeline import PipelineConfig, TBMPipeline, run_pipeline
from .submission import PredictionSet, create_submission_rows, write_submission
from .template_db import TemplateDB, TemplateHit, build_template_database

# Enhancement modules (lazy imports to avoid heavy dependencies)
def get_msa_module():
    """Get MSA covariation module."""
    from . import msa
    return msa

def get_embeddings_module():
    """Get embeddings module."""
    from . import embeddings
    return embeddings

def get_refinement_module():
    """Get torsion refinement module."""
    from . import refinement
    return refinement

def get_metal_ions_module():
    """Get metal ion prediction module."""
    from . import metal_ions
    return metal_ions

def get_functional_module():
    """Get functional homology module."""
    from . import functional
    return functional

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # CIF parsing
    "Residue",
    "ChainCoords", 
    "parse_cif_c1prime",
    # Template database
    "TemplateDB",
    "TemplateHit",
    "build_template_database",
    # Alignment
    "AlignmentResult",
    "align_sequences",
    "transfer_coordinates",
    # Gap filling
    "fill_gaps",
    "generate_geometric_baseline",
    "clip_coordinates",
    # Submission
    "PredictionSet",
    "write_submission",
    "create_submission_rows",
    # Pipeline
    "TBMPipeline",
    "PipelineConfig",
    "run_pipeline",
    # Config
    "PathConfig",
    "get_config",
    "set_config",
    "is_kaggle",
    "EnhancedPipelineConfig",
    # Enhancement module accessors
    "get_msa_module",
    "get_embeddings_module",
    "get_refinement_module",
    "get_metal_ions_module",
    "get_functional_module",
]
