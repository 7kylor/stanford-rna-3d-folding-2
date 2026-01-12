"""
Configuration management for RNA-TBM pipeline.
Handles paths, parameters, and environment detection.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for pyproject.toml
    current = Path(__file__).parent.parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to parent of rna_tbm package
    return Path(__file__).parent.parent


def is_kaggle() -> bool:
    """Check if running on Kaggle."""
    return os.path.exists("/kaggle")


@dataclass
class PathConfig:
    """Path configuration for the pipeline."""
    
    # Base directories
    project_root: Path = field(default_factory=get_project_root)
    base_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # Data subdirectories  
    pdb_rna_dir: Optional[Path] = None
    msa_dir: Optional[Path] = None
    metadata_dir: Optional[Path] = None
    sequences_dir: Optional[Path] = None
    
    # Specific files
    release_dates_file: Optional[Path] = None
    test_sequences_file: Optional[Path] = None
    train_sequences_file: Optional[Path] = None
    validation_sequences_file: Optional[Path] = None
    template_db_file: Optional[Path] = None
    
    @property
    def is_kaggle(self) -> bool:
        """Check if running on Kaggle."""
        return is_kaggle()
    
    def __post_init__(self):
        """Set up paths based on environment."""
        self.base_dir = self.project_root
        if is_kaggle():
            self._setup_kaggle_paths()
        else:
            self._setup_local_paths()
    
    def _setup_kaggle_paths(self):
        """Configure paths for Kaggle environment."""
        kaggle_input = Path("/kaggle/input/stanford-rna-3d-folding-2")
        kaggle_working = Path("/kaggle/working")
        
        self.data_dir = kaggle_input
        self.output_dir = kaggle_working
        
        self.pdb_rna_dir = kaggle_input / "PDB_RNA"
        self.msa_dir = kaggle_input / "MSA"
        self.metadata_dir = kaggle_input / "extra"
        self.sequences_dir = kaggle_input
        
        self.release_dates_file = self._find_release_dates_file()
        self.test_sequences_file = kaggle_input / "test_sequences.csv"
        self.train_sequences_file = kaggle_input / "train_sequences.csv"
        self.validation_sequences_file = kaggle_input / "validation_sequences.csv"
        self.template_db_file = kaggle_working / "template_db.pkl"
    
    def _setup_local_paths(self):
        """Configure paths for local development."""
        root = self.project_root
        
        self.data_dir = root / "data"
        self.output_dir = root / "output"
        
        # Check both new structure and legacy structure
        if (root / "data" / "pdb_rna").exists():
            # New structure
            self.pdb_rna_dir = root / "data" / "pdb_rna"
            self.msa_dir = root / "data" / "msa"
            self.metadata_dir = root / "data" / "metadata"
            self.sequences_dir = root / "data" / "sequences"
        else:
            # Legacy structure (for backward compatibility)
            self.pdb_rna_dir = root / "PDB_RNA"
            self.msa_dir = root / "MSA"
            self.metadata_dir = root / "extra"
            self.sequences_dir = root
        
        self.release_dates_file = self._find_release_dates_file()
        self.test_sequences_file = self._find_sequence_file("test_sequences.csv")
        self.train_sequences_file = self._find_sequence_file("train_sequences.csv")
        self.validation_sequences_file = self._find_sequence_file("validation_sequences.csv")
        self.template_db_file = self.output_dir / "template_db.pkl"
    
    def _find_release_dates_file(self) -> Optional[Path]:
        """Find the release dates file from multiple possible locations."""
        candidates = [
            self.metadata_dir / "rna_metadata.csv" if self.metadata_dir else None,
            self.project_root / "extra" / "rna_metadata.csv",
            self.project_root / "data" / "metadata" / "rna_metadata.csv",
            self.pdb_rna_dir / "pdb_release_dates_NA.csv" if self.pdb_rna_dir else None,
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None
    
    def _find_sequence_file(self, filename: str) -> Optional[Path]:
        """Find a sequence file from multiple possible locations."""
        candidates = [
            self.sequences_dir / filename if self.sequences_dir else None,
            self.project_root / filename,
            self.project_root / "data" / "sequences" / filename,
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None
    
    def find_sequences_file(self, dataset: str) -> Optional[Path]:
        """Find a sequences file by dataset name (train, validation, test)."""
        mapping = {
            'train': self.train_sequences_file,
            'validation': self.validation_sequences_file,
            'test': self.test_sequences_file,
        }
        return mapping.get(dataset)
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Configuration for the TBM pipeline."""
    
    # Paths
    paths: PathConfig = field(default_factory=PathConfig)
    
    # k-mer index parameters
    kmer_size: int = 6
    
    # Template search parameters
    max_template_hits: int = 50
    min_identity: float = 0.25
    min_coverage: float = 0.25
    
    # Template selection thresholds
    good_template_identity: float = 0.4
    good_template_coverage: float = 0.6
    
    # Database building parameters
    max_files: Optional[int] = None  # None = no limit
    max_file_size_mb: Optional[float] = None  # None = no limit
    
    # Gap filling
    gap_fill_method: str = "geometric"  # 'linear', 'cubic', 'geometric'
    
    # Model diversity
    num_models: int = 5
    perturbation_scale: float = 0.2
    
    @classmethod
    def for_kaggle(cls) -> "PipelineConfig":
        """Create configuration optimized for Kaggle."""
        return cls(
            paths=PathConfig(),
            max_files=None,
            max_file_size_mb=None,
        )
    
    @classmethod
    def for_testing(cls) -> "PipelineConfig":
        """Create configuration for quick testing."""
        return cls(
            paths=PathConfig(),
            max_files=1000,
            max_file_size_mb=5.0,
        )


# Global default configuration
_default_config: Optional[PipelineConfig] = None


def get_config() -> PipelineConfig:
    """Get the current pipeline configuration."""
    global _default_config
    if _default_config is None:
        _default_config = PipelineConfig()
    return _default_config


def set_config(config: PipelineConfig):
    """Set the global pipeline configuration."""
    global _default_config
    _default_config = config


@dataclass
class EnhancedPipelineConfig(PipelineConfig):
    """
    Enhanced pipeline configuration with deep learning features.
    
    Extends PipelineConfig with options for:
    - Phase 2A: MSA Covariation
    - Phase 2B: Structure-aware Embeddings
    - Phase 2C: GNN Torsion Refinement
    - Phase 2D: Metal Ion Prediction
    - Phase 2E: Functional Homology
    """
    
    # Phase 2A: MSA Covariation
    use_msa_covariation: bool = True
    dca_threshold: float = 0.5
    msa_pseudocount: float = 0.5
    
    # Phase 2B: Embeddings
    use_embeddings: bool = True
    rna_fm_model_path: Optional[str] = None
    embedding_cache_dir: Optional[str] = None
    kmer_weight: float = 0.4
    embedding_weight: float = 0.6
    
    # Phase 2C: Torsion Refinement
    use_torsion_refinement: bool = True
    torsion_model_path: Optional[str] = None
    refinement_iterations: int = 1
    torsion_blend_factor: float = 0.3
    
    # Phase 2D: Metal Ions
    use_metal_prediction: bool = True
    metal_model_path: Optional[str] = None
    metal_threshold: float = 0.5
    metal_adjustment_strength: float = 0.3
    
    # Phase 2E: Functional Homology
    use_functional_homology: bool = True
    rfam_cm_dir: Optional[str] = None
    functional_weight: float = 0.2
    
    def __post_init__(self):
        """Set up default paths for enhanced features."""
        # Note: PipelineConfig doesn't have __post_init__, so don't call super
        if self.paths.output_dir:
            if self.embedding_cache_dir is None:
                self.embedding_cache_dir = str(self.paths.output_dir / "embeddings")
    
    @classmethod
    def for_inference(cls) -> "EnhancedPipelineConfig":
        """Configuration optimized for inference (no training)."""
        return cls(
            use_msa_covariation=True,
            use_embeddings=True,
            use_torsion_refinement=True,
            use_metal_prediction=True,
            use_functional_homology=False,  # Slower, skip for speed
        )
    
    @classmethod
    def minimal(cls) -> "EnhancedPipelineConfig":
        """Minimal configuration (baseline TBM only)."""
        return cls(
            use_msa_covariation=False,
            use_embeddings=False,
            use_torsion_refinement=False,
            use_metal_prediction=False,
            use_functional_homology=False,
        )
