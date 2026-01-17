# RNA-TBM: Template-Based Modeling for RNA 3D Structure Prediction

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance pipeline for predicting RNA 3D structures using template-based modeling with k-mer indexed search. Developed for the [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) Kaggle competition.

## Overview

This pipeline predicts C1' atom coordinates for RNA molecules by:

1. **Template Search**: Finding similar structures in PDB using k-mer indexing
2. **Sequence Alignment**: Needleman-Wunsch global alignment
3. **Coordinate Transfer**: Mapping template coordinates to target
4. **Gap Filling**: Geometric interpolation for unmapped residues
5. **Model Diversity**: Generating 5 diverse predictions per target

### Why Template-Based Modeling?

> "95% of test targets in 2025 have structural templates in PDB"

This insight from the Part 1 competition makes TBM the optimal approach, achieving higher accuracy than deep learning methods when templates are available.

## Project Structure

```
rna-tbm/
├── rna_tbm/                    # Main Python package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Configuration management
│   ├── cif_parser.py          # mmCIF file parser
│   ├── template_db.py         # k-mer indexed template database
│   ├── alignment.py           # Sequence alignment
│   ├── gap_filling.py         # Gap interpolation
│   ├── submission.py          # Submission generation
│   ├── pipeline.py            # Main pipeline orchestration
│   └── cli.py                 # Command-line interface
│
├── scripts/                    # Utility scripts
│   ├── run_validation.py      # Validation pipeline
│   └── test_components.py     # Component tests
│
├── notebooks/                  # Jupyter/Kaggle notebooks
│   └── kaggle_inference.py    # Standalone Kaggle submission
│
├── data/                       # Data directory (download separately)
│   ├── sequences/             # CSV sequence files
│   ├── pdb_rna/               # PDB CIF structure files
│   ├── msa/                   # Multiple sequence alignments
│   └── metadata/              # Release dates and metadata
│
├── output/                     # Output directory
│   ├── submission.csv         # Generated predictions
│   └── template_db.pkl        # Cached template database
│
├── docs/                       # Documentation
├── tests/                      # Unit tests
│
├── pyproject.toml             # Package configuration
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/stanford-rna-folding/rna-tbm.git
cd rna-tbm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

### Data Setup

Download competition data from Kaggle and organize:

```bash
# Create data directories
mkdir -p data/{sequences,pdb_rna,msa,metadata}

# Copy/symlink your data:
# - data/sequences/: test_sequences.csv, train_sequences.csv, validation_sequences.csv
# - data/pdb_rna/: *.cif files from PDB_RNA
# - data/metadata/: rna_metadata.csv
```

Or use the legacy structure (auto-detected):

```
├── PDB_RNA/          # CIF files
├── MSA/              # MSA files
├── extra/            # Metadata
└── *.csv             # Sequence files
```

### Basic Usage

**Using the CLI:**

```bash
# Run full pipeline
rna-tbm run --sequences data/sequences/test_sequences.csv --output submission.csv

# Build database only
rna-tbm build-db --pdb-dir data/pdb_rna --output template_db.pkl

# Predict with existing database
rna-tbm predict test_sequences.csv --database template_db.pkl --output submission.csv
```

**Using Python API:**

```python
from rna_tbm import TBMPipeline, PipelineConfig

# Create pipeline with default config
pipeline = TBMPipeline()

# Build/load template database
pipeline.load_or_build_database()

# Predict single target
prediction = pipeline.predict_single(
    target_id="TEST_1",
    sequence="ACCGUGACGGG",
    temporal_cutoff="2025-01-01"
)

# Access predictions
for model_idx, coords in enumerate(prediction.models):
    print(f"Model {model_idx}: shape {coords.shape}")
```

**Run validation:**

```bash
python scripts/run_validation.py
```

## Configuration

Configuration can be customized via `PipelineConfig`:

```python
from rna_tbm import PipelineConfig, TBMPipeline

config = PipelineConfig(
    kmer_size=6,              # k-mer index size
    max_template_hits=50,     # Max templates to consider
    min_identity=0.25,        # Minimum sequence identity
    min_coverage=0.25,        # Minimum alignment coverage
    num_models=5,             # Models per target
    perturbation_scale=0.2,   # Diversity noise scale
    gap_fill_method='geometric',  # Gap filling method
)

pipeline = TBMPipeline(config)
```

For testing with limited resources:

```python
config = PipelineConfig.for_testing()  # Limits files processed
```

## Pipeline Components

### CIF Parser

Extracts C1' atom coordinates from mmCIF files, handling:

- Modified nucleotides (50+ types mapped to A/C/G/U)
- Multi-model structures (uses model 1 only)
- Quoted atom names (e.g., `"C1'"`)

### Template Database

Fast sequence-based retrieval using:

- k-mer index (k=6) for O(1) lookup
- Temporal filtering for competition compliance
- Diagonal-band alignment for scoring

### Alignment

Needleman-Wunsch global alignment with:

- Match: +2, Mismatch: -1, Gap: -2
- Position mapping for coordinate transfer

### Gap Filling

Handles unmapped residues via:

- Linear interpolation for internal gaps
- Geometric extrapolation for terminal gaps
- A-form helix baseline when no templates found

## Expected Performance

| Target Type | Expected TM-score |
|-------------|-------------------|
| Identical template | 0.95+ |
| High homology (>80%) | 0.80-0.95 |
| Medium homology (50-80%) | 0.60-0.80 |
| Low/no homology | 0.30-0.50 |

**Weighted average: ~0.75 TM-score**

## Kaggle Submission

For Kaggle, use the self-contained notebook:

```bash
# Copy to Kaggle
cp notebooks/kaggle_inference.py /path/to/kaggle/notebook.py
```

The notebook includes all code in a single file with no external dependencies beyond numpy, pandas, and scipy.

**Runtime estimates:**

- Database build: ~2-3 hours
- Predictions: ~30 minutes
- **Total: ~3.5 hours** (well under 8-hour limit)

## Testing

```bash
# Run component tests
python scripts/test_components.py

# Run with pytest (if installed)
pytest tests/
```

## Documentation

- [Pipeline Architecture](docs/architecture.md)
- [CIF Parser Details](docs/cif_parser.md)
- [Template Database](docs/template_db.md)
- [Competition Analysis](docs/challenge_analysied.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Stanford RNA Folding competition organizers
- PDB for structural data
- Kaggle community

## Contact

For questions or issues, please open a GitHub issue.

---

*Built for the Stanford Ribonanza RNA Folding Part 2 competition*
