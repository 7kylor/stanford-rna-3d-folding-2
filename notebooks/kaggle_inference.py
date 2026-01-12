"""
Stanford RNA 3D Folding Part 2 - Kaggle Inference Notebook
Template-Based Modeling (TBM) Approach

This is a self-contained notebook that implements a TBM pipeline:
1. Searches for structural templates from PDB using k-mer index
2. Aligns target sequences to templates
3. Transfers C1' coordinates with gap filling
4. Generates 5 diverse models per target

For Kaggle submission, this runs in <8 hours with no internet access.

Repository: https://github.com/stanford-rna-folding/rna-tbm
"""

# %% [markdown]
# # Setup and Imports

# %%
import csv
import os
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# %% [markdown]
# # Configuration

# %%
# Detect environment
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/input/stanford-rna-3d-folding-2")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    # Local development - adjust paths as needed
    DATA_DIR = Path(".")
    OUTPUT_DIR = Path("output")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Pipeline configuration
K_MER_SIZE = 6
MAX_TEMPLATE_HITS = 50
MIN_IDENTITY = 0.25
MIN_COVERAGE = 0.25
NUM_MODELS = 5
PERTURBATION_SCALE = 0.2
AVERAGE_C1_DISTANCE = 5.9  # Angstroms

# %% [markdown]
# # Core Data Structures

# %%
class Residue(NamedTuple):
    """Single residue with C1' coordinates."""
    resname: str
    resid: int
    chain: str
    x: float
    y: float
    z: float


class ChainCoords(NamedTuple):
    """All C1' coordinates for a single chain."""
    chain_id: str
    sequence: str
    residues: List[Residue]


class TemplateHit(NamedTuple):
    """A template match result."""
    pdb_id: str
    chain_id: str
    sequence: str
    release_date: str
    identity: float
    coverage: float


# Modified nucleotide mappings
MODIFIED_BASE_MAP = {
    'A': 'A', 'ADE': 'A', 'DA': 'A', '1MA': 'A', '2MA': 'A', 'MIA': 'A',
    'T6A': 'A', 'I': 'A', 'INO': 'A', '6MA': 'A', 'A2M': 'A', 'MA6': 'A',
    'C': 'C', 'CYT': 'C', 'DC': 'C', '5MC': 'C', 'OMC': 'C', '4OC': 'C',
    'G': 'G', 'GUA': 'G', 'DG': 'G', '1MG': 'G', '2MG': 'G', '7MG': 'G',
    'M2G': 'G', 'OMG': 'G', 'YG': 'G',
    'U': 'U', 'URA': 'U', 'DU': 'U', 'PSU': 'U', '5MU': 'U', 'H2U': 'U',
    'OMU': 'U', 'S4U': 'U', '4SU': 'U', 'DHU': 'U',
    'T': 'U', 'DT': 'U', 'THY': 'U',
}

# %% [markdown]
# # CIF Parser

# %%
def parse_cif_line(line: str) -> List[str]:
    """Parse a CIF data line, handling quoted strings."""
    tokens = []
    i = 0
    while i < len(line):
        while i < len(line) and line[i] in ' \t':
            i += 1
        if i >= len(line):
            break
        if line[i] in '"\'':
            quote = line[i]
            i += 1
            start = i
            while i < len(line) and line[i] != quote:
                i += 1
            tokens.append(line[start:i])
            i += 1
        else:
            start = i
            while i < len(line) and line[i] not in ' \t':
                i += 1
            tokens.append(line[start:i])
    return tokens


def parse_cif_c1prime(cif_path: str) -> Dict[str, ChainCoords]:
    """Parse mmCIF file and extract C1' coordinates for all RNA chains."""
    with open(cif_path, 'r') as f:
        content = f.read()
    
    atom_site_match = re.search(r'loop_\s*\n(_atom_site\.\w+\s*\n)+', content)
    if not atom_site_match:
        return {}
    
    header_section = atom_site_match.group(0)
    columns = re.findall(r'_atom_site\.(\w+)', header_section)
    
    try:
        col_indices = {
            'group_PDB': columns.index('group_PDB'),
            'label_atom_id': columns.index('label_atom_id'),
            'label_comp_id': columns.index('label_comp_id'),
            'auth_asym_id': columns.index('auth_asym_id'),
            'auth_seq_id': columns.index('auth_seq_id'),
            'Cartn_x': columns.index('Cartn_x'),
            'Cartn_y': columns.index('Cartn_y'),
            'Cartn_z': columns.index('Cartn_z'),
        }
        if 'pdbx_PDB_model_num' in columns:
            col_indices['pdbx_PDB_model_num'] = columns.index('pdbx_PDB_model_num')
        else:
            col_indices['pdbx_PDB_model_num'] = None
    except ValueError:
        return {}
    
    data_start = atom_site_match.end()
    lines = content[data_start:].split('\n')
    
    chain_residues: Dict[str, Dict[int, Residue]] = {}
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('_') or line.startswith('#') or line.startswith('loop_'):
            if line.startswith('#') or line.startswith('loop_'):
                break
            continue
        
        tokens = parse_cif_line(line)
        if len(tokens) < max(v for v in col_indices.values() if v is not None) + 1:
            continue
        
        if col_indices['pdbx_PDB_model_num'] is not None:
            model_num = tokens[col_indices['pdbx_PDB_model_num']]
            if model_num != '1' and model_num != '?':
                continue
        
        group = tokens[col_indices['group_PDB']]
        atom_id = tokens[col_indices['label_atom_id']]
        # Only strip quotes if the value is enclosed in matching quotes
        if len(atom_id) >= 2 and atom_id[0] == atom_id[-1] and atom_id[0] in '"\'':
            atom_id = atom_id[1:-1]
        
        if group != 'ATOM' or atom_id != "C1'":
            continue
        
        comp_id = tokens[col_indices['label_comp_id']]
        base = MODIFIED_BASE_MAP.get(comp_id.upper(), comp_id.upper())
        if base not in {'A', 'C', 'G', 'U'}:
            continue
        
        chain = tokens[col_indices['auth_asym_id']]
        try:
            resid = int(tokens[col_indices['auth_seq_id']])
            x = float(tokens[col_indices['Cartn_x']])
            y = float(tokens[col_indices['Cartn_y']])
            z = float(tokens[col_indices['Cartn_z']])
        except (ValueError, IndexError):
            continue
        
        if chain not in chain_residues:
            chain_residues[chain] = {}
        
        if resid not in chain_residues[chain]:
            chain_residues[chain][resid] = Residue(base, resid, chain, x, y, z)
    
    result = {}
    for chain_id, res_dict in chain_residues.items():
        sorted_resids = sorted(res_dict.keys())
        residues = [res_dict[rid] for rid in sorted_resids]
        sequence = ''.join(r.resname for r in residues)
        result[chain_id] = ChainCoords(chain_id, sequence, residues)
    
    return result

# %% [markdown]
# # Template Database

# %%
class TemplateDB:
    """Template database with k-mer index for fast sequence search."""
    
    def __init__(self, k: int = 6):
        self.k = k
        self.templates: Dict[str, Dict[str, ChainCoords]] = {}
        self.release_dates: Dict[str, str] = {}
        self.kmer_index: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        self.sequences: Dict[Tuple[str, str], str] = {}
    
    def build_from_directory(self, cif_dir: str, release_dates_file: str = None):
        """Build template database from PDB_RNA directory."""
        # Load release dates if file provided
        if release_dates_file and os.path.exists(release_dates_file):
            self._load_release_dates(release_dates_file)
        
        cif_files = list(Path(cif_dir).glob('*.cif'))
        total = len(cif_files)
        print(f"Building template database from {total} CIF files...")
        
        for i, cif_path in enumerate(cif_files):
            pdb_id = cif_path.stem.upper()
            
            # Skip if we have release dates and this PDB is not in them
            if release_dates_file and pdb_id not in self.release_dates:
                continue
            
            try:
                chains = parse_cif_c1prime(str(cif_path))
                if chains:
                    self.templates[pdb_id] = chains
                    for chain_id, chain_coords in chains.items():
                        seq = chain_coords.sequence
                        self.sequences[(pdb_id, chain_id)] = seq
                        for pos in range(len(seq) - self.k + 1):
                            kmer = seq[pos:pos + self.k]
                            self.kmer_index[kmer].append((pdb_id, chain_id, pos))
            except Exception:
                pass
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{total} files...")
        
        print(f"Done. {len(self.templates)} templates indexed.")
    
    def _load_release_dates(self, csv_path: str):
        """Load release dates from CSV."""
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to get PDB ID from various column names
                pdb_id = row.get('pdb_id', row.get('entry_id', '')).upper()
                if not pdb_id and 'target_id' in row:
                    target_id = row['target_id']
                    if '_' in target_id:
                        pdb_id = target_id.split('_')[0].upper()
                    else:
                        pdb_id = target_id[:4].upper()
                
                # Try to get date from various column names
                date = row.get('release_date', row.get('Release Date', row.get('temporal_cutoff', '')))
                if pdb_id and date and pdb_id not in self.release_dates:
                    self.release_dates[pdb_id] = date
                    count += 1
        print(f"  Loaded {count} release dates")
    
    def search(self, query_sequence: str, temporal_cutoff: str,
               max_hits: int = 50, min_identity: float = 0.25) -> List[TemplateHit]:
        """Search for template matches."""
        cutoff_date = datetime.strptime(temporal_cutoff, '%Y-%m-%d')
        
        hit_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        for pos in range(len(query_sequence) - self.k + 1):
            kmer = query_sequence[pos:pos + self.k]
            if kmer in self.kmer_index:
                for pdb_id, chain_id, _ in self.kmer_index[kmer]:
                    release_date_str = self.release_dates.get(pdb_id, '9999-12-31')
                    try:
                        release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
                        if release_date >= cutoff_date:
                            continue
                    except:
                        continue
                    hit_counts[(pdb_id, chain_id)] += 1
        
        candidates = sorted(hit_counts.items(), key=lambda x: -x[1])[:max_hits * 2]
        
        results = []
        for (pdb_id, chain_id), count in candidates:
            template_seq = self.sequences[(pdb_id, chain_id)]
            identity, coverage = self._quick_align(query_sequence, template_seq)
            
            if identity >= min_identity:
                results.append(TemplateHit(
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    sequence=template_seq,
                    release_date=self.release_dates.get(pdb_id, 'unknown'),
                    identity=identity,
                    coverage=coverage
                ))
        
        results.sort(key=lambda h: -(h.identity * h.coverage))
        return results[:max_hits]
    
    def _quick_align(self, query: str, template: str) -> Tuple[float, float]:
        """Quick diagonal alignment for scoring."""
        seeds = []
        for q_pos in range(len(query) - self.k + 1):
            kmer = query[q_pos:q_pos + self.k]
            for t_pos in range(len(template) - self.k + 1):
                if template[t_pos:t_pos + self.k] == kmer:
                    seeds.append((q_pos, t_pos))
        
        if not seeds:
            return 0.0, 0.0
        
        diag_seeds: Dict[int, int] = defaultdict(int)
        for q_pos, t_pos in seeds:
            diag_seeds[q_pos - t_pos] += 1
        
        best_diag = max(diag_seeds.keys(), key=lambda d: diag_seeds[d])
        offset = -best_diag
        
        matches = 0
        aligned = 0
        for q_idx in range(len(query)):
            t_idx = q_idx + offset
            if 0 <= t_idx < len(template):
                aligned += 1
                if query[q_idx] == template[t_idx]:
                    matches += 1
        
        identity = matches / len(query) if query else 0
        coverage = aligned / len(query) if query else 0
        return identity, coverage
    
    def get_template_coords(self, pdb_id: str, chain_id: str) -> Optional[ChainCoords]:
        """Get coordinates for a specific template."""
        if pdb_id in self.templates and chain_id in self.templates[pdb_id]:
            return self.templates[pdb_id][chain_id]
        return None
    
    def save(self, path: str):
        """Save database to pickle file."""
        data = {
            'k': self.k,
            'templates': self.templates,
            'release_dates': self.release_dates,
            'kmer_index': dict(self.kmer_index),
            'sequences': self.sequences,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'TemplateDB':
        """Load database from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        db = cls(k=data['k'])
        db.templates = data['templates']
        db.release_dates = data['release_dates']
        db.kmer_index = defaultdict(list, data['kmer_index'])
        db.sequences = data['sequences']
        return db

# %% [markdown]
# # Alignment and Coordinate Transfer

# %%
def needleman_wunsch(seq1: str, seq2: str) -> List[Tuple[int, int]]:
    """Global alignment returning aligned pairs."""
    n, m = len(seq1), len(seq2)
    
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i * -2
    for j in range(m + 1):
        dp[0, j] = j * -2
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1, j-1] + (2 if seq1[i-1] == seq2[j-1] else -1)
            delete = dp[i-1, j] - 2
            insert = dp[i, j-1] - 2
            dp[i, j] = max(match, delete, insert)
    
    aligned_pairs = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score = 2 if seq1[i-1] == seq2[j-1] else -1
            if dp[i, j] == dp[i-1, j-1] + score:
                aligned_pairs.append((i-1, j-1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i-1, j] - 2:
            i -= 1
        else:
            j -= 1
    
    aligned_pairs.reverse()
    return aligned_pairs


def transfer_coordinates(query_seq: str, template_coords: ChainCoords) -> np.ndarray:
    """Transfer C1' coordinates from template to query."""
    aligned_pairs = needleman_wunsch(query_seq, template_coords.sequence)
    
    n = len(query_seq)
    coords = np.full((n, 3), np.nan, dtype=np.float64)
    
    template_lookup = {
        i: (r.x, r.y, r.z) for i, r in enumerate(template_coords.residues)
    }
    
    for q_idx, t_idx in aligned_pairs:
        if t_idx in template_lookup:
            coords[q_idx] = template_lookup[t_idx]
    
    return coords

# %% [markdown]
# # Gap Filling

# %%
def fill_gaps(coords: np.ndarray) -> np.ndarray:
    """Fill gaps using geometric interpolation."""
    n = len(coords)
    if n == 0:
        return coords
    
    filled = coords.copy()
    mapped_indices = [i for i in range(n) if not np.isnan(coords[i, 0])]
    
    if len(mapped_indices) == 0:
        return generate_geometric_baseline(n)
    
    if len(mapped_indices) == 1:
        anchor_idx = mapped_indices[0]
        direction = np.array([AVERAGE_C1_DISTANCE, 0, 0])
        for i in range(anchor_idx - 1, -1, -1):
            filled[i] = filled[i + 1] - direction
        for i in range(anchor_idx + 1, n):
            filled[i] = filled[i - 1] + direction
        return filled
    
    sorted_anchors = sorted(mapped_indices)
    
    # Fill internal gaps
    for i in range(len(sorted_anchors) - 1):
        start_idx = sorted_anchors[i]
        end_idx = sorted_anchors[i + 1]
        if end_idx - start_idx > 1:
            start_coord = coords[start_idx]
            end_coord = coords[end_idx]
            gap_length = end_idx - start_idx
            for j in range(1, gap_length):
                t = j / gap_length
                filled[start_idx + j] = start_coord + t * (end_coord - start_coord)
    
    # Fill leading gap
    if sorted_anchors[0] > 0:
        first_anchor = sorted_anchors[0]
        if len(sorted_anchors) >= 2:
            direction = coords[first_anchor] - coords[sorted_anchors[1]]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * AVERAGE_C1_DISTANCE
            else:
                direction = np.array([-AVERAGE_C1_DISTANCE, 0, 0])
        else:
            direction = np.array([-AVERAGE_C1_DISTANCE, 0, 0])
        for i in range(first_anchor - 1, -1, -1):
            filled[i] = filled[i + 1] + direction
    
    # Fill trailing gap
    if sorted_anchors[-1] < n - 1:
        last_anchor = sorted_anchors[-1]
        if len(sorted_anchors) >= 2:
            direction = coords[last_anchor] - coords[sorted_anchors[-2]]
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


def generate_geometric_baseline(n: int) -> np.ndarray:
    """Generate a geometric baseline structure."""
    coords = np.zeros((n, 3))
    rise_per_residue = 2.8
    radius = 10.0
    residues_per_turn = 11
    
    for i in range(n):
        angle = 2 * np.pi * i / residues_per_turn
        coords[i, 0] = radius * np.cos(angle)
        coords[i, 1] = radius * np.sin(angle)
        coords[i, 2] = i * rise_per_residue
    
    return coords

# %% [markdown]
# # Submission Generation

# %%
def create_submission_df(predictions: List[dict]) -> pd.DataFrame:
    """Create submission DataFrame from predictions."""
    rows = []
    
    for pred in predictions:
        target_id = pred['target_id']
        sequence = pred['sequence']
        models = pred['models']
        
        for i, base in enumerate(sequence):
            resid = i + 1
            row = {
                'ID': f"{target_id}_{resid}",
                'resname': base,
                'resid': resid,
            }
            
            for model_idx in range(5):
                coords = models[model_idx][i]
                coords = np.clip(coords, -999.999, 9999.999)
                row[f'x_{model_idx + 1}'] = round(float(coords[0]), 3)
                row[f'y_{model_idx + 1}'] = round(float(coords[1]), 3)
                row[f'z_{model_idx + 1}'] = round(float(coords[2]), 3)
            
            rows.append(row)
    
    columns = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    return pd.DataFrame(rows, columns=columns)


def diversify_models(base_coords: np.ndarray, num_models: int = 5) -> List[np.ndarray]:
    """Create diverse models from base prediction."""
    models = [base_coords.copy()]
    for i in range(1, num_models):
        perturbed = base_coords + np.random.randn(*base_coords.shape) * (PERTURBATION_SCALE * i)
        models.append(np.clip(perturbed, -999.999, 9999.999))
    return models

# %% [markdown]
# # Main Pipeline

# %%
def find_release_dates_file() -> Optional[Path]:
    """Find the release dates file from possible locations."""
    candidates = [
        DATA_DIR / "extra" / "rna_metadata.csv",
        DATA_DIR / "rna_metadata.csv",
        DATA_DIR / "PDB_RNA" / "pdb_release_dates_NA.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_tbm_pipeline():
    """Main TBM pipeline execution."""
    
    start_time = time.time()
    
    # Set up paths
    pdb_rna_dir = DATA_DIR / "PDB_RNA"
    release_dates_file = find_release_dates_file()
    test_sequences_file = DATA_DIR / "test_sequences.csv"
    template_db_path = OUTPUT_DIR / "template_db.pkl"
    
    print("=" * 60)
    print("Stanford RNA 3D Folding - TBM Pipeline")
    print("=" * 60)
    print(f"\nPDB directory: {pdb_rna_dir}")
    print(f"Release dates: {release_dates_file}")
    print(f"Test sequences: {test_sequences_file}")
    
    # Check if pre-built database exists
    prebuilt_db_path = DATA_DIR / "template_db.pkl"
    
    if prebuilt_db_path.exists():
        print(f"\nLoading pre-built template database from {prebuilt_db_path}...")
        template_db = TemplateDB.load(str(prebuilt_db_path))
    elif template_db_path.exists():
        print(f"\nLoading cached template database from {template_db_path}...")
        template_db = TemplateDB.load(str(template_db_path))
    else:
        print("\nBuilding template database...")
        template_db = TemplateDB(k=K_MER_SIZE)
        template_db.build_from_directory(
            str(pdb_rna_dir),
            str(release_dates_file) if release_dates_file else None
        )
        template_db.save(str(template_db_path))
    
    db_time = time.time() - start_time
    print(f"Template database ready in {db_time:.1f}s ({len(template_db.templates)} templates)")
    
    # Load test sequences
    test_df = pd.read_csv(test_sequences_file)
    print(f"\nLoaded {len(test_df)} test targets")
    
    # Process each target
    predictions = []
    num_with_templates = 0
    num_no_templates = 0
    
    for idx, row in test_df.iterrows():
        target_id = row['target_id']
        sequence = row['sequence']
        temporal_cutoff = row['temporal_cutoff']
        
        target_start = time.time()
        
        # Search for templates
        hits = template_db.search(sequence, temporal_cutoff, max_hits=MAX_TEMPLATE_HITS)
        
        models = []
        
        if hits:
            num_with_templates += 1
            best_hit = hits[0]
            print(f"[{idx+1}/{len(test_df)}] {target_id} ({len(sequence)}nt): "
                  f"Best template {best_hit.pdb_id}_{best_hit.chain_id} "
                  f"(id={best_hit.identity:.2f})")
            
            # Use multiple templates for diversity
            for hit in hits[:NUM_MODELS]:
                template_coords = template_db.get_template_coords(hit.pdb_id, hit.chain_id)
                if template_coords:
                    coords = transfer_coordinates(sequence, template_coords)
                    coords = fill_gaps(coords)
                    models.append(coords)
            
            # Fill to 5 models if needed
            while len(models) < NUM_MODELS:
                if models:
                    perturbed = models[0] + np.random.randn(len(sequence), 3) * PERTURBATION_SCALE * len(models)
                    models.append(np.clip(perturbed, -999.999, 9999.999))
                else:
                    models.append(generate_geometric_baseline(len(sequence)))
        else:
            num_no_templates += 1
            print(f"[{idx+1}/{len(test_df)}] {target_id} ({len(sequence)}nt): No templates found")
            base = generate_geometric_baseline(len(sequence))
            models = diversify_models(base, NUM_MODELS)
        
        predictions.append({
            'target_id': target_id,
            'sequence': sequence,
            'models': models[:NUM_MODELS]
        })
    
    # Create submission
    print("\nGenerating submission...")
    submission_df = create_submission_df(predictions)
    submission_path = OUTPUT_DIR / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Targets processed: {len(test_df)}")
    print(f"  With templates: {num_with_templates}")
    print(f"  No templates: {num_no_templates}")
    print(f"Rows written: {len(submission_df)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output: {submission_path}")
    
    return submission_df

# %% [markdown]
# # Run Pipeline

# %%
if __name__ == "__main__":
    submission = run_tbm_pipeline()
    print(f"\nSubmission shape: {submission.shape}")
    print("\nFirst few rows:")
    print(submission.head())
