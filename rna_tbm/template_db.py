"""
Template Database for RNA structure prediction.
Provides fast sequence-based template retrieval with temporal filtering.
"""
import csv
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from .cif_parser import ChainCoords, parse_cif_c1prime


class TemplateHit(NamedTuple):
    """A template match result."""
    pdb_id: str
    chain_id: str
    sequence: str
    release_date: str
    identity: float      # Sequence identity (0-1)
    coverage: float      # Fraction of query covered (0-1)
    aligned_pairs: List[Tuple[int, int]]  # (query_idx, template_idx) 0-based


class TemplateDB:
    """
    Template database with k-mer index for fast sequence search.
    Supports temporal filtering for competition compliance.
    """
    
    def __init__(self, k: int = 6):
        """
        Args:
            k: k-mer size for indexing (default 6)
        """
        self.k = k
        # pdb_id -> chain_id -> ChainCoords
        self.templates: Dict[str, Dict[str, ChainCoords]] = {}
        # pdb_id -> release_date (YYYY-MM-DD string)
        self.release_dates: Dict[str, str] = {}
        # k-mer -> list of (pdb_id, chain_id, position)
        self.kmer_index: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        # (pdb_id, chain_id) -> sequence for fast lookup
        self.sequences: Dict[Tuple[str, str], str] = {}
        
    def build_from_directory(
        self,
        cif_dir: str,
        release_dates_file: str = None,
        progress_callback=None,
        max_files: int = None,
        max_file_size_mb: float = None
    ):
        """
        Build template database from PDB_RNA directory.
        
        Args:
            cif_dir: Path to directory with .cif files
            release_dates_file: Path to rna_metadata.csv or similar (optional)
            progress_callback: Optional callback(current, total, message)
            max_files: Maximum number of files to process (for testing)
            max_file_size_mb: Skip files larger than this (MB, for testing)
        """
        # Load release dates if provided
        if release_dates_file:
            self._load_release_dates(release_dates_file)
        
        cif_files = list(Path(cif_dir).glob('*.cif'))
        
        # Filter by file size if specified
        if max_file_size_mb:
            max_bytes = max_file_size_mb * 1024 * 1024
            cif_files = [f for f in cif_files if f.stat().st_size <= max_bytes]
        
        if max_files:
            cif_files = cif_files[:max_files]
        total = len(cif_files)
        
        for i, cif_path in enumerate(cif_files):
            pdb_id = cif_path.stem.upper()
            
            # Skip if no release date and we have date filtering
            if release_dates_file and pdb_id not in self.release_dates:
                continue
            
            try:
                chains = parse_cif_c1prime(str(cif_path))
                if chains:
                    self.templates[pdb_id] = chains
                    
                    # Index each chain's sequence
                    for chain_id, chain_coords in chains.items():
                        seq = chain_coords.sequence
                        self.sequences[(pdb_id, chain_id)] = seq
                        
                        # Build k-mer index
                        for pos in range(len(seq) - self.k + 1):
                            kmer = seq[pos:pos + self.k]
                            self.kmer_index[kmer].append((pdb_id, chain_id, pos))
                            
            except Exception as e:
                if progress_callback:
                    progress_callback(i + 1, total, f"Warning: {cif_path.name}: {e}")
            
            if progress_callback and (i + 1) % 500 == 0:
                progress_callback(i + 1, total, f"Indexed {i+1}/{total} structures")
        
        if progress_callback:
            progress_callback(total, total, f"Done. {len(self.templates)} templates indexed.")
    
    def _load_release_dates(self, csv_path: str):
        """Load release dates from CSV. Supports multiple column name formats."""
        import time
        start = time.time()
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to get PDB ID from various column names (case-insensitive)
                pdb_id = (
                    row.get('pdb_id') or 
                    row.get('entry_id') or 
                    row.get('Entry ID') or  # CSV may have space in column name
                    row.get('Entry_ID') or
                    ''
                ).upper()
                
                # If target_id column exists (like "4TNA_A"), extract pdb_id from it
                if not pdb_id and 'target_id' in row:
                    target_id = row['target_id']
                    if '_' in target_id:
                        pdb_id = target_id.split('_')[0].upper()
                    else:
                        pdb_id = target_id[:4].upper()
                
                # Try to get date from various column names
                date = (
                    row.get('release_date') or 
                    row.get('Release Date') or 
                    row.get('Release_Date') or
                    row.get('temporal_cutoff') or
                    ''
                )
                if pdb_id and date and pdb_id not in self.release_dates:
                    self.release_dates[pdb_id] = date
                    count += 1
        print(f"  Loaded {count} release dates in {time.time()-start:.1f}s")
    
    def search(
        self,
        query_sequence: str,
        temporal_cutoff: str = None,
        max_hits: int = 100,
        min_identity: float = 0.3,
        min_coverage: float = 0.3
    ) -> List[TemplateHit]:
        """
        Search for template matches.
        
        Args:
            query_sequence: Target RNA sequence
            temporal_cutoff: Only use templates released before this date (YYYY-MM-DD), optional
            max_hits: Maximum number of hits to return
            min_identity: Minimum sequence identity threshold
            min_coverage: Minimum coverage threshold
            
        Returns:
            List of TemplateHit sorted by identity * coverage (descending)
        """
        cutoff_date = None
        if temporal_cutoff:
            cutoff_date = datetime.strptime(temporal_cutoff, '%Y-%m-%d')
        
        # Count k-mer hits per template
        hit_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        for pos in range(len(query_sequence) - self.k + 1):
            kmer = query_sequence[pos:pos + self.k]
            if kmer in self.kmer_index:
                for pdb_id, chain_id, _ in self.kmer_index[kmer]:
                    # Temporal filter (skip if cutoff specified and template is too recent)
                    if cutoff_date:
                        release_date_str = self.release_dates.get(pdb_id, '9999-12-31')
                        try:
                            release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
                            if release_date >= cutoff_date:
                                continue
                        except:
                            continue
                    
                    hit_counts[(pdb_id, chain_id)] += 1
        
        # Score candidates by normalized k-mer density
        # Use density = hits / min(query_len, template_len) to favor high-identity partial matches
        candidates = []
        query_kmers = len(query_sequence) - self.k + 1
        
        for (pdb_id, chain_id), count in hit_counts.items():
            # Quick filter: need reasonable k-mer overlap
            if count < query_kmers * 0.1:
                continue
            
            # Normalize by the shorter of query/template to find high-identity partial matches
            template_len = len(self.sequences.get((pdb_id, chain_id), ''))
            if template_len == 0:
                continue
            min_len = min(len(query_sequence), template_len)
            density = count / min_len  # Hits per position in overlap region
            
            candidates.append((pdb_id, chain_id, count, density))
        
        # Sort by density (normalized score), not raw count
        # This helps shorter templates with high identity rank higher
        candidates.sort(key=lambda x: -x[3])
        
        # Increase candidate pool to catch high-identity partial matches
        candidates = candidates[:max(max_hits * 5, 200)]
        
        # Do proper alignment for top candidates
        results = []
        for pdb_id, chain_id, _, _ in candidates:
            template_seq = self.sequences[(pdb_id, chain_id)]
            
            identity, coverage, aligned_pairs = self._align_sequences(
                query_sequence, template_seq
            )
            
            if identity >= min_identity and coverage >= min_coverage:
                results.append(TemplateHit(
                    pdb_id=pdb_id,
                    chain_id=chain_id,
                    sequence=template_seq,
                    release_date=self.release_dates.get(pdb_id, 'unknown'),
                    identity=identity,
                    coverage=coverage,
                    aligned_pairs=aligned_pairs
                ))
        
        # Sort by score (identity * coverage)
        results.sort(key=lambda h: -(h.identity * h.coverage))
        return results[:max_hits]
    
    def _align_sequences(
        self,
        query: str,
        template: str
    ) -> Tuple[float, float, List[Tuple[int, int]]]:
        """
        Improved diagonal-band alignment for sequence matching.
        Returns (identity, coverage, aligned_pairs).
        
        Identity = matches / aligned_region_length (in query)
        Coverage = aligned_positions / query_length
        """
        # Use a simple seed-and-extend approach
        # First find exact k-mer matches as seeds
        seeds = []
        for q_pos in range(len(query) - self.k + 1):
            kmer = query[q_pos:q_pos + self.k]
            for t_pos in range(len(template) - self.k + 1):
                if template[t_pos:t_pos + self.k] == kmer:
                    seeds.append((q_pos, t_pos))
        
        if not seeds:
            return 0.0, 0.0, []
        
        # Group seeds by diagonal
        diag_seeds: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for q_pos, t_pos in seeds:
            diag = q_pos - t_pos
            diag_seeds[diag].append((q_pos, t_pos))
        
        # Find best diagonal (most seeds)
        best_diag = max(diag_seeds.keys(), key=lambda d: len(diag_seeds[d]))
        
        # Extend alignment along best diagonal
        aligned_pairs = []
        matches = 0
        
        # Determine offset from diagonal
        offset = -best_diag
        
        # Count positions in alignment region
        aligned_region_len = 0
        
        for q_idx in range(len(query)):
            t_idx = q_idx + offset
            if 0 <= t_idx < len(template):
                aligned_region_len += 1
                if query[q_idx] == template[t_idx]:
                    aligned_pairs.append((q_idx, t_idx))
                    matches += 1
        
        # Calculate metrics
        # Identity: fraction of aligned region that matches
        identity = matches / aligned_region_len if aligned_region_len > 0 else 0
        # Coverage: fraction of query that overlaps with template
        coverage = aligned_region_len / len(query) if query else 0
        
        return identity, coverage, aligned_pairs
    
    def get_template_coords(
        self,
        pdb_id: str,
        chain_id: str
    ) -> Optional[ChainCoords]:
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
        print(f"Saved template database to {path}")
    
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
    
    @classmethod
    def load_from_training(cls, path: str) -> 'TemplateDB':
        """
        Load database from training template format (from build_training_db.py).
        
        The training template database has a different structure:
        - structures: Dict[target_id, {'sequence': str, 'coords': np.ndarray, 'length': int}]
        - kmer_index: Dict[kmer, List[target_id]]
        """
        from .cif_parser import Residue
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(k=6)
        structures = data['structures']
        
        for target_id, struct in structures.items():
            sequence = struct['sequence']
            coords = struct['coords']
            
            # Build Residue list from sequence and coordinates
            residues = []
            for i, (base, coord) in enumerate(zip(sequence, coords)):
                residues.append(Residue(
                    resname=base,
                    resid=i + 1,
                    chain='A',
                    x=float(coord[0]),
                    y=float(coord[1]),
                    z=float(coord[2]),
                ))
            
            chain_coords = ChainCoords(
                chain_id='A',
                sequence=sequence,
                residues=residues,
            )
            db.templates[target_id] = {'A': chain_coords}
            db.sequences[(target_id, 'A')] = sequence
            db.release_dates[target_id] = '2000-01-01'  # Old date to always pass temporal filter
        
        # Rebuild k-mer index
        for target_id, struct in structures.items():
            seq = struct['sequence']
            for pos in range(len(seq) - db.k + 1):
                kmer = seq[pos:pos+db.k]
                if 'N' not in kmer:
                    db.kmer_index[kmer].append((target_id, 'A', pos))
        
        return db
    
    def __len__(self):
        return sum(len(chains) for chains in self.templates.values())




def build_template_database(
    pdb_rna_dir: str,
    release_dates_file: str = None,
    output_path: str = None,
    k: int = 6,
    max_files: int = None,
    max_file_size_mb: float = None
) -> TemplateDB:
    """
    Build and optionally save template database.
    
    Args:
        pdb_rna_dir: Path to PDB_RNA directory with .cif files
        release_dates_file: Path to release dates CSV (optional)
        output_path: Where to save the database (optional)
        k: k-mer size
        max_files: Limit number of files (for testing)
        max_file_size_mb: Skip large files (for testing)
        
    Returns:
        Built TemplateDB
    """
    def progress(current, total, msg=""):
        print(f"\r[{current}/{total}] {msg}", end='', flush=True)
    
    print("Building template database...")
    db = TemplateDB(k=k)
    db.build_from_directory(
        pdb_rna_dir,
        release_dates_file,
        progress,
        max_files=max_files,
        max_file_size_mb=max_file_size_mb
    )
    print()
    
    if output_path:
        db.save(output_path)
    
    print(f"Database contains {len(db)} template chains from {len(db.templates)} structures.")
    return db
