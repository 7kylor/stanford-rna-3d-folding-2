"""
Rfam covariance model matcher.
Matches RNA sequences to Rfam families for functional annotation.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Set
import re


@dataclass
class RfamHit:
    """Result from Rfam search."""
    family_id: str        # e.g., "RF00001"
    family_name: str      # e.g., "5S_rRNA"
    score: float          # Bit score
    e_value: float        # E-value
    start: int            # Start in query (1-indexed)
    end: int              # End in query (1-indexed)
    strand: str           # '+' or '-'
    description: str      # Family description


class RfamMatcher:
    """
    Match sequences to Rfam families using covariance models.
    
    Uses Infernal's cmscan for searching against Rfam database.
    Falls back to sequence-based matching if Infernal unavailable.
    """
    
    # Common RNA families and their characteristics
    KNOWN_FAMILIES = {
        'RF00001': {'name': '5S_rRNA', 'type': 'rRNA', 'typical_length': (100, 130)},
        'RF00002': {'name': '5_8S_rRNA', 'type': 'rRNA', 'typical_length': (150, 170)},
        'RF00005': {'name': 'tRNA', 'type': 'tRNA', 'typical_length': (70, 95)},
        'RF00010': {'name': 'RNase_P_RNA', 'type': 'ribozyme', 'typical_length': (300, 400)},
        'RF00012': {'name': 'U3', 'type': 'snoRNA', 'typical_length': (200, 250)},
        'RF00017': {'name': 'SRP_euk_arch', 'type': 'SRP', 'typical_length': (250, 350)},
        'RF00023': {'name': 'tmRNA', 'type': 'tmRNA', 'typical_length': (300, 400)},
        'RF00029': {'name': 'Group_II_intron', 'type': 'intron', 'typical_length': (500, 3000)},
        'RF00059': {'name': 'TPP', 'type': 'riboswitch', 'typical_length': (80, 120)},
        'RF00167': {'name': 'Purine', 'type': 'riboswitch', 'typical_length': (60, 100)},
        'RF01051': {'name': 'c-di-GMP-I', 'type': 'riboswitch', 'typical_length': (80, 120)},
    }
    
    # Sequence motifs for common families
    FAMILY_MOTIFS = {
        'tRNA': [
            r'CCA$',  # 3' CCA acceptor
            r'GG.{2,4}G',  # D-loop
            r'TψC',  # T-loop (with pseudouridine)
        ],
        'riboswitch': [
            r'GNRA',  # GNRA tetraloop
        ],
    }
    
    def __init__(
        self,
        rfam_cm_path: Optional[str] = None,
        use_infernal: bool = True,
    ):
        """
        Initialize matcher.
        
        Args:
            rfam_cm_path: Path to Rfam.cm covariance model file
            use_infernal: Whether to use Infernal tools
        """
        self.rfam_cm_path = Path(rfam_cm_path) if rfam_cm_path else None
        self.use_infernal = use_infernal and self._check_infernal()
        
        # Cache for family information
        self._family_cache: Dict[str, dict] = {}
        self._pdb_family_map: Dict[str, Set[str]] = {}  # PDB ID -> family IDs
    
    def _check_infernal(self) -> bool:
        """Check if Infernal tools are available."""
        try:
            result = subprocess.run(
                ['cmscan', '-h'],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def search(
        self,
        sequence: str,
        e_value_threshold: float = 1e-5,
        max_hits: int = 10,
    ) -> List[RfamHit]:
        """
        Search sequence against Rfam.
        
        Args:
            sequence: RNA sequence
            e_value_threshold: E-value threshold for hits
            max_hits: Maximum number of hits to return
            
        Returns:
            List of Rfam hits
        """
        if self.use_infernal and self.rfam_cm_path and self.rfam_cm_path.exists():
            return self._search_with_infernal(sequence, e_value_threshold, max_hits)
        else:
            return self._search_heuristic(sequence, max_hits)
    
    def _search_with_infernal(
        self,
        sequence: str,
        e_value_threshold: float,
        max_hits: int,
    ) -> List[RfamHit]:
        """Search using Infernal cmscan."""
        hits = []
        
        # Write sequence to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(f">query\n{sequence}\n")
            query_file = f.name
        
        output_file = tempfile.mktemp(suffix='.tbl')
        
        try:
            # Run cmscan
            cmd = [
                'cmscan',
                '--tblout', output_file,
                '-E', str(e_value_threshold),
                '--noali',
                str(self.rfam_cm_path),
                query_file,
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=300)
            
            # Parse results
            if os.path.exists(output_file):
                hits = self._parse_cmscan_tblout(output_file, max_hits)
        
        except Exception as e:
            print(f"Infernal search failed: {e}")
        
        finally:
            # Cleanup
            if os.path.exists(query_file):
                os.unlink(query_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
        
        return hits
    
    def _parse_cmscan_tblout(self, tblout_path: str, max_hits: int) -> List[RfamHit]:
        """Parse cmscan tblout format."""
        hits = []
        
        with open(tblout_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 15:
                    continue
                
                try:
                    hit = RfamHit(
                        family_id=parts[1],
                        family_name=parts[0],
                        score=float(parts[14]),
                        e_value=float(parts[15]),
                        start=int(parts[7]),
                        end=int(parts[8]),
                        strand=parts[9],
                        description=' '.join(parts[17:]) if len(parts) > 17 else '',
                    )
                    hits.append(hit)
                except (ValueError, IndexError):
                    continue
        
        # Sort by score
        hits.sort(key=lambda h: -h.score)
        return hits[:max_hits]
    
    def _search_heuristic(
        self,
        sequence: str,
        max_hits: int,
    ) -> List[RfamHit]:
        """
        Heuristic-based family prediction.
        Uses sequence patterns and length.
        """
        hits = []
        seq_len = len(sequence)
        sequence = sequence.upper().replace('T', 'U')
        
        for family_id, info in self.KNOWN_FAMILIES.items():
            length_range = info['typical_length']
            
            # Check length
            length_score = 0.0
            if length_range[0] <= seq_len <= length_range[1]:
                length_score = 0.5
            elif length_range[0] * 0.8 <= seq_len <= length_range[1] * 1.2:
                length_score = 0.3
            
            # Check motifs
            motif_score = 0.0
            family_type = info['type']
            if family_type in self.FAMILY_MOTIFS:
                for motif in self.FAMILY_MOTIFS[family_type]:
                    if re.search(motif, sequence):
                        motif_score += 0.3
            
            total_score = length_score + motif_score
            
            if total_score > 0.3:
                hits.append(RfamHit(
                    family_id=family_id,
                    family_name=info['name'],
                    score=total_score * 100,  # Convert to bit-score-like
                    e_value=10 ** (-total_score * 10),
                    start=1,
                    end=seq_len,
                    strand='+',
                    description=info['type'],
                ))
        
        # Sort by score
        hits.sort(key=lambda h: -h.score)
        return hits[:max_hits]
    
    def get_family_info(self, family_id: str) -> Optional[dict]:
        """Get information about an Rfam family."""
        if family_id in self.KNOWN_FAMILIES:
            return self.KNOWN_FAMILIES[family_id]
        return self._family_cache.get(family_id)
    
    def get_family_templates(self, family_id: str) -> List[str]:
        """
        Get PDB IDs associated with an Rfam family.
        
        Args:
            family_id: Rfam family ID (e.g., "RF00005")
            
        Returns:
            List of PDB IDs
        """
        # This would typically query Rfam database or a local mapping
        # For now, return empty list
        return list(self._pdb_family_map.get(family_id, set()))
    
    def load_pdb_family_mapping(self, mapping_file: str):
        """
        Load PDB to Rfam family mapping.
        
        Args:
            mapping_file: Path to mapping file (PDB_ID,CHAIN,RFAM_ID format)
        """
        with open(mapping_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 3:
                    pdb_id = parts[0].upper()
                    family_id = parts[2]
                    
                    if family_id not in self._pdb_family_map:
                        self._pdb_family_map[family_id] = set()
                    self._pdb_family_map[family_id].add(pdb_id)


def download_rfam_cm(output_dir: str, version: str = "14.10") -> str:
    """
    Download Rfam covariance model file.
    
    Args:
        output_dir: Directory to save file
        version: Rfam version
        
    Returns:
        Path to downloaded file
    """
    import urllib.request
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = f"https://ftp.ebi.ac.uk/pub/databases/Rfam/{version}/Rfam.cm.gz"
    output_file = output_dir / "Rfam.cm.gz"
    
    print(f"Downloading Rfam CM from {url}...")
    urllib.request.urlretrieve(url, output_file)
    
    # Decompress
    import gzip
    import shutil
    
    cm_file = output_dir / "Rfam.cm"
    with gzip.open(output_file, 'rb') as f_in:
        with open(cm_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    output_file.unlink()  # Remove compressed file
    
    print(f"Rfam CM saved to {cm_file}")
    return str(cm_file)
