"""
mmCIF Parser for extracting C1' atom coordinates from PDB structures.
Optimized for the Stanford RNA 3D Folding competition.
"""
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple


class Residue(NamedTuple):
    """Single residue with C1' coordinates."""
    resname: str      # A, C, G, U
    resid: int        # 1-based residue number
    chain: str        # Chain ID
    x: float
    y: float
    z: float


class ChainCoords(NamedTuple):
    """All C1' coordinates for a single chain."""
    chain_id: str
    sequence: str
    residues: List[Residue]


# Common modified nucleotide mappings
MODIFIED_BASE_MAP = {
    # Adenine modifications
    'A': 'A', 'ADE': 'A', 'DA': 'A', '1MA': 'A', '2MA': 'A', 'MIA': 'A',
    'T6A': 'A', 'I': 'A', 'INO': 'A', '6MA': 'A', 'A2M': 'A', 'MA6': 'A',
    'RIA': 'A', '6IA': 'A', 'AMP': 'A', 'ATP': 'A', 'ADP': 'A',
    # Cytosine modifications  
    'C': 'C', 'CYT': 'C', 'DC': 'C', '5MC': 'C', 'OMC': 'C', '4OC': 'C',
    'CMP': 'C', 'CTP': 'C', 'CDP': 'C', 'M5M': 'C', 'S4C': 'C',
    # Guanine modifications
    'G': 'G', 'GUA': 'G', 'DG': 'G', '1MG': 'G', '2MG': 'G', '7MG': 'G',
    'M2G': 'G', 'OMG': 'G', 'YG': 'G', 'GMP': 'G', 'GTP': 'G', 'GDP': 'G',
    'G7M': 'G', 'QUO': 'G',
    # Uracil modifications
    'U': 'U', 'URA': 'U', 'DU': 'U', 'PSU': 'U', '5MU': 'U', 'H2U': 'U',
    'OMU': 'U', 'S4U': 'U', '4SU': 'U', 'DHU': 'U', 'UMP': 'U', 'UTP': 'U',
    'UDP': 'U', 'UR3': 'U', 'SSU': 'U', 'MNU': 'U',
    # Thymine (treat as U for RNA context)
    'T': 'U', 'DT': 'U', 'THY': 'U',
}


def parse_cif_c1prime(cif_path: str) -> Dict[str, ChainCoords]:
    """
    Parse mmCIF file and extract C1' coordinates for all RNA chains.
    
    Args:
        cif_path: Path to .cif file
        
    Returns:
        Dict mapping auth_chain_id -> ChainCoords
    """
    with open(cif_path, 'r') as f:
        content = f.read()
    
    # Find atom_site loop - capture the header section
    atom_site_match = re.search(r'loop_\s*\n((_atom_site\.\w+\s*\n)+)', content)
    if not atom_site_match:
        return {}
    
    # Parse column headers from captured group
    header_section = atom_site_match.group(1)
    columns = re.findall(r'_atom_site\.(\w+)', header_section)
    
    # Find column indices we need
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
    
    # Find data section (after the loop headers)
    data_start = atom_site_match.end()
    lines = content[data_start:].split('\n')
    
    # Collect C1' atoms by chain, model 1 only
    chain_residues: Dict[str, Dict[int, Residue]] = {}
    
    # Calculate minimum required tokens (exclude None values)
    min_tokens = max(v for v in col_indices.values() if v is not None) + 1
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('_') or line.startswith('#') or line.startswith('loop_'):
            if line.startswith('#') or line.startswith('loop_'):
                break
            continue
        
        # Parse the line - handle quoted strings like "C1'"
        tokens = _parse_cif_line(line)
        if len(tokens) < min_tokens:
            continue
        
        # Check if this is model 1 (or if no model number)
        if col_indices.get('pdbx_PDB_model_num') is not None:
            model_num = tokens[col_indices['pdbx_PDB_model_num']]
            if model_num != '1' and model_num != '?':
                continue
        
        # Check if ATOM record and C1' atom
        group = tokens[col_indices['group_PDB']]
        atom_id = tokens[col_indices['label_atom_id']]
        # Only strip quotes if the value is enclosed in matching quotes
        if len(atom_id) >= 2 and atom_id[0] == atom_id[-1] and atom_id[0] in '"\'':
            atom_id = atom_id[1:-1]
        
        if group != 'ATOM' or atom_id != "C1'":
            continue
        
        comp_id = tokens[col_indices['label_comp_id']]
        
        # Map modified bases to canonical
        base = _map_to_canonical(comp_id)
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
        
        # Only store first occurrence (model 1)
        if resid not in chain_residues[chain]:
            chain_residues[chain][resid] = Residue(
                resname=base,
                resid=resid,
                chain=chain,
                x=x, y=y, z=z
            )
    
    # Convert to ChainCoords
    result = {}
    for chain_id, res_dict in chain_residues.items():
        sorted_resids = sorted(res_dict.keys())
        residues = [res_dict[rid] for rid in sorted_resids]
        sequence = ''.join(r.resname for r in residues)
        result[chain_id] = ChainCoords(
            chain_id=chain_id,
            sequence=sequence,
            residues=residues
        )
    
    return result


def _parse_cif_line(line: str) -> List[str]:
    """Parse a CIF data line, handling quoted strings."""
    tokens = []
    i = 0
    while i < len(line):
        # Skip whitespace
        while i < len(line) and line[i] in ' \t':
            i += 1
        if i >= len(line):
            break
        
        # Check for quoted string
        if line[i] in '"\'':
            quote = line[i]
            i += 1
            start = i
            while i < len(line) and line[i] != quote:
                i += 1
            tokens.append(line[start:i])
            i += 1  # skip closing quote
        else:
            # Regular token
            start = i
            while i < len(line) and line[i] not in ' \t':
                i += 1
            tokens.append(line[start:i])
    
    return tokens


def _map_to_canonical(comp_id: str) -> str:
    """Map a nucleotide component ID to canonical A/C/G/U."""
    return MODIFIED_BASE_MAP.get(comp_id.upper(), comp_id.upper())


def extract_sequence_from_cif(cif_path: str) -> Dict[str, str]:
    """
    Extract RNA sequences from a CIF file.
    
    Returns:
        Dict mapping chain_id -> sequence
    """
    chains = parse_cif_c1prime(cif_path)
    return {chain_id: coords.sequence for chain_id, coords in chains.items()}


def get_c1_coords_as_array(chain_coords: ChainCoords) -> List[Tuple[float, float, float]]:
    """Convert ChainCoords to list of (x, y, z) tuples."""
    return [(r.x, r.y, r.z) for r in chain_coords.residues]


def parse_all_cifs_in_directory(
    cif_dir: str,
    progress_callback=None
) -> Dict[str, Dict[str, ChainCoords]]:
    """
    Parse all CIF files in a directory.
    
    Args:
        cif_dir: Path to directory containing .cif files
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        Dict mapping pdb_id -> {chain_id -> ChainCoords}
    """
    cif_files = list(Path(cif_dir).glob('*.cif'))
    total = len(cif_files)
    
    result = {}
    for i, cif_path in enumerate(cif_files):
        pdb_id = cif_path.stem.upper()
        try:
            chains = parse_cif_c1prime(str(cif_path))
            if chains:
                result[pdb_id] = chains
        except Exception as e:
            print(f"Warning: Failed to parse {cif_path}: {e}")
        
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1, total)
    
    return result
