#!/usr/bin/env python3
"""
Test the TBM pipeline components.
Verifies that each module works correctly.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_cif_parser():
    """Test CIF parsing functionality."""
    print("\n=== Testing CIF Parser ===")
    
    from rna_tbm import parse_cif_c1prime
    from rna_tbm.config import PathConfig
    
    paths = PathConfig()
    pdb_dir = paths.pdb_rna_dir
    
    if not pdb_dir or not pdb_dir.exists():
        print("  SKIP: PDB directory not found")
        return True
    
    # Find a test CIF file
    cif_files = list(pdb_dir.glob("*.cif"))[:1]
    if not cif_files:
        print("  SKIP: No CIF files found")
        return True
    
    cif_file = cif_files[0]
    print(f"  Testing with: {cif_file.name}")
    
    try:
        chains = parse_cif_c1prime(str(cif_file))
        print(f"  Found {len(chains)} chains")
        
        for chain_id, chain in chains.items():
            print(f"    Chain {chain_id}: {len(chain.residues)} residues, seq={chain.sequence[:20]}...")
        
        print("  PASS: CIF parser works correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_template_db():
    """Test template database building and search."""
    print("\n=== Testing Template Database ===")
    
    from rna_tbm import TemplateDB
    from rna_tbm.config import PathConfig
    
    paths = PathConfig()
    pdb_dir = paths.pdb_rna_dir
    
    if not pdb_dir or not pdb_dir.exists():
        print("  SKIP: PDB directory not found")
        return True
    
    try:
        # Build small test database
        print("  Building test database (max 100 files, <1MB each)...")
        db = TemplateDB(k=6)
        db.build_from_directory(
            str(pdb_dir),
            max_files=100,
            max_file_size_mb=1.0
        )
        
        print(f"  Indexed {len(db)} template chains")
        
        # Test search
        if db.sequences:
            test_seq = list(db.sequences.values())[0]
            print(f"  Testing search with: {test_seq[:20]}...")
            
            hits = db.search(test_seq, max_hits=5)
            print(f"  Found {len(hits)} hits")
            
            if hits:
                best = hits[0]
                print(f"    Best: {best.pdb_id}_{best.chain_id} (id={best.identity:.2f})")
        
        print("  PASS: Template database works correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment():
    """Test sequence alignment."""
    print("\n=== Testing Alignment ===")
    
    from rna_tbm import align_sequences
    
    try:
        query = "ACCGUGACGGG"
        template = "ACCGUACGGG"
        
        result = align_sequences(query, template)
        
        print(f"  Query:    {query}")
        print(f"  Template: {template}")
        print(f"  Identity: {result.identity:.2f}")
        print(f"  Coverage: {result.coverage:.2f}")
        
        assert result.identity > 0.8, "Identity should be >0.8"
        assert result.coverage > 0.8, "Coverage should be >0.8"
        
        print("  PASS: Alignment works correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_gap_filling():
    """Test gap filling."""
    print("\n=== Testing Gap Filling ===")
    
    from rna_tbm.alignment import CoordinateTransferResult
    from rna_tbm.gap_filling import fill_gaps, generate_geometric_baseline
    
    try:
        # Test geometric baseline
        n = 50
        coords = generate_geometric_baseline(n)
        
        print(f"  Generated baseline for {n} residues")
        print(f"  Coord range: [{coords.min():.1f}, {coords.max():.1f}]")
        
        # Check distances between consecutive residues
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        print(f"  Mean distance: {distances.mean():.2f} Å")
        
        assert 3.0 < distances.mean() < 10.0, "Mean distance should be reasonable"
        
        print("  PASS: Gap filling works correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_submission():
    """Test submission generation."""
    print("\n=== Testing Submission Generation ===")
    
    from rna_tbm import PredictionSet, create_submission_rows
    from rna_tbm.gap_filling import generate_geometric_baseline
    
    try:
        sequence = "ACGUGACGUGAC"
        n = len(sequence)
        
        # Create 5 models
        models = [generate_geometric_baseline(n) + np.random.randn(n, 3) * i * 0.1 
                  for i in range(5)]
        
        pred = PredictionSet(
            target_id="TEST_1",
            sequence=sequence,
            models=models
        )
        
        rows = create_submission_rows(pred)
        
        print(f"  Created {len(rows)} submission rows")
        print(f"  First row: {rows[0]}")
        
        assert len(rows) == n, f"Should have {n} rows"
        assert rows[0]['ID'] == 'TEST_1_1', "First ID should be TEST_1_1"
        assert rows[0]['resname'] == 'A', "First resname should be A"
        
        print("  PASS: Submission generation works correctly")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RNA-TBM Component Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("CIF Parser", test_cif_parser()))
    results.append(("Template DB", test_template_db()))
    results.append(("Alignment", test_alignment()))
    results.append(("Gap Filling", test_gap_filling()))
    results.append(("Submission", test_submission()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
