"""
Command-line interface for RNA-TBM pipeline.
"""
import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="rna-tbm",
        description="Template-Based Modeling for RNA 3D Structure Prediction"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build database command
    build_parser = subparsers.add_parser(
        "build-db",
        help="Build template database from PDB CIF files"
    )
    build_parser.add_argument(
        "--pdb-dir",
        type=str,
        help="Directory containing PDB CIF files"
    )
    build_parser.add_argument(
        "--metadata",
        type=str,
        help="Path to release dates CSV file"
    )
    build_parser.add_argument(
        "--output", "-o",
        type=str,
        default="template_db.pkl",
        help="Output path for database pickle file"
    )
    build_parser.add_argument(
        "--kmer-size", "-k",
        type=int,
        default=6,
        help="k-mer size for indexing (default: 6)"
    )
    build_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of CIF files to process (for testing)"
    )
    build_parser.add_argument(
        "--max-file-size",
        type=float,
        help="Maximum file size in MB to process (for testing)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict structures from sequences"
    )
    predict_parser.add_argument(
        "sequences",
        type=str,
        help="Path to sequences CSV file"
    )
    predict_parser.add_argument(
        "--output", "-o",
        type=str,
        default="submission.csv",
        help="Output path for submission CSV"
    )
    predict_parser.add_argument(
        "--database", "-d",
        type=str,
        help="Path to template database pickle file"
    )
    predict_parser.add_argument(
        "--pdb-dir",
        type=str,
        help="Directory containing PDB CIF files (if building new DB)"
    )
    predict_parser.add_argument(
        "--metadata",
        type=str,
        help="Path to release dates CSV file"
    )
    
    # Run full pipeline command
    run_parser = subparsers.add_parser(
        "run",
        help="Run full TBM pipeline"
    )
    run_parser.add_argument(
        "--sequences",
        type=str,
        help="Path to sequences CSV file"
    )
    run_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for submission CSV"
    )
    run_parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Force rebuild of template database"
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    
    args = parser.parse_args()
    
    if args.command == "version":
        from rna_tbm import __version__
        print(f"rna-tbm version {__version__}")
        return 0
    
    if args.command == "build-db":
        return build_database(args)
    
    if args.command == "predict":
        return predict(args)
    
    if args.command == "run":
        return run_full_pipeline(args)
    
    # No command specified
    parser.print_help()
    return 1


def build_database(args):
    """Build template database."""
    from rna_tbm import build_template_database
    from rna_tbm.config import PathConfig
    
    paths = PathConfig()
    
    pdb_dir = args.pdb_dir or (str(paths.pdb_rna_dir) if paths.pdb_rna_dir else None)
    metadata = args.metadata or (str(paths.release_dates_file) if paths.release_dates_file else None)
    
    if not pdb_dir:
        print("Error: PDB directory not specified and not found in default locations.")
        print("Use --pdb-dir to specify the directory containing CIF files.")
        return 1
    
    if not Path(pdb_dir).exists():
        print(f"Error: PDB directory does not exist: {pdb_dir}")
        return 1
    
    print(f"Building template database from: {pdb_dir}")
    if metadata:
        print(f"Using metadata file: {metadata}")
    
    try:
        db = build_template_database(
            pdb_rna_dir=pdb_dir,
            release_dates_file=metadata,
            output_path=args.output,
            k=args.kmer_size,
            max_files=args.max_files,
            max_file_size_mb=args.max_file_size
        )
        print(f"\nDatabase saved to: {args.output}")
        print(f"Total templates: {len(db)}")
        return 0
    except Exception as e:
        print(f"Error building database: {e}")
        return 1


def predict(args):
    """Run predictions."""
    import pandas as pd

    from rna_tbm import PipelineConfig, TBMPipeline
    from rna_tbm.config import PathConfig
    
    if not Path(args.sequences).exists():
        print(f"Error: Sequences file not found: {args.sequences}")
        return 1
    
    paths = PathConfig()
    config = PipelineConfig(paths=paths)
    
    pipeline = TBMPipeline(config)
    
    # Load or build database
    if args.database and Path(args.database).exists():
        from rna_tbm import TemplateDB
        print(f"Loading database from: {args.database}")
        pipeline.template_db = TemplateDB.load(args.database)
    else:
        print("Building template database...")
        pdb_dir = args.pdb_dir or (str(paths.pdb_rna_dir) if paths.pdb_rna_dir else None)
        metadata = args.metadata or (str(paths.release_dates_file) if paths.release_dates_file else None)
        
        if not pdb_dir:
            print("Error: No template database or PDB directory specified.")
            return 1
        
        pipeline.load_or_build_database()
    
    # Load sequences and predict
    print(f"Loading sequences from: {args.sequences}")
    sequences_df = pd.read_csv(args.sequences)
    print(f"Found {len(sequences_df)} targets")
    
    print("\nRunning predictions...")
    predictions = pipeline.predict_all(sequences_df, args.output, verbose=True)
    
    print(f"\nSubmission saved to: {args.output}")
    return 0


def run_full_pipeline(args):
    """Run the full pipeline."""
    from rna_tbm import PipelineConfig, run_pipeline
    from rna_tbm.config import PathConfig
    
    paths = PathConfig()
    config = PipelineConfig(paths=paths)
    
    sequences_file = args.sequences or (str(paths.test_sequences_file) if paths.test_sequences_file else None)
    output_path = args.output or str(paths.output_dir / "submission.csv") if paths.output_dir else "submission.csv"
    
    if not sequences_file:
        print("Error: No sequences file specified and not found in default locations.")
        return 1
    
    try:
        from rna_tbm import TBMPipeline
        pipeline = TBMPipeline(config)
        pipeline.run(
            sequences_file=sequences_file,
            output_path=output_path,
            force_rebuild_db=args.rebuild_db
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
