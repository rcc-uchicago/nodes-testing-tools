#!/usr/bin/env python3
"""
Convert GPU Benchmark Results Between JSON and YAML Formats

This utility converts benchmark results files between JSON and YAML formats.

Usage:
    # JSON to YAML
    python convert_format.py results.json results.yaml
    python convert_format.py results.json  # Auto-generates results.yaml
    
    # YAML to JSON
    python convert_format.py results.yaml results.json
    python convert_format.py results.yaml  # Auto-generates results.json
    
    # Auto-detect and convert
    python convert_format.py results.json --auto
    python convert_format.py results.yaml --auto
"""

import json
import yaml
import argparse
import sys
from pathlib import Path

def detect_format(filename: str) -> str:
    """Detect file format from extension"""
    ext = filename.lower().split('.')[-1]
    if ext == 'json':
        return 'json'
    elif ext in ['yaml', 'yml']:
        return 'yaml'
    else:
        raise ValueError(f"Unknown file extension: {ext}")

def load_file(filename: str, fmt: str = None):
    """Load JSON or YAML file"""
    if fmt is None:
        fmt = detect_format(filename)
    
    with open(filename, 'r') as f:
        if fmt == 'json':
            return json.load(f)
        elif fmt == 'yaml':
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown format: {fmt}")

def save_file(data, filename: str, fmt: str = None):
    """Save data to JSON or YAML file"""
    if fmt is None:
        fmt = detect_format(filename)
    
    with open(filename, 'w') as f:
        if fmt == 'json':
            json.dump(data, f, indent=2)
        elif fmt == 'yaml':
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unknown format: {fmt}")

def auto_output_filename(input_file: str, target_format: str) -> str:
    """Generate output filename automatically"""
    base = '.'.join(input_file.split('.')[:-1])
    if target_format == 'json':
        return f"{base}.json"
    elif target_format == 'yaml':
        return f"{base}.yaml"
    else:
        raise ValueError(f"Unknown format: {target_format}")

def convert_file(input_file: str, output_file: str = None, verbose: bool = True):
    """Convert file between JSON and YAML formats"""
    
    # Check input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        return False
    
    # Detect input format
    try:
        input_format = detect_format(input_file)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    # Determine output format and filename
    if output_file is None:
        # Auto-generate output filename with opposite format
        output_format = 'yaml' if input_format == 'json' else 'json'
        output_file = auto_output_filename(input_file, output_format)
    else:
        try:
            output_format = detect_format(output_file)
        except ValueError as e:
            print(f"Error: {e}")
            return False
    
    # Check if conversion is needed
    if input_format == output_format:
        print(f"Warning: Input and output formats are the same ({input_format})")
        print(f"Creating a copy instead...")
    
    if verbose:
        print(f"Converting: {input_file} ({input_format.upper()}) → {output_file} ({output_format.upper()})")
    
    try:
        # Load data
        data = load_file(input_file, input_format)
        
        if verbose:
            print(f"Loaded {len(data.get('results', []))} benchmark results")
            if 'gpu_info' in data:
                print(f"GPU: {data['gpu_info'].get('name', 'Unknown')}")
        
        # Save in new format
        save_file(data, output_file, output_format)
        
        if verbose:
            print(f"✓ Successfully converted to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def batch_convert(input_files: list, output_format: str, verbose: bool = True):
    """Convert multiple files to the same output format"""
    success_count = 0
    fail_count = 0
    
    for input_file in input_files:
        output_file = auto_output_filename(input_file, output_format)
        
        if convert_file(input_file, output_file, verbose):
            success_count += 1
        else:
            fail_count += 1
        
        if verbose and len(input_files) > 1:
            print()  # Blank line between conversions
    
    if len(input_files) > 1 and verbose:
        print(f"\nSummary: {success_count} successful, {fail_count} failed")

def main():
    parser = argparse.ArgumentParser(
        description='Convert GPU benchmark results between JSON and YAML formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSON to YAML (auto-generate output filename)
  python convert_format.py results.json
  
  # Convert YAML to JSON (auto-generate output filename)
  python convert_format.py results.yaml
  
  # Specify output filename
  python convert_format.py results.json output.yaml
  
  # Batch convert multiple files to YAML
  python convert_format.py *.json --batch --to yaml
  
  # Batch convert multiple files to JSON
  python convert_format.py *.yaml --batch --to json
        """
    )
    
    parser.add_argument('input_file', type=str, nargs='+',
                       help='Input file(s) to convert')
    parser.add_argument('output_file', type=str, nargs='?',
                       help='Output file (optional, auto-generated if not provided)')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-generate output filename')
    parser.add_argument('--batch', action='store_true',
                       help='Batch convert multiple files')
    parser.add_argument('--to', type=str, choices=['json', 'yaml'],
                       help='Target format for batch conversion')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Batch conversion mode
    if args.batch:
        if not args.to:
            print("Error: --to must be specified for batch conversion (json or yaml)")
            sys.exit(1)
        
        batch_convert(args.input_file, args.to, verbose)
        sys.exit(0)
    
    # Single file conversion
    if len(args.input_file) > 1 and not args.output_file:
        print("Error: Multiple input files require --batch mode")
        sys.exit(1)
    
    input_file = args.input_file[0]
    output_file = args.output_file if not args.auto else None
    
    success = convert_file(input_file, output_file, verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
