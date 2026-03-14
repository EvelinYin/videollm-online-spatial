#!/usr/bin/env python3
"""
Combine all JSON files in a folder into a single JSON file.
"""

import json
import argparse
from pathlib import Path

# example usage:
#  python combine_json.py ./data ./output/combined.json


def combine_json_files(input_folder, output_path, flatten=False):
    """
    Combine all JSON files in a folder into one JSON file.

    Args:
        input_folder: Path to folder containing JSON files
        output_path: Path where combined JSON will be saved
        flatten: If True, merges all objects into one. If False, creates an array.
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")

    # Find all JSON files
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {input_folder}")

    print(f"Found {len(json_files)} JSON files")

    # Combine files
    if flatten:
        # Merge all objects into one
        combined_data = {}
        for json_file in json_files:
            print(f"  Reading {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    combined_data.update(data)
                else:
                    print(f"    Warning: {json_file.name} is not a dict, skipping")
    else:
        # Create an array of all data
        combined_data = []
        for json_file in json_files:
            print(f"  Reading {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.append(data)

    # Save combined file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Combined JSON saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all JSON files in a folder into one JSON file"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing JSON files"
    )
    parser.add_argument(
        "output_path",
        help="Path where combined JSON will be saved"
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Merge all objects into one instead of creating an array"
    )

    args = parser.parse_args()

    try:
        combine_json_files(args.input_folder, args.output_path, args.flatten)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
