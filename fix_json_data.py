#!/usr/bin/env python3
import json
from pathlib import Path

def fix_json_data(input_file, output_file):
    """
    Fix inconsistent data types in JSON dataset files.
    Specifically handles the 'shots' column issue where some entries are lists
    while others are null or different types.
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original data size: {len(data)} entries")
    
    # Fix each entry in the data
    fixed_data = []
    for i, entry in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing entry {i}/{len(data)}")
        
        # Create a copy of the entry to avoid modifying the original
        fixed_entry = entry.copy()
        
        # Fix the 'shots' field - ensure it's always a list
        if 'shots' in fixed_entry:
            shots = fixed_entry['shots']
            if shots is None:
                fixed_entry['shots'] = []
            elif not isinstance(shots, list):
                # Convert to list if it's not already
                fixed_entry['shots'] = [shots] if shots else []
        
        # Ensure other fields are consistent types
        for field in ['system_prompt', 'user_prompt', 'assistant_prompt']:
            if field in fixed_entry and fixed_entry[field] is None:
                fixed_entry[field] = ""
        
        fixed_data.append(fixed_entry)
    
    print(f"Writing fixed data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"✓ Fixed {len(fixed_data)} entries and saved to {output_file}")

if __name__ == "__main__":
    # Fix both train and eval splits
    fix_json_data("train_split.json", "train_split.json")
    fix_json_data("eval_split.json", "eval_split.json")
    
    print("✓ All files fixed! Use train_split_fixed.json and eval_split_fixed.json") 