#!/usr/bin/env python3
import json
from collections import Counter

def examine_data(filename, num_entries=5):
    print(f"=== Examining {filename} ===")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    print(f"First {num_entries} entries:")
    
    # Check the structure of first few entries
    for i, entry in enumerate(data[:num_entries]):
        print(f"\nEntry {i}:")
        print(f"  Keys: {list(entry.keys())}")
        for key, value in entry.items():
            if key == 'shots':
                print(f"  {key}: {type(value)} - {value}")
            else:
                print(f"  {key}: {type(value)} - {str(value)[:100]}...")
    
    # Check data types for 'shots' column across all entries
    if data:
        shots_types = []
        shots_values = []
        for entry in data:
            if 'shots' in entry:
                shots_types.append(type(entry['shots']))
                shots_values.append(entry['shots'])
        
        print(f"\nShots column analysis:")
        print(f"  Type distribution: {Counter(shots_types)}")
        print(f"  Sample values: {shots_values[:10]}")

if __name__ == "__main__":
    examine_data("train_split.json")
    print("\n" + "="*50 + "\n")
    examine_data("eval_split.json") 