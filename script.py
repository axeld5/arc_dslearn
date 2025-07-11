import json
from datasets import load_dataset

# First, let's preprocess the JSON to ensure consistent data types
def preprocess_json_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert inputs and outputs to JSON strings for consistency
    for record in data:
        if record['shots']:
            for shot in record['shots']:
                shot['inputs'] = json.dumps(shot['inputs'])
                shot['output'] = json.dumps(shot['output'])
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

# Preprocess the data
preprocess_json_file("eval_split.json", "eval_split_processed.json")

# Now load the preprocessed dataset
raw_ds = load_dataset("json",
                      data_files="eval_split_processed.json",
                      split="train")

print(f"Dataset loaded successfully with {len(raw_ds)} examples!")