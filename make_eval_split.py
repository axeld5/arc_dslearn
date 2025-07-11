import json, random
from pathlib import Path

SRC_FILE   = Path("train_set/train_set.json")
TRAIN_OUT  = Path("train_split.json")
EVAL_OUT   = Path("eval_split.json")
SPLIT_SEED = 42
EVAL_FRAC  = 0.10            # 10 %

data = json.loads(SRC_FILE.read_text())
random.Random(SPLIT_SEED).shuffle(data)

split = int(len(data) * (1 - EVAL_FRAC))
train_data, eval_data = data[:split], data[split:]

TRAIN_OUT.write_text(json.dumps(train_data, indent=2))
EVAL_OUT.write_text(json.dumps(eval_data, indent=2))
print(f"✓ wrote {len(train_data)} train   → {TRAIN_OUT}")
print(f"✓ wrote {len(eval_data)} eval    → {EVAL_OUT}")