#!/usr/bin/env python
# validate_shots.py
import json, sys, pathlib

PATH = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "train_split.json")

def iter_rows(path):
    """Yield one JSON object at a time from .json or .jsonl."""
    txt = path.read_text()
    try:
        data = json.loads(txt)              # regular JSON array?
        if isinstance(data, list):
            yield from data
            return
    except json.JSONDecodeError:
        pass                                # fall through to JSONL

    # JSON-Lines fallback
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def check_row(row):
    shots = row.get("shots")
    if not isinstance(shots, list):
        return "shots is not list"

    for j, shot in enumerate(shots):
        if not isinstance(shot, dict):
            return f"shot[{j}] is {type(shot).__name__}"
        if "inputs" not in shot or "output" not in shot:
            return f"shot[{j}] missing keys"
    return None            # row is good

def main():
    bad = []
    for i, row in enumerate(iter_rows(PATH), 1):
        reason = check_row(row)
        if reason:
            bad.append((i, reason))
    good = i - len(bad) if 'i' in locals() else 0

    print(f"✓ {good} good rows")
    print(f"✗ {len(bad)} bad rows")
    if bad:
        print("\nFirst few problems:")
        for line_no, why in bad[:20]:
            print(f"  line {line_no}: {why}")

if __name__ == "__main__":
    if not PATH.exists():
        sys.exit(f"File not found: {PATH}")
    main()