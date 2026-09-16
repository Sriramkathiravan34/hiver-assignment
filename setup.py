"""Runs the full setup: generate data -> split golden eval -> train classifier -> build retrieval index.
Run this once before pipeline.py or eval_harness.py.
"""
import subprocess, sys, os

os.chdir(os.path.dirname(__file__))

steps = [
    ["python3", "generate_sample_data.py"],
    ["python3", "build_golden_eval.py"],
    ["python3", "intents.py"],
]
for s in steps:
    print(f"\n$ {' '.join(s)}")
    subprocess.run(s, check=True)

# Build retrieval index (import, not subprocess, so it shares this process's cwd)
import pandas as pd
from retrieval import build_index
train_df = pd.read_csv("../data/train_pairs.csv")
build_index(train_df)
print("\nRetrieval index built.")
print("\nSetup complete. Try: python3 pipeline.py  or  python3 ../eval/eval_harness.py")
