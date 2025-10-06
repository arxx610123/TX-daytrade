#!/usr/bin/env python3
"""main.py - Arxx-daytrade entrypoint

Defensive entry script:
- checks model subfolders (trend_brain, reversal_brain, volatility_brain, volume_brain, risk_veto)
- tries to import model.master_brain.MasterBrain; if not present uses a dummy MasterBrain for testing
- saves run results to model/logs/*.json
Usage: python main.py
"""
import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
EXPECTED_SUBDIRS = ["trend_brain", "reversal_brain", "volatility_brain", "volume_brain", "risk_veto"]

def check_structure():
    missing = []
    if not os.path.isdir(MODEL_DIR):
        missing.append("model (folder)")
    else:
        for d in EXPECTED_SUBDIRS:
            p = os.path.join(MODEL_DIR, d)
            if not os.path.isdir(p):
                missing.append(os.path.join("model", d))
    return missing

MasterBrain = None
try:
    from model.master_brain import MasterBrain
except Exception as e:
    print("Warning: could not import model.master_brain.MasterBrain:", e)
    print("Using dummy MasterBrain for testing.")
    import random
    class MasterBrain:
        def __init__(self, config_path=None):
            self.config_path = config_path

        def run_once(self):
            mapping = {1: "BUY", 0: "HOLD", -1: "SELL"}
            return {
                "trend": mapping[random.choice([1,0,-1])],
                "reversal": mapping[random.choice([1,0,-1])],
                "volatility": mapping[random.choice([0,0,1,-1])],
                "volume": mapping[random.choice([1,0,-1])],
                "risk_veto": "PASS",
                "master_decision": mapping[random.choice([1,0,-1])]
            }

def main():
    print("Arxx-daytrade main.py - start:", datetime.now().isoformat())
    print("Script path:", os.path.abspath(__file__))
    missing = check_structure()
    if missing:
        print("\nWarning: expected folders missing:")
        for m in missing:
            print("  -", m)
        print("Please create them in the project root. Continuing with available modules.\n")

    config_path = os.path.join(ROOT, "config.yaml")
    brain = MasterBrain(config_path)
    try:
        result = brain.run_once()
    except Exception as e:
        print("Error during MasterBrain.run_once():", e)
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n=== Sub-brain outputs ===")
    for k, v in result.items():
        print(f"{k:14s}: {v}")

    out_dir = os.path.join(ROOT, "model", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nSaved run result to:", out_file)
    print("Done.")

if __name__ == '__main__':
    main()
