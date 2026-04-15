import os
import argparse
import numpy as np
import json
import torch

def main():
    p = argparse.ArgumentParser(description="Evaluate audio counterfactual results (V2)")
    p.add_argument("--exp_dir", required=True, help="Path like audio/results/<exp_name>")
    p.add_argument("--steps", type=int, default=1, help="Number of sequential steps")
    args = p.parse_args()

    for s in range(args.steps):
        step_dir = os.path.join(args.exp_dir, f"step_{s}")
        info_dir = os.path.join(step_dir, "info")
        if not os.path.isdir(info_dir):
            print(f"Skipping step {s}, missing info dir.")
            continue
            
        json_files = sorted([x for x in os.listdir(info_dir) if x.endswith(".json")])
        if not json_files:
            continue
            
        flipped = []
        l1s = []
        tcb, tca = [], []
        mnac = []
        
        for jf in json_files:
            with open(os.path.join(info_dir, jf), "r") as f:
                d = json.load(f)
            
            flipped.append(1.0 if d["flipped"] else 0.0)
            l1s.append(d["l1"])
            tcb.append(d["target_conf_before"])
            tca.append(d["target_conf_after"])
            
            # MNAC: how many OTHER classes flipped?
            pb = np.array(d["all_probs_before"])
            pa = np.array(d["all_probs_after"])
            target_idx = d["target_class"]
            
            # Exclude target class
            mask = np.ones_like(pb, dtype=bool)
            mask[target_idx] = False
            
            # Count changes in other classes crossing 0.5
            changed = ((pb[mask] > 0.5) != (pa[mask] > 0.5)).sum()
            mnac.append(changed)

        print(f"\n--- Results for Step {s} ---")
        print(f"Samples: {len(json_files)}")
        print(f"Flip rate (%): {100 * np.mean(flipped):.2f}")
        print(f"Mean L1: {np.mean(l1s):.6f}")
        print(f"MNAC (Mean Number of Attributes Changed): {np.mean(mnac):.4f}")
        print(f"Target conf before: {np.mean(tcb):.4f}")
        print(f"Target conf after:  {np.mean(tca):.4f}")

if __name__ == "__main__":
    main()
