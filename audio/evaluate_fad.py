import os
import argparse
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="Compute FAD for audio CF")
    parser.add_argument("--cf_dir", type=str, required=True, help="Directory with cf_wav")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with original_wav or real dataset samples")
    parser.add_argument("--model", type=str, default="vggish", choices=["vggish", "pann", "clap"], help="Embedding model for FAD")
    
    args = parser.parse_args()

    # STRIP sys.argv before importing to prevent hijacking
    sys.argv = [sys.argv[0]]
    from frechet_audio_distance import FrechetAudioDistance
    
    fad = FrechetAudioDistance(model_name=args.model, verbose=False)
    
    score = fad.score(args.ref_dir, args.cf_dir)
    print(f"FAD ({args.model}) score: {score:.4f}")

if __name__ == "__main__":
    main()
