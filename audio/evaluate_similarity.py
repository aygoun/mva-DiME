import os
import argparse
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

def main():
    parser = argparse.ArgumentParser(description="Compute Source Preservation/Similarity")
    parser.add_argument("--cf_dir", type=str, required=True, help="Directory with cf_wav")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with original_wav")
    parser.add_argument("--model", type=str, default="microsoft/unispeech-sat-base-plus-sv", help="Model for speaker embeddings")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use Unispeech-SAT for speaker verification embeddings
    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModelForAudioClassification.from_pretrained(args.model, ignore_mismatched_sizes=True).to(device)
    model.eval()
    
    wav_files = sorted([x for x in os.listdir(args.ref_dir) if x.endswith(".wav")])
    scores = []
    
    for fn in wav_files:
        p0 = os.path.join(args.ref_dir, fn)
        p1 = os.path.join(args.cf_dir, fn)
        if not os.path.exists(p1):
            continue
            
        try:
            # Load and preprocess
            w0, sr0 = librosa.load(p0, sr=16000)
            w1, sr1 = librosa.load(p1, sr=16000)
            
            inputs0 = extractor(w0, sampling_rate=16000, return_tensors="pt").to(device)
            inputs1 = extractor(w1, sampling_rate=16000, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Extract embeddings (last hidden state or specific head)
                # For SAT models, the output usually contains embeddings
                emb0 = model(**inputs0).logits
                emb1 = model(**inputs1).logits
                
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(emb0, emb1).item()
                scores.append(sim)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
        
    if scores:
        print(f"Mean Source Similarity (Cosine via {args.model}): {np.mean(scores):.4f}")
    else:
        print("No files found or all failed.")

if __name__ == "__main__":
    main()
