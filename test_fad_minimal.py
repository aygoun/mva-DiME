from frechet_audio_distance import FrechetAudioDistance
import sys

# Try to bypass any library that might be hijacking sys.argv
sys.argv = [sys.argv[0]]

fad = FrechetAudioDistance(model_name="vggish")
# Note: score expects paths to directories
res = fad.score("audio/results/exp1_guitar_remove/step_0/original_wav", 
                "audio/results/exp1_guitar_remove/step_0/cf_wav")
print(f"FAD Score: {res}")
