#!/usr/bin/env bash
set -e

# Helper: extract a zip with Python (handles zip64 archives > 4 GB that
# macOS's built-in `unzip` corrupts).
extract_zip() {
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$2"
}

# ==================================================================
# DDPM checkpoint
# ==================================================================
wget https://huggingface.co/guillaumejs2403/DiME/resolve/main/ddpm-celeba.pt
mkdir -p models
mv ddpm-celeba.pt models/ddpm-celeba.pt

# ==================================================================
# FSD50K dataset (eval-only subset ~ 1.6 GB — full dev set is ~17 GB)
#
# We download only ground_truth + eval_audio, which gives 10 231 clips
# for counterfactual generation.  Uncomment the dev_audio block below
# if you also need the ~41 k training clips for DDPM fine-tuning.
# NOTE: the dev audio is a split zip (6 parts totalling ~17 GB).
# ==================================================================
FSD50K_DIR=dataset/FSD50K
mkdir -p "$FSD50K_DIR"

echo "Downloading FSD50K ground truth ..."
wget -q --show-progress -O "$FSD50K_DIR/FSD50K.ground_truth.zip" \
    https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip
extract_zip "$FSD50K_DIR/FSD50K.ground_truth.zip" "$FSD50K_DIR"
rm "$FSD50K_DIR/FSD50K.ground_truth.zip"

echo "Downloading FSD50K eval audio (~1.6 GB) ..."
wget -q --show-progress -O "$FSD50K_DIR/FSD50K.eval_audio.zip" \
    https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip
extract_zip "$FSD50K_DIR/FSD50K.eval_audio.zip" "$FSD50K_DIR"
rm "$FSD50K_DIR/FSD50K.eval_audio.zip"

echo "FSD50K eval subset ready at $FSD50K_DIR"

# Uncomment below to also download the dev set (~17 GB compressed,
# needed for DDPM fine-tuning).  The dev audio is a split zip archive
# (z01–z05 + .zip); all parts must be downloaded and concatenated
# before extraction.

echo "Downloading FSD50K dev audio (6 parts, ~17 GB total) ..."
for part in z01 z02 z03 z04 z05 zip; do
    echo "  Downloading FSD50K.dev_audio.$part ..."
    wget -q --show-progress -O "$FSD50K_DIR/FSD50K.dev_audio.$part" \
        https://zenodo.org/records/4060432/files/FSD50K.dev_audio.$part
done
echo "Reassembling split archive ..."
cat "$FSD50K_DIR"/FSD50K.dev_audio.z{01,02,03,04,05} \
    "$FSD50K_DIR"/FSD50K.dev_audio.zip \
    > "$FSD50K_DIR/FSD50K.dev_audio_combined.zip"
rm "$FSD50K_DIR"/FSD50K.dev_audio.z{01,02,03,04,05} "$FSD50K_DIR"/FSD50K.dev_audio.zip
extract_zip "$FSD50K_DIR/FSD50K.dev_audio_combined.zip" "$FSD50K_DIR"
rm "$FSD50K_DIR/FSD50K.dev_audio_combined.zip"
echo "FSD50K dev audio ready."
