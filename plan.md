# Implementation Plan: Audio Counterfactual Explanations via DiME

This plan outlines the steps to adapt the DiME (Diffusion Models for
Counterfactual Explanations) pipeline for audio using Mel spectrograms,
implement key evaluation metrics, and perform experiments on the OpenMIC
dataset.

## 1. Environment & Dependencies

- [ ] Install missing dependencies using uv:
  - `frechet-audio-distance` (for FAD)
  - `speechbrain` (for Speaker/Source Similarity via ECAPA-TDNN)

- [ ] Verify `uv` environment and GPU availability.

## 2. Core Implementation Improvements

- [ ] **Enhance `audio/main_audio.py`**:
  - [ ] Add support for multiple target labels and sequential removal (e.g.,
        remove instrument A, then instrument B).
  - [ ] Integrate `Mel` class from `audio-diffusion` for consistent
        audio-to-spectrogram conversion if needed (currently using
        `diffusers.pipelines.deprecated.audio_diffusion.Mel`).
  - [ ] Improve logging of intermediate steps for visualization.
- [ ] **Create `audio/evaluation_fad.py`**:
  - [ ] Implement FAD computation comparing generated counterfactuals against
        the real dataset distribution.
- [ ] **Create `audio/evaluation_similarity.py`**:
  - [ ] Implement Source Preservation metric using `ECAPA-TDNN` from
        `speechbrain`.
- [ ] **Enhance `audio/evaluate_metrics.py`**:
  - [ ] Integrate FAD, Source Similarity, and MNAC (Mean Number of Attributes
        Changed) using the classifier as an oracle.

## 3. Experiments

- [ ] **Experiment 1: Single Class Removal**
  - [ ] Target: "Guitar" (label 8).
  - [ ] Run DiME to remove guitar from samples where it's present.
  - [ ] Evaluate Flip Ratio, L1/L2, FAD, Similarity, MNAC.
- [ ] **Experiment 2: Sequential Class Removal and then add another instrument,
      switch**
  - [ ] Target: "Guitar" (label 8) then "Drums" (label 6).
  - [ ] Generate CF1 (remove guitar), then generate CF2 (remove drums from CF1).
  - [ ] Evaluate metrics at each step.

## 4. Visualization & Reporting

- [ ] **Jupyter Notebook (`notebooks/audio_results_analysis.ipynb`)**:
  - [ ] **Figure 1**: Qualitative Mel-Spectrogram Comparisons (Input, CF,
        Difference Map).
  - [ ] **Figure 2**: Diversity Assessment (3-5 CFs for one input).
  - [ ] **Figure 3**: Generation Pipeline visualization (Forward/Reverse
        diffusion).
  - [ ] **Table 1**: Main Results Table (Flip Ratio, FAD, Similarity, LSD/L1,
        MNAC).
- [ ] **Exporting**:
  - [ ] Export LaTeX tables.
  - [ ] Export PGF plots.
  - [ ] Save all figures to `report/figures/`.

## 5. Validation

- [ ] Ensure `uv run` works for all scripts.
- [ ] Verify that audio inversion (`tensor_to_audio`) produces audible and
      correct results for both original and CF.
- [ ] Check sparsity of perturbations via Difference Maps.
