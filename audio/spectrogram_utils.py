import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F


SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
SPEC_SIZE = 128
NUM_CHANNELS = 3

REF_DB = 0.0
MIN_DB = -80.0


def load_audio(path, sr=SAMPLE_RATE, duration=None):
    """Load an audio file and resample to sr."""
    y, orig_sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    if duration is not None:
        max_len = int(sr * duration)
        if len(y) > max_len:
            y = y[:max_len]
        elif len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
    return y


def audio_to_mel(y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 n_mels=N_MELS):
    """Compute a log-mel spectrogram from a waveform."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def normalize_spec(log_mel):
    """Map a log-mel spectrogram to [-1, 1] using a fixed dB range."""
    normed = (log_mel - MIN_DB) / (REF_DB - MIN_DB)
    normed = np.clip(normed, 0.0, 1.0)
    return normed * 2.0 - 1.0


def denormalize_spec(normed):
    """Inverse of normalize_spec: [-1, 1] → dB."""
    normed = np.clip(normed, -1.0, 1.0)
    s01 = (normed + 1.0) / 2.0
    return s01 * (REF_DB - MIN_DB) + MIN_DB


def spec_to_tensor(log_mel, size=SPEC_SIZE, num_channels=NUM_CHANNELS):
    """Normalize, resize, and replicate a log-mel spectrogram to a torch tensor."""
    normed = normalize_spec(log_mel)
    t = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.squeeze(0)                              # (1, size, size)
    t = t.repeat(num_channels, 1, 1)              # (C, size, size)
    return t


def tensor_to_spec(tensor, orig_time_frames=None):
    """Convert a (C, H, W) tensor in [-1, 1] back to a log-mel in dB."""
    arr = tensor[0].cpu().numpy()                  # (H, W)
    if orig_time_frames is not None:
        from PIL import Image
        img = Image.fromarray(((arr + 1) / 2 * 255).astype(np.uint8))
        img = img.resize((orig_time_frames, N_MELS), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
    return denormalize_spec(arr)


def mel_to_audio(log_mel_db, sr=SAMPLE_RATE, n_fft=N_FFT,
                 hop_length=HOP_LENGTH, n_iter=64):
    """Reconstruct a waveform from a log-mel spectrogram via Griffin-Lim."""
    mel_power = librosa.db_to_power(log_mel_db)
    y = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter,
    )
    return y


def audio_to_tensor(path, sr=SAMPLE_RATE, duration=None, size=SPEC_SIZE):
    """End-to-end: audio file → (C, size, size) tensor in [-1, 1]."""
    y = load_audio(path, sr=sr, duration=duration)
    log_mel = audio_to_mel(y, sr=sr)
    orig_time_frames = log_mel.shape[1]
    t = spec_to_tensor(log_mel, size=size)
    return t, orig_time_frames


def tensor_to_audio(tensor, orig_time_frames, sr=SAMPLE_RATE):
    log_mel_db = tensor_to_spec(tensor, orig_time_frames=orig_time_frames)
    return mel_to_audio(log_mel_db, sr=sr)


def spec_augment(tensor, freq_mask_param=20, time_mask_param=20,
                 num_freq_masks=2, num_time_masks=2):
    """Apply SpecAugment (frequency + time masking) to a (C, H, W) tensor."""
    t = tensor.clone()
    _, h, w = t.shape
    fill = t.mean().item()

    for _ in range(num_freq_masks):
        f = int(torch.randint(0, min(freq_mask_param, h), (1,)).item())
        f0 = int(torch.randint(0, h - f, (1,)).item())
        t[:, f0:f0 + f, :] = fill

    for _ in range(num_time_masks):
        tw = int(torch.randint(0, min(time_mask_param, w), (1,)).item())
        t0 = int(torch.randint(0, w - tw, (1,)).item())
        t[:, :, t0:t0 + tw] = fill

    return t


def random_time_shift(y, sr=SAMPLE_RATE, max_shift_sec=0.5):
    """Randomly roll a waveform by up to max_shift_sec seconds."""
    max_shift = int(sr * max_shift_sec)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(y, shift)


def mixup(x1, y1, x2, y2, alpha=0.4):
    """Mixup two (tensor, label) pairs."""
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    return x, y1, y2, lam
