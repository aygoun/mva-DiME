"""Compatibility layer between HuggingFace diffusers and DiME's sampling loop.

DiME's ``p_sample_loop`` (core/sample_utils.py) expects:
  diffusion.q_sample(x_start, t)
  diffusion.p_mean_variance(model, x_t, t, ...)  -> dict(mean, variance, log_variance)
  diffusion.sqrt_alphas_cumprod              (1-D **numpy** array, indexed by t)

And a model callable:  model(x, t, **kwargs) -> predicted noise (epsilon).

``core.gaussian_diffusion._extract_into_tensor`` converts numpy→Tensor
internally, so all pre-computed schedules are stored as **numpy arrays**
to stay compatible with both the original guided-diffusion code and DiME.

This module wraps ``DDPMScheduler`` + ``UNet2DModel`` from ``diffusers`` to
expose those interfaces, enabling drop-in use of the pretrained
``teticio/audio-diffusion-breaks-256`` checkpoint inside DiME.
"""

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

from core.gaussian_diffusion import _extract_into_tensor


# ---------------------------------------------------------------------------
# Diffusion wrapper
# ---------------------------------------------------------------------------

def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


class DiffusersDiffusion:
    """Wraps a ``DDPMScheduler`` to match guided-diffusion's API.

    All pre-computed schedule arrays are stored as **numpy** to be compatible
    with ``core.gaussian_diffusion._extract_into_tensor``.
    """

    def __init__(self, scheduler):
        betas = scheduler.betas
        alphas = 1.0 - betas
        alphas_cumprod = scheduler.alphas_cumprod
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = scheduler.config.num_train_timesteps

        self.betas = _to_np(betas)
        self.alphas_cumprod = _to_np(alphas_cumprod)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

        _betas = _to_np(betas)
        _acp = _to_np(alphas_cumprod)
        _acp_prev = _to_np(alphas_cumprod_prev)
        _alphas = _to_np(alphas)

        self.posterior_variance = (
            _betas * (1.0 - _acp_prev) / (1.0 - _acp)
        )
        self.posterior_log_variance_clipped = np.log(
            np.clip(self.posterior_variance, a_min=1e-20, a_max=None)
        )
        self.posterior_mean_coef1 = (
            _betas * np.sqrt(_acp_prev) / (1.0 - _acp)
        )
        self.posterior_mean_coef2 = (
            (1.0 - _acp_prev) * np.sqrt(_alphas) / (1.0 - _acp)
        )

    # -- forward process --------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    # -- reverse process ---------------------------------------------------

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * eps
        )

    def p_mean_variance(self, model, x_t, t, clip_denoised=True,
                        denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        eps = model(x_t, t, **model_kwargs)
        pred_xstart = self._predict_xstart_from_eps(x_t, t, eps)

        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape)
            * pred_xstart
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape)
            * x_t
        )
        variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_variance = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance,
            "pred_xstart": pred_xstart,
        }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class DiffusersModelFn:
    """Makes a diffusers ``UNet2DModel`` callable as ``model(x, t, **kw)``."""

    def __init__(self, unet):
        self.unet = unet

    def __call__(self, x, t, **kwargs):
        return self.unet(sample=x, timestep=t).sample

    def to(self, device):
        self.unet = self.unet.to(device)
        return self

    def eval(self):
        self.unet.eval()
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

AUDIO_DIFFUSION_REPO = "teticio/audio-diffusion-breaks-256"


def load_audio_diffusion(repo_id=AUDIO_DIFFUSION_REPO, device="cpu"):
    """Download (if needed) and return ``(model_fn, diffusion)``.

    Uses ``DiffusionPipeline.from_pretrained`` which auto-detects the
    ``AudioDiffusionPipeline`` class registered in the model repo.
    """
    print(f"Loading diffusers pipeline from {repo_id} ...")
    pipeline = DiffusionPipeline.from_pretrained(repo_id)

    model_fn = DiffusersModelFn(pipeline.unet).to(device).eval()
    diffusion = DiffusersDiffusion(pipeline.scheduler)

    print(f"  UNet params: {sum(p.numel() for p in pipeline.unet.parameters()):,}")
    print(f"  Timesteps:   {diffusion.num_timesteps}")
    return model_fn, diffusion
