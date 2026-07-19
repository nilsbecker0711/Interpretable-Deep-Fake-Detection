"""Shared utilities for the pointing-game evaluators.

Kept dependency-light (torch/numpy only) so every evaluator can import it
without pulling in the training stack.
"""
import numpy as np
import torch
import torch.nn.functional as F


def smooth_map(attribution, smooth):
    """Smooth a 2D attribution map with the original B-cos localisation protocol:
    F.avg_pool2d(x, smooth, stride=1, padding=(smooth-1)//2).

    The SAME smoothing must be applied to every XAI method under comparison —
    method-specific denoising would bias localization scores.

    Args:
        attribution: 2D map, numpy array or torch tensor [H, W].
        smooth: odd kernel size; values <= 1 disable smoothing.

    Returns:
        2D float32 numpy array, same shape.
    """
    if not smooth or smooth <= 1:
        return np.asarray(attribution, dtype=np.float32) if not isinstance(attribution, torch.Tensor) \
            else attribution.detach().cpu().numpy().astype(np.float32)
    assert smooth % 2 == 1, f"smooth kernel must be odd, got {smooth}"

    if isinstance(attribution, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(attribution)).float()
    else:
        t = attribution.detach().float().cpu()
    t = t.reshape(1, 1, *t.shape[-2:])
    t = F.avg_pool2d(t, smooth, stride=1, padding=(smooth - 1) // 2)
    return t[0, 0].numpy().astype(np.float32)


def normalize_max(map_2d):
    """Scale a non-negative 2D map to [0, 1] by its max (no-op if empty/all-zero).

    Only needed so the threshold sweep keeps its [0,1] semantics; the weighted
    localization score itself is scale-invariant.
    """
    m = float(np.max(map_2d)) if map_2d.size else 0.0
    if m > 0:
        return (map_2d / m).astype(np.float32)
    return map_2d.astype(np.float32)


# ---------------------------------------------------------------------------
# Canonical grid representation and per-model input adaptation.
#
# Grids are stored model-agnostically as raw [0,1] RGB (3 channels). Each model
# then gets its OWN preprocessing at load time: standard models mean/std
# normalization, b-cos models the [x, 1-x] 6-channel encoding. Feeding one
# family's tensor values to the other family evaluates that model out of
# distribution and contaminates every cross-model comparison.
# ---------------------------------------------------------------------------

def canonicalize_grid(tensor, mean=0.5, std=0.5, warn_name=""):
    """Convert a grid/image tensor to canonical raw [0,1] RGB, 3 channels.

    Handles [C,H,W] and [B,C,H,W]. 6-channel b-cos tensors are sliced to their
    RGB half. Tensors that look mean/std-normalized (negative values) are
    denormalized with the given mean/std and flagged loudly, so stale grid
    assets are surfaced rather than silently mis-scored.
    """
    import logging
    logger = logging.getLogger(__name__)

    t = tensor.clone().float()
    ch_dim = 1 if t.dim() == 4 else 0
    if t.shape[ch_dim] >= 6:
        t = t[:, :3] if ch_dim == 1 else t[:3]

    lo, hi = float(t.min()), float(t.max())
    if lo < -0.05:  # mean/std-normalized inputs live in [-1, 1]
        logger.warning(
            "canonicalize_grid: %s has values in [%.3f, %.3f] — looks mean/std-"
            "normalized. Denormalizing with mean=%s std=%s. Regenerate this grid "
            "asset in canonical [0,1] form.", warn_name or "tensor", lo, hi, mean, std)
        mean_t = torch.as_tensor(mean, dtype=t.dtype, device=t.device).reshape(-1, 1, 1)
        std_t = torch.as_tensor(std, dtype=t.dtype, device=t.device).reshape(-1, 1, 1)
        t = t * std_t + mean_t
        lo, hi = float(t.min()), float(t.max())
    if lo < -0.01 or hi > 1.5:
        raise ValueError(
            f"canonicalize_grid: {warn_name or 'tensor'} range [{lo:.3f}, {hi:.3f}] "
            f"is neither raw [0,1] nor mean/std-normalized — unknown input scale.")
    return t.clamp(0.0, 1.0)


def infer_in_channels(model, default=3):
    """Return the in_channels of the model's first Conv2d (3 standard, 6 b-cos)."""
    if model is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                return m.in_channels
    return default


def adapt_for_model(tensor, model, mean=0.5, std=0.5):
    """Take a CANONICAL [0,1] 3-channel tensor and produce the model's input.

    b-cos models (6 input channels): append the inverse channels [x, 1-x].
    standard models (3 input channels): apply the model's mean/std normalization.
    """
    ch_dim = 1 if tensor.dim() == 4 else 0
    assert tensor.shape[ch_dim] == 3, (
        f"adapt_for_model expects a canonical 3-channel tensor, got shape {tuple(tensor.shape)}")

    if infer_in_channels(model) == 6:
        return torch.cat([tensor, 1.0 - tensor], dim=ch_dim)
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).reshape(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).reshape(-1, 1, 1)
    return (tensor - mean_t) / std_t


def mpg_mask_game(mask, heatmap):
    """Mask Pointing Game score for one heatmap vs. a ground-truth
    manipulation mask. Single source of truth — used by both the MPG CLI
    (notebooks/.../MPG_eval.py) and the in-training XAI monitor.

    The mask is nearest-neighbor-resized to the heatmap resolution if needed.

    Returns:
        (unweighted_accuracy, intensity_accuracy):
        unweighted — fraction of nonzero heatmap pixels inside the mask;
        intensity  — fraction of total heatmap mass inside the mask.
    """
    import cv2

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    intensity_map = heatmap.copy()

    if mask.shape != intensity_map.shape:
        h, w = intensity_map.shape
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(mask.dtype)

    binary = (heatmap > 0).astype(np.uint8)
    correct_pixels = np.sum((binary == 1) & (mask == 1))
    total_predicted = np.sum(binary == 1)
    accuracy = correct_pixels / total_predicted if total_predicted > 0 else 0

    total_intensity = np.sum(intensity_map)
    mask_intensity = np.sum(intensity_map[mask == 1])
    intensity_accuracy = mask_intensity / total_intensity if total_intensity > 0 else 0

    return round(float(accuracy), 4), round(float(intensity_accuracy), 4)
