"""
Shared B-cos explanation visualizer.

Used by:
  - training/generate_explanations.py  (standalone batch job)
  - training/trainer/trainer.py        (validation monitoring during training)
"""
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def has_explain(model):
    """True if this detector's backbone exposes a B-cos explain() method."""
    backbone = getattr(model, "backbone", None)
    return backbone is not None and callable(getattr(backbone, "explain", None))


def save_explanation_grid(model, data_loader, out_dir, device, num_images=10, step=None):
    """
    Generate B-cos explanations for both classes for up to `num_images` images.

    Saves one PNG per image with 4 panels:
      [original | class-0 (real) heatmap | class-1 (fake) heatmap | class-1 overlay]

    Parameters
    ----------
    model       : detector with model.backbone.explain()
    data_loader : DataLoader yielding dicts with 'image', 'label', 'image_path'
    out_dir     : directory to write PNGs into
    device      : torch.device
    num_images  : how many images to explain (default 10)
    step        : global training step; included in filename when provided (None for standalone use)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0

    for data_dict in data_loader:
        if saved >= num_images:
            break

        for key in data_dict:
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(device)

        images = data_dict["image"]       # [B, 6, H, W]
        labels = data_dict["label"]
        paths  = data_dict.get("image_path", [None] * images.shape[0])

        for j in range(images.shape[0]):
            if saved >= num_images:
                break

            # detach from the batch graph; explain() needs its own grad tape
            img = images[j].detach().unsqueeze(0).requires_grad_(True)
            label = int(labels[j].item())

            exp0 = model.backbone.explain(img, idx=0)   # real class
            exp1 = model.backbone.explain(img, idx=1)   # fake class

            hm0      = exp0["explanation"]   # [H, W, 4] RGBA numpy
            hm1      = exp1["explanation"]
            pred_cls = int(exp1["prediction"])

            img_rgb = img[0, :3].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            alpha1  = hm1[:, :, 3]
            overlay = (
                img_rgb * (1 - alpha1[..., None]) + hm1[:, :, :3] * alpha1[..., None]
            ).clip(0, 1)

            p = paths[j]
            img_name = Path(
                p[0] if isinstance(p, (list, tuple)) else p
            ).stem if p else f"img{saved}"
            correct = "ok" if pred_cls == label else "wrong"
            step_prefix = f"step{step:06d}_" if step is not None else ""
            fname = out_dir / f"{step_prefix}{saved:03d}_{img_name}_l{label}_p{pred_cls}_{correct}.png"

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original")
            axes[1].imshow(hm0)
            axes[1].set_title("B-cos class 0 (real)")
            axes[2].imshow(hm1)
            axes[2].set_title("B-cos class 1 (fake)")
            axes[3].imshow(overlay)
            axes[3].set_title("Overlay (class 1)")
            for ax in axes:
                ax.axis("off")
            step_str = f"step {step} | " if step is not None else ""
            fig.suptitle(
                f"{step_str}true: {'real' if label == 0 else 'fake'} | "
                f"pred: {'real' if pred_cls == 0 else 'fake'} | {correct}"
            )
            plt.tight_layout()
            plt.savefig(fname, dpi=120, bbox_inches="tight")
            plt.close(fig)

            saved += 1

    return saved
