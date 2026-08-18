# Grad-CAM and variations (consolidated evaluator)
#
# Single method-aware evaluator supporting: gradcam, xgrad, grad++, layergrad.
# Handles both 3-channel (standard) and 6-channel (b-cos) inputs.
# Optionally auto-searches the best conv layer for layergrad.
#
# This file replaces the former GradCam_eval / GradCam_evallayer / GradCam_evalbcos trio.
#
## Sources
# Parts of the implementation below are inspired by:
# - grad_cam.py / layer_cam.py - pytorch-grad-cam (https://github.com/jacobgil/pytorch-grad-cam) by Jacob Gildenblat
# - captum.py - B-cos-v2 (https://github.com/B-cos/B-cos-v2) by the B-cos authors

import os
import sys

# Set up project root and ensure it's in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerGradCam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .xai_common import smooth_map, normalize_max, topn_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CAM backends that come from the pytorch_grad_cam package (share one call signature).
_PGC_METHODS = {"gradcam": GradCAM, "xgrad": XGradCAM, "grad++": GradCAMPlusPlus}


def evaluate_heatmap(heatmap, grid_split=3, true_fake_pos=None, background_pixel=0):
    """Score a 2D grayscale heatmap on the grid pointing game.

    Returns (weighted guessed cell, per-cell intensity sums, weighted accuracy,
    unweighted guessed cell, unweighted accuracy).
    """
    rows, cols = heatmap.shape
    sec_rows = rows // grid_split
    sec_cols = cols // grid_split

    sections = [
        heatmap[i * sec_rows:(i + 1) * sec_rows, j * sec_cols:(j + 1) * sec_cols]
        for i in range(grid_split) for j in range(grid_split)
    ]

    # unweighted: count of "on" pixels per cell
    intensity_counts = [np.sum(sec > background_pixel) for sec in sections]
    fake_pred_unweighted = int(np.argmax(intensity_counts))
    total_nonzero = float(sum(intensity_counts))
    unweighted_accuracy = (
        intensity_counts[true_fake_pos] / total_nonzero
        if total_nonzero > 0 and true_fake_pos is not None
        and 0 <= true_fake_pos < len(intensity_counts) else 0.0
    )

    # weighted: sum of intensities per cell
    intensity_sums = [float(np.sum(sec)) for sec in sections]
    fake_pred_weighted = int(np.argmax(intensity_sums))
    total_intensity = float(sum(intensity_sums))
    weighted_accuracy = (
        intensity_sums[true_fake_pos] / total_intensity
        if total_intensity > 0 and true_fake_pos is not None
        and 0 <= true_fake_pos < len(intensity_sums) else 0.0
    )

    return fake_pred_weighted, intensity_sums, weighted_accuracy, fake_pred_unweighted, unweighted_accuracy


def find_last_feature_map_layer(model, example_input):
    """Return (module, name) whose output is the LAST spatial feature map, i.e.
    the tensor that enters global pooling. That is the correct Grad-CAM target.

    Determined dynamically from one forward pass, because the right module cannot
    be inferred from the module list:
      * the last raw nn.Conv2d is an *inner* conv (its output is pre-BatchNorm,
        pre-residual-add and pre-ReLU — not the feature map the classifier sees);
      * on b-cos backbones the last BcosConv2d is the CLASSIFIER head (fc /
        classifier_head), so hooking it yields a class-logit map, not features.
    Both mistakes silently biased b-cos vs. standard comparisons (fixed 2026-07).

    Rule: the last 4-D output with H,W >= 2 whose channel count is NOT
    num_classes. The channel test is essential for b-cos nets, where the
    classifier is CONVOLUTIONAL and runs before pooling (layer4 -> 1x1 conv ->
    pool), so it emits a *spatial* class map that would otherwise be picked;
    excluding it yields layer4 — the same kind of tensor hooked on the standard
    twin, which is what makes the comparison apples-to-apples.

    NOTE ViT: transformer blocks emit 3-D (B, N, D) token tensors, so the last
    4-D map is the conv stem. Grad-CAM on ViT needs a reshape_transform; pass an
    explicit target_layer there instead of relying on this.
    """
    order = []
    hooks = []
    backbone = getattr(model, "backbone", model)

    def _make_hook(name, mod):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) and len(out) else out
            if torch.is_tensor(t) and t.dim() == 4 and t.shape[-1] >= 2 and t.shape[-2] >= 2:
                order.append((name, mod, tuple(t.shape)))
        return hook

    for name, mod in backbone.named_modules():
        if name:  # skip the backbone itself
            hooks.append(mod.register_forward_hook(_make_hook(name, mod)))
    try:
        with torch.no_grad():
            out = model({"image": example_input})
        num_classes = out["cls"].shape[-1]
    except Exception as exc:  # discovery must never break the evaluation
        logger.warning("Feature-map discovery forward failed: %s", exc)
        return None, None
    finally:
        for h in hooks:
            h.remove()

    # drop class maps (b-cos: convolutional classifier before pooling)
    feats = [(n, m, s) for n, m, s in order if s[1] != num_classes]
    if not feats:
        return None, None
    # hooks fire on completion, so children precede their parent: the last entry
    # is the outermost module still emitting a feature map (e.g. layer4 / the
    # feature-extractor Sequential) — exactly the tensor the classifier consumes.
    name, mod, _ = feats[-1]
    return mod, name


def find_last_valid_conv_layer(module):
    """FALLBACK ONLY — prefer find_last_feature_map_layer().

    Returns the last conv-like module (excluding 'adjust'/'proj'). This is NOT the
    final feature map: on standard backbones it is an inner conv (pre-BN /
    pre-residual / pre-ReLU) and on b-cos backbones it is the classifier head.
    Kept for the layergrad layer search and as a fallback when the dynamic
    discovery forward pass fails.
    """
    last_conv, last_bcos = None, None
    for name, m in module.named_modules():
        if any(x in name for x in ("adjust", "proj")):
            continue
        if type(m).__name__ in ("BcosConv2d", "BcosConv2dWithScale"):
            last_bcos = m
        elif isinstance(m, nn.Conv2d):
            last_conv = m
    return last_bcos if last_bcos is not None else last_conv


def find_all_valid_conv_layers(module):
    """Return list of (name, conv-like module) excluding 'adjust'/'proj'.

    Prefers BcosConv2d wrappers on b-cos backbones (see find_last_valid_conv_layer).
    """
    convs, bcos_convs = [], []
    for name, m in module.named_modules():
        if any(x in name for x in ("adjust", "proj")):
            continue
        if type(m).__name__ in ("BcosConv2d", "BcosConv2dWithScale"):
            bcos_convs.append((name, m))
        elif isinstance(m, nn.Conv2d):
            convs.append((name, m))
    return bcos_convs if bcos_convs else convs


class WrappedModel(nn.Module):
    """Wrap a detector so CAM APIs see a simple forward(x) -> logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # detectors expect a dict with key 'image'
        return self.model({"image": x})["cls"]


def _to_rgb01(tensor):
    """[C,H,W] float tensor -> H x W x 3 float image in [0, 1] (b-cos 6ch -> first 3)."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    if img.shape[2] > 3:  # b-cos stacks [RGB, 1-RGB]; keep only the RGB channels
        img = img[:, :, :3]
    return ((img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.float32)


class GradCamEvaluator:
    def __init__(self, model, device, method="gradcam", target_layer=None, smooth=0,
                 reshape_transform=None):
        self.model = model.to(device)
        self.device = device
        self.method = method.lower()
        # Token models (ViT) emit (B, N, D) instead of (B, C, H, W); the reshape
        # folds the tokens back onto the patch grid. Resolved from the backbone
        # in _resolve_target_layer when not passed explicitly.
        self.reshape_transform = reshape_transform
        # NO smoothing for the CAM family, matching the B-cos paper (Sec. 4):
        # "all attribution maps (except for GradCAM, which is of much lower
        #  resolution to begin with) are smoothed by a 15x15 kernel to better
        #  account for negative attributions".
        # The smoothing exists to let positive and negative attributions cancel
        # before the positive-clamping step. CAM maps are ReLU'd (no negatives)
        # and already coarse — upsampled from a 24x24 feature map at our grid
        # size — so smoothing them serves no purpose. The b-cos contribution map
        # IS signed and therefore keeps its 15x15 kernel (see B_COS_eval).
        # Pass smooth=15 explicitly to restore the old behaviour.
        self.smooth = smooth
        if self.method not in _PGC_METHODS and self.method != "layergrad":
            raise ValueError(f"Unknown CAM method: {self.method}")

        # bookkeeping for layergrad auto-search
        self.best_layer_name = None
        self.best_layer_idx = None
        self._searched_layer = False

        self.wrapped_model = WrappedModel(self.model)

        if target_layer is not None:
            self.target_layer = target_layer
            self.cam = self._build_cam(self.target_layer)
        else:
            # Resolved lazily on the first explanation: the correct target is the
            # last SPATIAL FEATURE MAP (the tensor entering global pooling), which
            # is identified by running the real model on the real input size.
            self.target_layer = None
            self.cam = None

    def _resolve_target_layer(self, tensor):
        """Pick the final feature map as Grad-CAM target (see find_last_feature_map_layer)."""
        example = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
        # A backbone may declare its own target (B-cos-v2 style) instead of
        # relying on the shape heuristic. Required for token models, whose
        # feature map is not 4-D and which the heuristic cannot identify.
        backbone = getattr(self.model, "backbone", self.model)
        if hasattr(backbone, "get_gradcam_target"):
            layer = backbone.get_gradcam_target()
            name = "<declared by backbone.get_gradcam_target()>"
            if self.reshape_transform is None:
                self.reshape_transform = getattr(
                    backbone, "gradcam_reshape_transform", None)
            logger.info("Grad-CAM target layer: %s (%s)", name, type(layer).__name__)
            self.target_layer = layer
            self.cam = self._build_cam(layer)
            return
        layer, name = find_last_feature_map_layer(self.model, example.to(self.device))
        if layer is None:
            layer = find_last_valid_conv_layer(self.model.backbone)
            name = "<fallback: last conv module — NOT the final feature map>"
            if layer is None:
                raise ValueError("No valid target layer found in model backbone.")
        logger.info("Grad-CAM target layer: %s (%s)", name, type(layer).__name__)
        self.target_layer = layer
        self.cam = self._build_cam(layer)

    def _build_cam(self, layer):
        if self.method == "layergrad":
            if self.reshape_transform is not None:
                # captum's LayerGradCam has no reshape hook, so it would average
                # a (B, N, D) token tensor as if D were the spatial axes.
                raise ValueError(
                    "layergrad does not support token models (ViT): captum's "
                    "LayerGradCam cannot apply a reshape_transform. Use "
                    "method='gradcam' for these backbones.")
            # Captum LayerGradCam expects positional args: (forward_func, layer)
            return LayerGradCam(self.wrapped_model, layer)
        kwargs = ({} if self.reshape_transform is None
                  else {"reshape_transform": self.reshape_transform})
        cam = _PGC_METHODS[self.method](model=self.wrapped_model,
                                        target_layers=[layer], **kwargs)
        # Suppress pytorch_grad_cam's internal cv2.resize (bilinear): returning
        # None as the target size makes scale_cam_image skip resizing, leaving
        # the CAM at feature-map resolution. _raw_cam then upsamples NEAREST,
        # as B-cos-v2 and captum do.
        cam.get_target_width_height = lambda input_tensor: None
        return cam

    def extract_fake_position(self, path):
        try:
            return int(os.path.basename(path).split("_fake_")[1].split("_conf_")[0])
        except Exception:
            logger.warning("Could not extract fake position from %s", path)
            return -1

    def convert_to_numpy(self, tensor):
        """[1,C,H,W] or [C,H,W] float in [0,1] -> uint8 H x W x 3 (b-cos 6ch -> first 3)."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return (_to_rgb01(tensor) * 255).clip(0, 255).astype(np.uint8)

    def _raw_cam(self, tensor):
        """Compute the 2D grayscale CAM map for a single [C,H,W] tensor.

        Upsampling is NEAREST, matching B-cos-v2 (their GradCam calls
        LayerGradCam.interpolate(..., interpolate_mode="nearest")) and captum's
        own default. pytorch_grad_cam would otherwise upsample bilinearly with
        cv2.resize, so we suppress its internal resize (get_target_width_height
        -> None makes scale_cam_image skip it) and do the upsampling here.

        Measured on 250 test grids, nearest vs bilinear: ~0 at threshold 0,
        -0.012 (standard) and -0.037 (b-cos) at higher thresholds.
        """
        if self.cam is None:
            self._resolve_target_layer(tensor)
        target_hw = tuple(tensor.shape[-2:])

        if self.method == "layergrad":
            inp = tensor.unsqueeze(0).to(self.device).requires_grad_(True)
            # target=1 because the fake class is index 1
            attributions = self.cam.attribute(inp, target=1)  # -> [1, C, h, w]
            grayscale_cam = attributions.squeeze(0).mean(dim=0).detach().cpu().numpy()
            grayscale_cam = np.maximum(grayscale_cam, 0)  # ReLU (pytorch_grad_cam does this internally)
            if grayscale_cam.max() > 0:
                grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
        else:
            grayscale_cam = self.cam(
                input_tensor=tensor.unsqueeze(0),
                targets=[ClassifierOutputTarget(1)],
            )[0]

        # Every CAM variant now returns a map at FEATURE-MAP resolution; upsample
        # it here so all downstream code sees a full-resolution map.
        if grayscale_cam.shape != target_hw:
            t = torch.from_numpy(np.ascontiguousarray(grayscale_cam)).float()[None, None]
            grayscale_cam = F.interpolate(t, size=target_hw, mode="nearest")[0, 0].numpy()
        return grayscale_cam

    def generate_heatmap(self, tensor):
        """Return (grayscale_cam [H,W] in [0,1], rgb_img [H,W,3] in [0,1], overlay uint8).

        `tensor` is a single image of shape [C,H,W] (3 or 6 channels).
        """
        grayscale_cam = self._raw_cam(tensor)

        rgb_img = _to_rgb01(tensor)

        # (upsampling now happens inside _raw_cam, NEAREST for every CAM variant)

        # apply the shared protocol smoothing (identical across all XAI methods),
        # then rescale to [0,1] so the threshold sweep keeps its semantics
        grayscale_cam = normalize_max(smooth_map(grayscale_cam, self.smooth))

        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return grayscale_cam, rgb_img, overlay

    def _autosearch_layergrad(self, tensor_list, path_list, grid_split, auto_search_k):
        """Pick the conv layer giving the best mean weighted score over the first K grids."""
        self._searched_layer = True
        all_convs = find_all_valid_conv_layers(self.model.backbone)
        sample_n = min(auto_search_k, len(tensor_list))
        best_score, best_name, best_layer, best_idx = -1.0, None, None, None

        for idx, (name, conv) in enumerate(all_convs):
            cam_try = LayerGradCam(self.wrapped_model, conv)
            scores = []
            for tensor_grid, path in zip(tensor_list[:sample_n], path_list[:sample_n]):
                inp = tensor_grid[0].unsqueeze(0).to(self.device).requires_grad_(True)
                attr = cam_try.attribute(inp, target=1)
                gcam = attr.squeeze(0).mean(dim=0).detach().cpu().numpy()
                gcam = np.maximum(gcam, 0)
                if gcam.max() > 0:
                    gcam /= (gcam.max() + 1e-8)
                _, _, wa, _, _ = evaluate_heatmap(
                    heatmap=gcam, grid_split=grid_split,
                    true_fake_pos=self.extract_fake_position(path),
                )
                scores.append(wa)
            avg = float(np.mean(scores)) if scores else -1.0
            logger.info("[AutoSearch] Layer %2d (%s): mean weighted %.3f", idx, name, avg)
            if avg > best_score:
                best_score, best_name, best_layer, best_idx = avg, name, conv, idx

        if best_layer is None:
            logger.warning("[AutoSearch] no best layer found; keeping initial layer %s",
                           self.target_layer.__class__.__name__)
        else:
            self.best_layer_name, self.best_layer_idx = best_name, best_idx
            self.target_layer = best_layer
            self.cam = LayerGradCam(self.wrapped_model, self.target_layer)
            logger.info("[AutoSearch] Picked layer '%s' (index %d) with score %.3f",
                        best_name, best_idx, best_score)

    def evaluate(self, tensor_list, path_list, grid_split, threshold_steps=0, auto_search_k=0,
                 topn_fractions=(0.025,), store_images=True):
        """Run CAM on each grid tensor, threshold the map, and score localization.

        auto_search_k: if > 0 and method == 'layergrad', search the conv layer that
        maximizes the weighted score over the first `auto_search_k` grids.
        DISABLED by default: selecting the layer on the evaluation metric over the
        evaluation grids optimizes layergrad on test data, which no other method
        gets, and biases method comparisons. Use only for exploration, never for
        reported comparisons.
        """
        if self.method == "layergrad" and auto_search_k and not self._searched_layer:
            logger.warning(
                "[AutoSearch] layer auto-search is enabled (auto_search_k=%d). "
                "This tunes layergrad on the evaluation metric/grids and is NOT "
                "valid for method comparisons.", auto_search_k)
            self._autosearch_layergrad(tensor_list, path_list, grid_split, auto_search_k)

        results = []
        logger.info("Processing %d grids with grid_split=%d using CAM method=%s",
                    len(tensor_list), grid_split, self.method)

        for idx, (tensor_grid, path) in enumerate(zip(tensor_list, path_list)):
            logger.info("Evaluating grid %d from file: %s", idx, path)
            true_fake_pos = self.extract_fake_position(path)

            intensity_map, norm_img, _ = self.generate_heatmap(tensor_grid[0])
            # store_images=False keeps only the scalar scores. The images dominate
            # the pickle by ~4 orders of magnitude (1.3 GB vs the 549-byte summary
            # that plotting actually reads), and rendering them costs a
            # show_cam_on_image call per grid per operating point at 768x768.
            original_image = self.convert_to_numpy(tensor_grid[0]) if store_images else None

            # Actual model prediction on this grid (the CAM target stays class 1 =
            # fake, but the real prediction lets results be conditioned on correct
            # classification, e.g. when models share grids in COMPARE_eval).
            with torch.no_grad():
                logits = self.wrapped_model(tensor_grid[0].unsqueeze(0).to(self.device))
                probs = torch.softmax(logits, dim=1)
                model_prediction = int(logits[0].argmax().item())
                model_confidence = float(probs[0, model_prediction].item())

            thresholds = [None]
            if threshold_steps > 0:
                thresholds += [i / threshold_steps for i in range(1, threshold_steps + 1)]

            for t in thresholds:
                mask = intensity_map if t is None else np.where(intensity_map < t, 0.0, intensity_map)
                thresholded_overlay = show_cam_on_image(norm_img, mask, use_rgb=True) if store_images else None

                (fake_pred_weighted, intensity_sums, weighted_accuracy,
                 fake_pred_unweighted, unweighted_accuracy) = evaluate_heatmap(
                    heatmap=mask, grid_split=grid_split, true_fake_pos=true_fake_pos,
                )

                results.append({
                    "threshold": t if t is not None else 0,
                    "path": path,
                    "original_image": original_image,
                    "heatmap": thresholded_overlay,
                    "weighted_guessed_fake_position": fake_pred_weighted,
                    "unweighted_guess_fake_position": fake_pred_unweighted,
                    "true_fake_position": true_fake_pos,
                    "weighted_localization_score": weighted_accuracy,
                    "unweighted_localization_score": unweighted_accuracy,
                    "model_prediction": model_prediction,
                    "model_confidence": model_confidence,
                })

                logger.info(
                    "Method %s | Layer %s | Threshold %s | %s: true pos %d, "
                    "weighted pred %d (acc %.3f) | unweighted pred %d (acc %.3f)",
                    self.method, self.best_layer_name or "default",
                    "none" if t is None else f"{t:.3f}", os.path.basename(path),
                    true_fake_pos, fake_pred_weighted, weighted_accuracy,
                    fake_pred_unweighted, unweighted_accuracy,
                )

            # TOP-N variant ('OursQ' in the B-cos paper, Figs. 6+7): score the
            # map restricted to its most strongly contributing pixels. Stored
            # ADDITIONALLY, under a string key, so the threshold results above are
            # untouched and every existing pickle stays readable.
            for frac in (topn_fractions or ()):
                mask = topn_mask(intensity_map, frac)
                (fake_pred_weighted, intensity_sums, weighted_accuracy,
                 fake_pred_unweighted, unweighted_accuracy) = evaluate_heatmap(
                    heatmap=mask, grid_split=grid_split, true_fake_pos=true_fake_pos,
                )
                results.append({
                    "threshold": f"top{frac:g}",
                    "topn_fraction": float(frac),
                    "topn_pixels": int((mask > 0).sum()),
                    "path": path,
                    "original_image": original_image,
                    "heatmap": show_cam_on_image(norm_img, mask, use_rgb=True) if store_images else None,
                    "weighted_guessed_fake_position": fake_pred_weighted,
                    "unweighted_guess_fake_position": fake_pred_unweighted,
                    "true_fake_position": true_fake_pos,
                    "weighted_localization_score": weighted_accuracy,
                    "unweighted_localization_score": unweighted_accuracy,
                    "model_prediction": model_prediction,
                    "model_confidence": model_confidence,
                })
                logger.info(
                    "Method %s | TOP-%g (%d px) | %s: true pos %d, weighted acc %.3f",
                    self.method, frac, int((mask > 0).sum()),
                    os.path.basename(path), true_fake_pos, weighted_accuracy,
                )

        return results
