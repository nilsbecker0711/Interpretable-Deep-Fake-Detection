"""Integrated Gradients evaluator, mirroring the B-cos-v2 IntGrad protocol.

The reference (B-cos-v2 interpretability/explanation_methods/explainers/captum.py)
is a thin wrapper -- `class IntGrad(CaptumDerivative, IntegratedGradients)` that
only stores a config and forwards it to captum's attribute(). The algorithm is
captum's IntegratedGradients, so this uses captum directly with the reference's
exact configuration:

    IntGrad: {"n_steps": 50, "internal_batch_size": 8}      explanation_configs.py:28
    baselines: NEVER passed anywhere in the reference repo, so captum's default
               applies -- "use zero scalar corresponding to each input tensor",
               i.e. an ALL-ZEROS baseline, for b-cos and standard models alike.

Note the zero baseline is off-manifold for b-cos: its input is [r,g,b,1-r,1-g,1-b],
and all-zeros asserts x = 0 and 1-x = 0 at once, so it encodes no real image (a
black image is [0,0,0,1,1,1]). That is what the authors did and what keeps our
numbers comparable to theirs; convergence_delta is logged so the cost is visible.

Post-processing is IDENTICAL to BCOSEvaluator's -- sum over channels, smooth with
the shared box kernel, clamp to positive, rescale -- because comparing methods
under different post-processing would bias the localization scores. Everything
else (threshold sweep, top-k variant, store_images) is inherited.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from .B_COS_eval import BCOSEvaluator
from .GradCam_eval import WrappedModel
from .xai_common import smooth_map, normalize_max

logger = logging.getLogger(__name__)

# The reference's IntGrad config (explanation_configs.py). The n_steps=20 in the
# IntGrad.__init__ signature is a class default that this config overrides.
REFERENCE_N_STEPS = 50
REFERENCE_INTERNAL_BATCH_SIZE = 8


class IGEvaluator(BCOSEvaluator):
    """Integrated Gradients over the detector's class-1 (fake) logit.

    Subclasses BCOSEvaluator to reuse evaluate(), the threshold sweep and the
    top-k variant; only heatmap generation and the input preparation differ.
    Unlike the `bcos` method this runs on BOTH architectures -- IG needs nothing
    but gradients -- which makes it the one post-hoc method with no structural
    mismatch to b-cos (Grad-CAM averages a spatially varying gradient, which the
    b-cos pre-pooling convolutional head violates).
    """

    def __init__(self, model=None, device=None, smooth=15,
                 n_steps=REFERENCE_N_STEPS,
                 internal_batch_size=REFERENCE_INTERNAL_BATCH_SIZE,
                 target=1):
        super().__init__(model=model, device=device, smooth=smooth)
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.target = target
        self.in_channels = self._infer_in_channels(model)
        # NO explanation_mode here: in the reference only ours.py enters it, and
        # it detaches the dynamic scaling, which would break IG's completeness.
        self.wrapped = WrappedModel(model).to(device).eval()
        self.ig = IntegratedGradients(self.wrapped)
        logger.info("IG: n_steps=%d, internal_batch_size=%d, target=%d, "
                    "baseline=zeros, model expects %d channels",
                    self.n_steps, self.internal_batch_size, self.target,
                    self.in_channels)

    @staticmethod
    def _infer_in_channels(model):
        """in_channels of the first Conv2d (3 for standard, 6 for b-cos)."""
        if model is not None:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    return m.in_channels
        return 3

    def prepare_input(self, tensor):
        """Only b-cos models get the inverse channels (see BCOSEvaluator hook)."""
        if self.in_channels == 6 and tensor.shape[1] == 3:
            tensor = torch.cat([tensor, 1.0 - tensor], dim=1)
        return tensor

    def generate_heatmap(self, tensor):
        """Return (scored_map, visualization, output, model_prediction).

        scored_map is the positive-clamped, smoothed per-pixel attribution --
        the same object BCOSEvaluator scores, so the two are directly comparable.
        """
        img = self.prepare_input(tensor.to(self.device))
        if img.dim() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            logits = self.wrapped(img)
        model_prediction = int(logits[0].argmax().item())

        # baselines deliberately omitted -> captum's zero-scalar default, which
        # is exactly what the reference does (it never passes baselines).
        #
        # internal_batch_size only controls how many of the n_steps interpolation
        # points are pushed through the model at once; captum sums over all
        # n_steps either way, so lowering it changes NOTHING numerically -- only
        # peak memory. The reference's 8 was tuned for 224px ImageNet; a 768px
        # GPG grid is ~12x larger per sample and OOMs a 24 GB card on a 6-channel
        # b-cos net. Halve until it fits rather than forcing the caller to tune it.
        ibs = self.internal_batch_size
        while True:
            try:
                attributions, delta = self.ig.attribute(
                    img.requires_grad_(True),
                    target=self.target,
                    n_steps=self.n_steps,
                    internal_batch_size=ibs,
                    return_convergence_delta=True,
                )
                break
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                # A memory shortfall in the BACKWARD pass surfaces as
                # "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR" rather than a clean
                # OutOfMemoryError, so match on the message too. Anything that is
                # not a memory problem is re-raised untouched.
                msg = str(exc).lower()
                if not isinstance(exc, torch.OutOfMemoryError) and \
                        "out of memory" not in msg and "cudnn" not in msg:
                    raise
                torch.cuda.empty_cache()
                if ibs <= 1:
                    raise
                ibs = max(1, ibs // 2)
                logger.warning("IG hit a memory error (%s); retrying with "
                               "internal_batch_size=%d (result is unchanged, only "
                               "the chunking)", type(exc).__name__, ibs)
        if ibs != self.internal_batch_size:
            self.internal_batch_size = ibs   # remember, so later grids skip the retries
        logger.debug("IG convergence_delta: %.4e", float(delta.abs().max()))

        # Same protocol as the B-cos contribution map: sum the signed per-channel
        # attributions, smooth, THEN keep positives, then rescale to [0,1] for
        # the threshold sweep (the weighted cell/mask fraction is scale-free).
        contribs = attributions.sum(dim=1)[0].detach().cpu().numpy()
        scored_map = smooth_map(contribs, self.smooth)
        scored_map = np.clip(scored_map, 0, None)
        scored_map = normalize_max(scored_map)

        output = {"cls": logits, "convergence_delta": float(delta.abs().max())}
        return scored_map, None, output, model_prediction
