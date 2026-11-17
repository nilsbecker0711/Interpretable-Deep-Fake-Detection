"""Input x Gradient evaluator.

attribution = x * d(logit_target)/dx, via captum's InputXGradient on the
detector's class-1 (fake) logit. It is the single-step relative of Integrated
Gradients: for a model that is linear in its input, IxG equals IG with a zero
baseline. Like IG it runs on BOTH architectures.

On b-cos models this is deliberately NOT the native B-cos explanation. That one
is also "input x gradient", but it is taken inside explanation_mode, which
detaches the dynamic scaling so the gradient is the dynamic linear map W(x).
Here gradients flow through the scaling too, so IxG is the post-hoc baseline
that shows what the B-cos detach buys. Do not enter explanation_mode here.

The multiplication is in the model's input space (mean/std-normalised RGB, plus
[1-RGB] for b-cos), so the implicit reference point is the zero input -- the
same zero baseline IG uses, which keeps the two directly comparable.

Post-processing is IDENTICAL to IGEvaluator / BCOSEvaluator (shared
_postprocess): sum over channels, smooth, clamp to positive, rescale.
"""
import logging

import torch
from captum.attr import InputXGradient

from .IG_eval import IGEvaluator

logger = logging.getLogger(__name__)


class IxGEvaluator(IGEvaluator):
    """Input x Gradient over the detector's class-1 (fake) logit.

    Inherits prepare_input (3 vs. 6 channels), WrappedModel, evaluate(), the
    threshold sweep and the top-k variant from IGEvaluator; only the attribution
    call differs. One forward + one backward, so no chunking/OOM retry needed.
    """

    def __init__(self, model=None, device=None, smooth=15, target=1):
        super().__init__(model=model, device=device, smooth=smooth, target=target)
        self.ixg = InputXGradient(self.wrapped)
        logger.info("IxG: target=%d, model expects %d channels",
                    self.target, self.in_channels)

    def generate_heatmap(self, tensor):
        """Return (scored_map, visualization, output, model_prediction)."""
        img = self.prepare_input(tensor.to(self.device))
        if img.dim() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            logits = self.wrapped(img)
        model_prediction = int(logits[0].argmax().item())

        attributions = self.ixg.attribute(img.requires_grad_(True),
                                          target=self.target)

        scored_map = self._postprocess(attributions)
        output = {"cls": logits}
        return scored_map, None, output, model_prediction
