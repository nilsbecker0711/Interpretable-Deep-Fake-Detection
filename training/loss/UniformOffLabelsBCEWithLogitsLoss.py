from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="bce_off_labels")
class UniformOffLabelsBCEWithLogitsLoss(nn.Module):
    """
    BCE loss with off value targets equal to some value.
    If not provided then it is `1/N`, where `N` is the number of classes.
    The on values are set to 1 as normal.

    This is best explained with an example, as follows:

    Examples
    --------
    Let N=5 and our target be t=3. Then t will be mapped to the following:
    `[0.2, 0.2, 0.2, 1.0, 0.2]`.

    If a particular off value is provided instead for example 2e-3 then it's:
    `[2e-3, 2e-3, 2e-3, 1.0, 2e-3]`
    """

    def __init__(self, reduction: str = "mean", off_label: Optional[float] = 0.05):
        super().__init__()
        self.reduction = reduction
        self.off_label = off_label

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]

        num_classes = x.shape[-1]
        off_value = self.off_label or (1.0 / num_classes)
        if target.shape != x.shape:
            target = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)

        # make off values (0) to at least 1/N
        target = target.clamp(min=off_value)

        return F.binary_cross_entropy_with_logits(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        result = f"reduction={self.reduction}, "
        if self.off_label is not None:
            result += f"off_label={self.off_label}, "
        result = result[:-2]
        return result