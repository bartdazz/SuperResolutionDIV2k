import torch
import torch.nn as nn


class ConditionalVectorField(nn.Module):
    """Wraps a SuperResUNet for ODE integration with torchdiffeq.

    torchdiffeq expects the vector field as f(t, x), so all conditioning
    tensors are captured at construction time and only (t, x_t) are
    exposed at call time.

    Args:
        model:    a SuperResUNet instance (in eval mode during integration).
        x0_cond:  low-res conditioning tensor (B, C, H, W), or None when
                  condition_on_lr=False.
        y:        integer class labels (B,), or None when num_classes=None.
    """

    def __init__(
        self,
        model: nn.Module,
        x0_cond: torch.Tensor = None,
        y: torch.Tensor = None,
    ):
        super().__init__()
        self.model   = model
        self.x0_cond = x0_cond
        self.y       = y

    def forward(self, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        t_batch = t.expand(x_t.shape[0])
        return self.model(x_t, t_batch, x0=self.x0_cond, y=self.y)
