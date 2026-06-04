from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class FlowMatchingLoss(ABC):
    """Base class for flow matching losses.

    It can be used bot for super-resolution and for generation.

    Handles time sampling, linear interpolation, and the model call.
    Subclasses define only the base distribution via sample_base().

    The __call__ method reads two attributes from the model to decide
    what conditioning to pass:
        model.condition_on_lr  — if True, x0_clean is forwarded as x0
        model.num_classes      — if not None, y is forwarded as a class label

    This means the same loss instance can be used with any SuperResUNet
    variant; the model itself controls what conditioning it expects.
    """

    def __call__(
        self,
        model: torch.nn.Module,
        x0_clean: torch.Tensor,
        x1: torch.Tensor,
        y: torch.Tensor = None,
        ot_eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute the flow-matching MSE loss for one mini-batch.

        Args:
            model:    a SuperResUNet instance.
            x0_clean: upsampled low-res image (B, C, H, W).
                      Used for DataDependentLoss;
                      ignored by GaussianLoss.
            x1:       ground-truth high-res image (B, C, H, W).
            y:        integer class labels (B,), or None.
            ot_eps:   small offset to keep t strictly above 0.
        """
        device = x1.device
        t = torch.rand(x1.shape[0], device=device)
        t = (t + ot_eps) / (1 + ot_eps)
        T = t[:, None, None, None]

        x0_noisy = self.sample_base(x0_clean, x1) # IMPROVE THIS!
        # I want to super resolute also without adding noise
        x_t      = (1 - T) * x0_noisy + T * x1
        v_true   = x1 - x0_noisy

        x0_cond = x0_clean if model.condition_on_lr else None
        y_cond  = y if model.num_classes is not None else None
        v_pred  = model(x_t, t, x0=x0_cond, y=y_cond)

        return F.mse_loss(v_pred, v_true)

    @abstractmethod
    def sample_base(self, x0_clean: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        r"""Return a sample x0_noisy from the base distribution \rho_0(\cdot|x1)."""


class DataDependentLoss(FlowMatchingLoss):
    r"""Data-dependent coupling: x0_noisy = x0_clean + epsilon · \xi, \xi \sim N(0, I).

    Args:
        epsilon: standard deviation of the noise added to x0_clean.
    """

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def sample_base(self, x0_clean: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x0_clean + self.epsilon * torch.randn_like(x0_clean)


class GaussianLoss(FlowMatchingLoss):
    r"""Independent Gaussian coupling: x0_noisy \sim N(0, I).

    Standard flow-matching base distribution, independent of x1.
    Used for the class-conditional generation variant (variant 3),
    where no low-res image is available at generation time.
    """

    def sample_base(self, x0_clean: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x1)
