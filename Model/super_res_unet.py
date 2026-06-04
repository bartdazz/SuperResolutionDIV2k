import torch
from .unet_dario import UNetModel


class SuperResUNet(UNetModel):
    """Flexible UNet for CIFAR10 flow matching, supporting four operating modes.

    The mode is determined at construction time by two flags:

        condition_on_lr=False, num_classes=None  →  unconditional SR  (variant 1)
        condition_on_lr=True,  num_classes=None  →  conditional SR    (variant 2)
        condition_on_lr=False, num_classes=10    →  class-cond. gen.  (variant 3)
        condition_on_lr=True,  num_classes=10    →  class-cond. SR    (variant 4)

    When condition_on_lr=True the low-res image x0 is concatenated with
    the current state x_t along the channel dimension before entering the
    UNet, so the effective number of input channels is doubled.

    Class conditioning is already implemented in UNetModel: when num_classes
    is not None, a learned embedding label_emb(y) is added to the time
    embedding (AdaGN-style).

    Args:
        in_channels: number of image channels (3 for RGB CIFAR10).
        condition_on_lr: whether to concatenate the LR image with x_t.
        **kwargs: forwarded verbatim to UNetModel — must include
                  model_channels, out_channels, num_res_blocks,
                  attention_resolutions, channel_mult, etc.
    """

    def __init__(self, in_channels: int, condition_on_lr: bool = False, **kwargs):
        super().__init__(
            in_channels * 2 if condition_on_lr else in_channels,
            **kwargs,
        )
        self.condition_on_lr = condition_on_lr

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor = None,
        y: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: current interpolated state  (B, C, H, W).
            t:   time values                 (B,).
            x0:  low-res conditioning image  (B, C, H, W).
                 Required when condition_on_lr=True, ignored otherwise.
            y:   integer class labels        (B,).
                 Required when num_classes is not None, ignored otherwise.
        """
        x = torch.cat([x_t, x0], dim=1) if self.condition_on_lr else x_t
        return super().forward(x, t, y=y)
