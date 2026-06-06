"""
Same as ImgGen.py but with attention at downsampling factor 4 disabled.
This tests whether the horizontal stripe artifacts come from the ds=4
attention blocks operating on 128×128 (for 512px input) feature maps.
"""
import sys
import os
import types
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torchdiffeq import odeint

from Model.super_res_unet    import SuperResUNet
from Model.super_res_dataset import DIV2KDataset
from Model.inference         import ConditionalVectorField
from Model.utils             import plot_comparison
from Model.unet_dario        import AttentionBlock, Downsample, Upsample
from torch.utils.data        import DataLoader


# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 0
SCALE_FACTOR  = 4
HR_PATCH      = 512
IN_CHANNELS   = 3
MODEL_CHANNELS = 128
NUM_RES_BLOCKS = 3
CHANNEL_MULT  = (1, 2, 2, 4, 4)
ATTENTION_RES = [4, 8]
EPSILON       = 0.05
N_EVAL_IMGS   = 8
N_ODE_STEPS   = 25
CHECKPOINT    = os.path.join(
    os.path.dirname(__file__),
    "../2026-06-05/16-41-56_cond_sr_div2k_x4/final_checkpoint.pt"
)
DISABLE_DS    = 4   # set to None to keep all attention

_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
HR_VALID_DIR = os.path.join(_ROOT, "data", "DIV2K_valid_HR")
LR_VALID_DIR = os.path.join(_ROOT, "data", "DIV2K_valid_LR_bicubic", "X" + str(SCALE_FACTOR))

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ── Disable attention at ds=DISABLE_DS ────────────────────────────────────────
def disable_attention_at_ds(model, disable_ds, channel_mult):
    """
    Patch AttentionBlocks at the given downsampling factor to act as identity
    (return input unchanged). Works by replaying the ds-tracking logic from
    the UNet constructor.

    Returns the number of patched blocks.
    """
    if disable_ds is None:
        return 0

    def _identity(self, x):
        return x

    patched = 0

    # Encoder: ds starts at 1, doubles at each Downsample block
    ds = 1
    for block in model.input_blocks:
        for module in block:
            if isinstance(module, Downsample):
                ds *= 2
                break
        for module in block:
            if isinstance(module, AttentionBlock) and ds == disable_ds:
                module.forward = types.MethodType(_identity, module)
                patched += 1

    # Decoder: ds starts at max, halves after each Upsample block
    max_ds = 2 ** (len(channel_mult) - 1)
    ds = max_ds
    for block in model.output_blocks:
        for module in block:
            if isinstance(module, AttentionBlock) and ds == disable_ds:
                module.forward = types.MethodType(_identity, module)
                patched += 1
        for module in block:
            if isinstance(module, Upsample):
                ds //= 2
                break

    return patched


# ── Model ─────────────────────────────────────────────────────────────────────
model = SuperResUNet(
    in_channels           = IN_CHANNELS,
    condition_on_lr       = True,
    model_channels        = MODEL_CHANNELS,
    out_channels          = IN_CHANNELS,
    num_res_blocks        = NUM_RES_BLOCKS,
    attention_resolutions = ATTENTION_RES,
    dropout               = 0.0,
    channel_mult          = CHANNEL_MULT,
    conv_resample         = True,
    num_heads             = 4,
    num_heads_upsample    = -1,
    use_scale_shift_norm  = True,
    num_classes           = None,
).to(device)

state_dict = torch.load(CHECKPOINT, map_location=device)['final_ema_model']
model.load_state_dict(state_dict)
model.eval()

n_patched = disable_attention_at_ds(model, DISABLE_DS, CHANNEL_MULT)
print(f"Disabled attention at ds={DISABLE_DS}: {n_patched} block(s) patched")

# ── Data ──────────────────────────────────────────────────────────────────────
eval_ds = DIV2KDataset(HR_VALID_DIR, LR_VALID_DIR, hr_patch_size=HR_PATCH,
                       scale=SCALE_FACTOR, seed=SEED)
lr_batch, hr_batch, _ = next(iter(
    DataLoader(eval_ds, batch_size=N_EVAL_IMGS, shuffle=False, num_workers=0)
))
lr_batch = lr_batch.to(device)
hr_batch = hr_batch.to(device)

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    x_init = lr_batch + EPSILON * torch.randn_like(lr_batch)
    vf     = ConditionalVectorField(model, x0_cond=lr_batch, y=None)
    t_span = torch.linspace(0, 1, N_ODE_STEPS, device=device)
    traj   = odeint(vf, x_init, t_span, method='euler')

suffix = f"no_attn_ds{DISABLE_DS}" if DISABLE_DS else "all_attn"
plot_comparison(
    lr_batch, traj[-1], hr_batch,
    epoch=0,
    show=False,
    n_img=N_EVAL_IMGS,
    save_prefix=f"cond_sr_div2k_x{SCALE_FACTOR}_{suffix}",
    output_dir=os.path.dirname(__file__),
)
print(f"Saved to {os.path.dirname(__file__)}/")
