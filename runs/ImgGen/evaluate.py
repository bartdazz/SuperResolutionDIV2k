"""
Evaluation script: computes PSNR, SSIM, and LPIPS for the SR model on the
DIV2K validation set. Reports per-image scores and means, comparing the
SR model against the bicubic baseline (the LR_up input).

Usage:
    python evaluate.py

Requirements:
    pip install torchmetrics lpips        # for SSIM and LPIPS
    torchmetrics and lpips are optional;  # PSNR is always computed manually
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from Model.super_res_unet    import SuperResUNet
from Model.super_res_dataset import DIV2KDataset
from Model.inference         import ConditionalVectorField
from torch.utils.data        import DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 0
SCALE_FACTOR  = 4
HR_PATCH      = 128        # patch size for evaluation (matches training)
IN_CHANNELS   = 3
MODEL_CHANNELS = 128
NUM_RES_BLOCKS = 3
CHANNEL_MULT  = (1, 2, 2, 4, 4)
ATTENTION_RES = [4, 8]
EPSILON       = 0.05
N_ODE_STEPS   = 25
N_EVAL_IMGS   = 100        # number of validation images to evaluate on
BATCH_SIZE    = 8          # images per batch during eval
CHECKPOINT    = os.path.join(
    os.path.dirname(__file__),
    "../2026-06-05/16-41-56_cond_sr_div2k_x4/final_checkpoint.pt"
)

_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
HR_VALID_DIR = os.path.join(_ROOT, "data", "DIV2K_valid_HR")
LR_VALID_DIR = os.path.join(_ROOT, "data", "DIV2K_valid_LR_bicubic", "X" + str(SCALE_FACTOR))

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("torchmetrics not found — SSIM will not be computed. Install with: pip install torchmetrics")

try:
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("lpips not found — LPIPS will not be computed. Install with: pip install lpips")

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_psnr(pred, target, data_range=2.0):
    """pred, target: (B, C, H, W) in [-1, 1]. Returns PSNR in dB per image."""
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))          # (B,)
    psnr = 20 * torch.log10(torch.tensor(data_range, device=pred.device)) \
           - 10 * torch.log10(mse.clamp(min=1e-10))
    return psnr                                                # (B,)

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
print(f"Loaded checkpoint: {CHECKPOINT}")

# ── Data ──────────────────────────────────────────────────────────────────────
# seed=i gives a different deterministic crop per image index
eval_ds = DIV2KDataset(HR_VALID_DIR, LR_VALID_DIR, hr_patch_size=HR_PATCH,
                       scale=SCALE_FACTOR, seed=SEED)
eval_ds_size = min(N_EVAL_IMGS, len(eval_ds))
subset = torch.utils.data.Subset(eval_ds, range(eval_ds_size))
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Evaluation loop ───────────────────────────────────────────────────────────
sr_psnr_list, sr_ssim_list, sr_lpips_list = [], [], []
bic_psnr_list, bic_ssim_list, bic_lpips_list = [], [], []

print(f"\nEvaluating on {eval_ds_size} validation patches ({HR_PATCH}×{HR_PATCH})...")
print(f"ODE steps: {N_ODE_STEPS} | ε: {EPSILON}\n")

with torch.no_grad():
    for batch_idx, (lr, hr, _) in enumerate(loader):
        lr = lr.to(device)   # (B, 3, H, W) in [-1, 1]
        hr = hr.to(device)

        # SR model
        x_init = lr + EPSILON * torch.randn_like(lr)
        vf     = ConditionalVectorField(model, x0_cond=lr, y=None)
        t_span = torch.linspace(0, 1, N_ODE_STEPS, device=device)
        traj   = odeint(vf, x_init, t_span, method='euler')
        sr = traj[-1].clamp(-1, 1)

        # PSNR
        sr_psnr_list.append(compute_psnr(sr, hr))
        bic_psnr_list.append(compute_psnr(lr, hr))

        # SSIM
        if HAS_SSIM:
            sr_ssim_list.append(ssim_fn(sr, hr).unsqueeze(0).expand(sr.shape[0]))
            bic_ssim_list.append(ssim_fn(lr, hr).unsqueeze(0).expand(lr.shape[0]))

        # LPIPS (expects [-1, 1] — matches our normalisation)
        if HAS_LPIPS:
            sr_lpips_list.append(lpips_fn(sr, hr).squeeze())
            bic_lpips_list.append(lpips_fn(lr, hr).squeeze())

        n_done = (batch_idx + 1) * BATCH_SIZE
        print(f"  batch {batch_idx + 1}/{len(loader)} "
              f"({min(n_done, eval_ds_size)}/{eval_ds_size} images) "
              f"| SR PSNR: {compute_psnr(sr, hr).mean().item():.2f} dB")

# ── Aggregate results ─────────────────────────────────────────────────────────
sr_psnr  = torch.cat(sr_psnr_list).mean().item()
bic_psnr = torch.cat(bic_psnr_list).mean().item()

print("\n" + "=" * 52)
print(f"{'Metric':<12} {'Bicubic':>12} {'SR Model':>12} {'Delta':>10}")
print("-" * 52)
print(f"{'PSNR (dB)':<12} {bic_psnr:>12.3f} {sr_psnr:>12.3f} {sr_psnr - bic_psnr:>+10.3f}")

if HAS_SSIM and sr_ssim_list:
    sr_ssim  = torch.cat(sr_ssim_list).mean().item()
    bic_ssim = torch.cat(bic_ssim_list).mean().item()
    print(f"{'SSIM':<12} {bic_ssim:>12.4f} {sr_ssim:>12.4f} {sr_ssim - bic_ssim:>+10.4f}")

if HAS_LPIPS and sr_lpips_list:
    # LPIPS: lower is better
    sr_lpips  = torch.cat([x.view(-1) for x in sr_lpips_list]).mean().item()
    bic_lpips = torch.cat([x.view(-1) for x in bic_lpips_list]).mean().item()
    print(f"{'LPIPS↓':<12} {bic_lpips:>12.4f} {sr_lpips:>12.4f} {sr_lpips - bic_lpips:>+10.4f}")

print("=" * 52)
print(f"\nNote: PSNR/SSIM penalise perceptual textures — SR may score lower than")
print(f"bicubic despite looking sharper. LPIPS is the most reliable metric here.")
print(f"DIV2K ×4 reference: bicubic ~28.4 dB | good regression SR ~30–32 dB")
