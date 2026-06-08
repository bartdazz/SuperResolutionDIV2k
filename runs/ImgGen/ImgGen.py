# %%
import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchdiffeq import odeint

from Model.super_res_unet import SuperResUNet
from Model.inference      import ConditionalVectorField
from Model.utils          import plot_sr_patch, _auto_zoom_box

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 0
SCALE_FACTOR = 4
IN_CHANNELS  = 3
MODEL_CHANNELS  = 128
NUM_RES_BLOCKS  = 3
CHANNEL_MULT    = (1, 2, 2, 4, 4)
ATTENTION_RES   = [4, 8]
EPSILON      = 0.05
N_ODE_STEPS  = 25

# Size of the patch to super-resolve, in LR image pixels.
# The model will receive this patch bicubic-upsampled to LR_CROP_SIZE * SCALE_FACTOR.
LR_CROP_SIZE = 128

# Crop origins (r0, c0) in LR pixels, or None for auto-select.
# Two crops are shown with different border colours.
CROP_ORIGIN_1 = None          # first crop  — auto = highest variance region
CROP_ORIGIN_2 = None          # second crop — auto = highest variance non-overlapping
CROP_COLOR_1  = 'red'
CROP_COLOR_2  = 'blue'

# Folder containing the LR images you want to super-resolve.
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
IMAGE_DIR  = os.path.join(_ROOT, "data", "extra")

CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    "../2026-06-05/16-41-56_cond_sr_div2k_x4/final_checkpoint.pt"
)

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

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

state_dict = torch.load(CHECKPOINT, map_location=device)['final_model']
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded: {CHECKPOINT}")

# ── Load images ───────────────────────────────────────────────────────────────
def load_image(path):
    """Load a PNG/JPG as a (3, H, W) tensor normalised to [-1, 1]."""
    img = Image.open(path).convert('RGB')
    t   = ToTensor()(img)          # [0, 1]
    return (t - 0.5) / 0.5        # [-1, 1]

paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))
)
if not paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")
print(f"Found {len(paths)} image(s) in {IMAGE_DIR}")

# ── Per-image inference ───────────────────────────────────────────────────────
out_dir = os.path.dirname(__file__)

def sr_patch(full_lr, r0, c0):
    """Extract LR crop at (r0, c0), upsample, run SR. Returns (lr_crop, sr_crop)."""
    r0 = min(r0, full_lr.shape[1] - LR_CROP_SIZE)
    c0 = min(c0, full_lr.shape[2] - LR_CROP_SIZE)
    lr_crop = full_lr[:, r0:r0 + LR_CROP_SIZE, c0:c0 + LR_CROP_SIZE]
    hr_size = LR_CROP_SIZE * SCALE_FACTOR
    lr_up   = F.interpolate(
        lr_crop.unsqueeze(0), size=(hr_size, hr_size),
        mode='bicubic', align_corners=False,
    )
    with torch.no_grad():
        x_init = lr_up + EPSILON * torch.randn_like(lr_up)
        vf     = ConditionalVectorField(model, x0_cond=lr_up, y=None)
        t_span = torch.linspace(0, 1, N_ODE_STEPS, device=device)
        traj   = odeint(vf, x_init, t_span, method='euler')
    sr = traj[-1].squeeze(0).clamp(-1, 1)
    return lr_crop, sr, r0, c0


for img_path in paths:
    name    = os.path.splitext(os.path.basename(img_path))[0]
    full_lr = load_image(img_path).to(device)
    _, H, W = full_lr.shape

    if H < LR_CROP_SIZE or W < LR_CROP_SIZE:
        print(f"  Skipping {name}: image ({H}×{W}) smaller than LR_CROP_SIZE={LR_CROP_SIZE}")
        continue

    # ── Crop 1 ────────────────────────────────────────────────────────────
    if CROP_ORIGIN_1 is None:
        r1, c1 = _auto_zoom_box(full_lr, LR_CROP_SIZE)
    else:
        r1, c1 = CROP_ORIGIN_1
    lr_crop1, sr_crop1, r1, c1 = sr_patch(full_lr, r1, c1)
    box1 = (r1, c1, LR_CROP_SIZE, LR_CROP_SIZE)

    # ── Crop 2 (non-overlapping with crop 1) ──────────────────────────────
    if CROP_ORIGIN_2 is None:
        r2, c2 = _auto_zoom_box(full_lr, LR_CROP_SIZE, avoid=box1)
    else:
        r2, c2 = CROP_ORIGIN_2
    lr_crop2, sr_crop2, r2, c2 = sr_patch(full_lr, r2, c2)
    box2 = (r2, c2, LR_CROP_SIZE, LR_CROP_SIZE)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_sr_patch(
        full_lr  = full_lr.cpu(),
        patches  = [
            (lr_crop1.cpu(), sr_crop1.cpu(), box1, CROP_COLOR_1),
            (lr_crop2.cpu(), sr_crop2.cpu(), box2, CROP_COLOR_2),
        ],
        scale_factor = SCALE_FACTOR,
        show         = False,
        save_prefix  = name,
        output_dir   = out_dir,
    )
    print(f"  Saved: {out_dir}/{name}_sr_patch.png")
    print(f"    crop1 ({CROP_COLOR_1}): r={r1}, c={c1}  |  crop2 ({CROP_COLOR_2}): r={r2}, c={c2}")
