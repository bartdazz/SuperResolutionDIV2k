# %%
import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from Model.super_res_unet    import SuperResUNet
from Model.loss              import DataDependentLoss
from Model.super_res_dataset import build_DIV2K, DIV2KDataset
from Model.inference         import ConditionalVectorField
from Model.utils             import plot_comparison
from torch.utils.data        import DataLoader


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ── Hyperparameters ───────────────────────────────────────────────────────────
_ROOT           = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SCALE_FACTOR    = 4
HR_DIR          = os.path.join(_ROOT, "data", "DIV2K_train_HR")
LR_DIR          = os.path.join(_ROOT, "data", "DIV2K_train_LR_bicubic", "X" + str(SCALE_FACTOR))
# HR_VALID_DIR    = os.path.join(_ROOT, "data", "DIV2K_valid_HR")
# LR_VALID_DIR    = os.path.join(_ROOT, "data", "DIV2K_valid_LR_bicubic", "X" + str(SCALE_FACTOR))
HR_VALID_DIR    = os.path.join(_ROOT, "data", "extra")
LR_VALID_DIR    = os.path.join(_ROOT, "data", "extra")
HR_PATCH        = 512
IN_CHANNELS     = 3
MODEL_CHANNELS  = 128
NUM_RES_BLOCKS  = 3
CHANNEL_MULT    = (1, 2, 2, 4, 4)
ATTENTION_RES   = [4, 8]
BATCH_SIZE      = 32
EPSILON         = 0.05
NUM_EPOCHS      = 100
LR              = 3e-4
OT_EPS          = 1e-6
N_EVAL_IMGS     = 1
VARIANT = "imgen_sr_div2k_x" + str(SCALE_FACTOR)
run_dir = '.'

# %%
eval_ds  = DIV2KDataset(HR_VALID_DIR, LR_VALID_DIR, hr_patch_size=HR_PATCH, scale=SCALE_FACTOR, seed=0)
fixed_lr, fixed_hr, _ = next(iter(DataLoader(eval_ds, batch_size=N_EVAL_IMGS, shuffle=False, num_workers=0)))
fixed_lr = fixed_lr.to(device)
fixed_hr = fixed_hr.to(device)


# %%
model = SuperResUNet(
    in_channels          = IN_CHANNELS,
    condition_on_lr      = True,          # x0 concatenated with x_t → 6 input channels
    model_channels       = MODEL_CHANNELS,
    out_channels         = IN_CHANNELS,
    num_res_blocks       = NUM_RES_BLOCKS,
    attention_resolutions = ATTENTION_RES,
    dropout              = 0.1,
    channel_mult         = CHANNEL_MULT,
    conv_resample        = True,
    num_heads            = 4,
    num_heads_upsample   = -1,
    use_scale_shift_norm = True,
    num_classes          = None,          # no class conditioning
).to(device)

# %%
pth_path = os.path.join(os.path.dirname(__file__), "../2026-06-05/16-41-56_cond_sr_div2k_x4/final_checkpoint.pt")
state_dict = torch.load(pth_path, map_location=device)['final_model']
model.load_state_dict(state_dict)

model.eval()

# %%
N_ODE_STEPS = 100

lr_batch = fixed_lr[:N_EVAL_IMGS].to(device)
hr_batch = fixed_hr[:N_EVAL_IMGS].to(device)


with torch.no_grad():
    x_init = lr_batch + EPSILON * torch.randn_like(lr_batch)
    vf     = ConditionalVectorField(model, x0_cond=lr_batch, y=None)
    t_span = torch.linspace(0, 1, N_ODE_STEPS, device=device)
    traj   = odeint(vf, x_init, t_span, method='euler') # 'euler' / 'heun'

plot_comparison(lr_batch, traj[-1], hr_batch, epoch=13,
                show=False, n_img=N_EVAL_IMGS, save_prefix=f"{VARIANT}_final", output_dir=run_dir)

# %%



