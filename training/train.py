"""
Variant 2 — Conditional super-resolution (main experiment).

Data-dependent coupling (x0 = bicubic-blurry + noise) AND the upsampled
low-res image ξ is appended to x_t in the channel dimension at every ODE
step, following Albergo et al. (2023) Appendix B.
"""
import sys
import os
import json
import copy
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import math

from Model.super_res_unet    import SuperResUNet
from Model.loss              import DataDependentLoss
from Model.super_res_dataset import build_DIV2K, DIV2KDataset
from Model.inference         import ConditionalVectorField
from Model.utils             import plot_comparison, ema
from torch.utils.data        import DataLoader

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 0
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# ── Hyperparameters ───────────────────────────────────────────────────────────
_ROOT           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCALE_FACTOR    = 2
HR_DIR          = os.path.join(_ROOT, "data", "DIV2K_train_HR")
LR_DIR          = os.path.join(_ROOT, "data", "DIV2K_train_LR_bicubic", "X" + str(SCALE_FACTOR))
HR_VALID_DIR    = os.path.join(_ROOT, "data", "DIV2K_valid_HR")
LR_VALID_DIR    = os.path.join(_ROOT, "data", "DIV2K_valid_LR_bicubic", "X" + str(SCALE_FACTOR))
HR_PATCH        = 128
IN_CHANNELS     = 3
MODEL_CHANNELS  = 128
NUM_RES_BLOCKS  = 2
CHANNEL_MULT    = (1, 2, 2, 2)
ATTENTION_RES   = [2, 4]
DROPOUT         = 0.0
BATCH_SIZE      = 16
EPSILON         = 0.05
NUM_EPOCHS      = 800
LR              = 1e-4
LR_END          = 1e-8
POLY_POWER      = 1.0
WARMUP_EPOCHS   = 20
EMA_DECAY       = 0.9999
OT_EPS          = 1e-6
N_ODE_STEPS     = 100
N_EVAL_IMGS     = 8
VARIANT         = "cond_sr_div2k_x" + str(SCALE_FACTOR)
RESUME_PATH     = None
DESCRIPTION     = ""


def setup_run_dir(variant: str, config: dict, description: str) -> str:
    timestamp   = os.environ.get("RUN_TIMESTAMP") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    description = os.environ.get("RUN_DESCRIPTION") or description
    date, time = timestamp.split("_", 1)
    run_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs', date, f"{time}_{variant}"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "description.txt"), "w") as f:
        f.write(description + "\n")
    return run_dir

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader = build_DIV2K(
    hr_dir=HR_DIR,
    lr_dir=LR_DIR,
    hr_patch_size=HR_PATCH,
    scale=SCALE_FACTOR,
    batch_size=BATCH_SIZE,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = SuperResUNet(
    in_channels          = IN_CHANNELS,
    condition_on_lr      = True,          # x0 concatenated with x_t → 6 input channels
    model_channels       = MODEL_CHANNELS,
    out_channels         = IN_CHANNELS,
    num_res_blocks       = NUM_RES_BLOCKS,
    attention_resolutions = ATTENTION_RES,
    dropout              = DROPOUT,
    channel_mult         = CHANNEL_MULT,
    conv_resample        = True,
    num_heads            = 4,
    num_heads_upsample   = -1,
    use_scale_shift_norm = True,
    num_classes          = None,          # no class conditioning
).to(device)
ema_model = copy.deepcopy(model)

# setting up the learning rate scheduler
steps_per_epoch = math.ceil(len(train_loader.dataset) / BATCH_SIZE)
total_steps  = NUM_EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch

# Polynomial decay with linear warmup, matching the paper (Table 3 / Appendix E.2).
# Warmup:  lr linearly rises from LR_END to LR over warmup_steps.
# Decay:   lr follows polynomial decay from LR down to LR_END over the remaining steps.
# LambdaLR multiplies the base lr (=LR) by the returned scalar, so we express
# everything as a fraction of LR.
_lr_ratio = LR_END / LR   # fraction of peak lr at the floor

def poly_warmup_lr(step):
    if warmup_steps > 0 and step < warmup_steps:
        # linear ramp: LR_END/LR → 1.0
        return _lr_ratio + (step / warmup_steps) * (1.0 - _lr_ratio)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress    = (step - warmup_steps) / decay_steps
    progress    = min(progress, 1.0)
    # polynomial decay: 1.0 → LR_END/LR
    return (1.0 - _lr_ratio) * (1.0 - progress) ** POLY_POWER + _lr_ratio

# optimization
loss_fn   = DataDependentLoss(epsilon=EPSILON)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
sched     = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_warmup_lr)

# show model size
model_size = 0
for param in model.parameters():
    model_size += param.data.nelement()
print("Model params: %.2f M" % (model_size / 1e6))

# saving config
config = dict(
    resume=True if RESUME_PATH else False,
    scale_factor=SCALE_FACTOR, hr_patch=HR_PATCH,
    model_channels=MODEL_CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
    channel_mult=list(CHANNEL_MULT), attention_res=ATTENTION_RES,
    dropout=DROPOUT, batch_size=BATCH_SIZE, epsilon=EPSILON,
    num_epochs=NUM_EPOCHS, lr=LR, lr_end=LR_END, poly_power=POLY_POWER,
    warmup_epochs=WARMUP_EPOCHS,
    ema_decay=EMA_DECAY, ot_eps=OT_EPS, n_ode_steps=N_ODE_STEPS,
    condition_on_lr=model.condition_on_lr, num_classes=model.num_classes,
)
run_dir = setup_run_dir(VARIANT, config, DESCRIPTION)
print(f"Run directory: {run_dir}")

# ── Fixed eval batch (seeded → same patches every run) ───────────────────────
eval_ds  = DIV2KDataset(HR_VALID_DIR, LR_VALID_DIR, hr_patch_size=HR_PATCH, scale=SCALE_FACTOR, seed=0)
fixed_lr, fixed_hr, _ = next(iter(DataLoader(eval_ds, batch_size=N_EVAL_IMGS, shuffle=False, num_workers=0)))
fixed_lr = fixed_lr.to(device)
fixed_hr = fixed_hr.to(device)

#── RESUME ─────────────────────────────────────────────────────────────


if RESUME_PATH and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(checkpoint["final_model"])
    ema_model.load_state_dict(checkpoint["final_ema_model"])
    optimizer.load_state_dict(checkpoint["optim"])
    sched.load_state_dict(checkpoint["sched"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")
else:
    start_epoch = 0
    print("Starting fresh")

# ── Training ──────────────────────────────────────────────────────────────────
train_losses = []

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for lr_imgs, hr_imgs, _ in train_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, lr_imgs, hr_imgs, ot_eps=OT_EPS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        sched.step()
        ema(model, ema_model, EMA_DECAY)
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    print(f'| epoch {epoch:4d} | train {epoch_loss:.6f}')

    if epoch in [200, 400, 600, 800]:
        torch.save({
            "final_model":     model.state_dict(),
            "final_ema_model": ema_model.state_dict(),
            "optim":           optimizer.state_dict(),
            "sched":           sched.state_dict(),
            "epoch":           epoch,
        }, os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt"))

    if epoch % 25 == 0:
        ema_model.eval()
        with torch.no_grad():
            x_init = fixed_lr + EPSILON * torch.randn_like(fixed_lr)
            vf     = ConditionalVectorField(ema_model, x0_cond=fixed_lr, y=None)
            t_span = torch.linspace(0, 1, 25, device=device)
            traj   = odeint(vf, x_init, t_span, method='euler')
        plot_comparison(fixed_lr, traj[-1], fixed_hr, epoch,
                        show=False, n_img=N_EVAL_IMGS, save_prefix=VARIANT, output_dir=run_dir)


torch.save({
            "final_model": model.state_dict(),
            "final_ema_model": ema_model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
        }, os.path.join(run_dir, f"final_checkpoint.pt"))

# ── Loss curves ───────────────────────────────────────────────────────────────
plt.figure()
plt.plot(train_losses, label='train')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(os.path.join(run_dir, "loss_curve.png"), bbox_inches='tight', dpi=300)
plt.close()

# ── Final sample ──────────────────────────────────────────────────────────────
ema_model.load_state_dict(torch.load(os.path.join(run_dir, "final_checkpoint.pt"), map_location=device)["final_ema_model"])
ema_model.eval()
with torch.no_grad():
    x_init = fixed_lr + EPSILON * torch.randn_like(fixed_lr)
    vf     = ConditionalVectorField(ema_model, x0_cond=fixed_lr, y=None)
    t_span = torch.linspace(0, 1, N_ODE_STEPS, device=device)
    traj   = odeint(vf, x_init, t_span, method='euler')

plot_comparison(fixed_lr, traj[-1], fixed_hr, epoch=NUM_EPOCHS,
                show=False, n_img=N_EVAL_IMGS, save_prefix=f"{VARIANT}_final", output_dir=run_dir)
