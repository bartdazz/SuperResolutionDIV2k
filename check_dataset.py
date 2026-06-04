"""
Run this script to visually inspect DIV2K image pairs and dataset patches.
Usage: python check_dataset.py
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

from Model.super_res_dataset import DIV2KDataset

DATA_ROOT  = os.path.join(os.path.dirname(__file__), "data")
HR_DIR     = os.path.join(DATA_ROOT, "DIV2K_valid_HR")
LR_DIR     = os.path.join(DATA_ROOT, "DIV2K_valid_LR_bicubic", "X2")
HR_PATCH   = 128
SCALE      = 2


def denorm(t: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor (3,H,W) → uint8 numpy (H,W,3) for imshow."""
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── 1. Raw image pair (no patch, no normalisation) ───────────────────────────
print("Section 1: raw HR and LR image sizes")
hr_path = os.path.join(HR_DIR, "0850.png")
lr_path = os.path.join(LR_DIR, "0850x2.png")
hr_img  = Image.open(hr_path).convert("RGB")
lr_img  = Image.open(lr_path).convert("RGB")
print(f"  HR: {hr_img.size}   LR: {lr_img.size}   ratio: {hr_img.size[0]/lr_img.size[0]:.1f}×")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(hr_img)
axes[0].set_title(f"HR  {hr_img.size[0]}×{hr_img.size[1]}")
axes[0].axis("off")
axes[1].imshow(lr_img)
axes[1].set_title(f"LR (2× downsampled)  {lr_img.size[0]}×{lr_img.size[1]}")
axes[1].axis("off")
plt.suptitle("Full image pair — image 0850")
plt.tight_layout()
plt.savefig("check_full_image.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: check_full_image.png")


# ── 2. Dataset patches: HR, LR upsampled, difference ─────────────────────────
print("\nSection 2: dataset patches")
ds = DIV2KDataset(HR_DIR, LR_DIR, hr_patch_size=HR_PATCH, scale=SCALE, train=True)
print(f"  Dataset length: {len(ds)}")

lr_up, hr, _ = ds[0]
print(f"  lr_up shape: {lr_up.shape}  hr shape: {hr.shape}")
print(f"  lr_up range: [{lr_up.min():.2f}, {lr_up.max():.2f}]")
print(f"  hr    range: [{hr.min():.2f}, {hr.max():.2f}]")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    lr_up, hr, _ = ds[i * 10]
    diff = (hr - lr_up).abs()

    axes[0, i].imshow(denorm(hr))
    axes[0, i].set_title(f"HR patch {i}")
    axes[0, i].axis("off")

    axes[1, i].imshow(denorm(lr_up))
    axes[1, i].set_title(f"LR (bicubic up) {i}")
    axes[1, i].axis("off")

plt.suptitle(f"Top: HR patches ({HR_PATCH}×{HR_PATCH}) — Bottom: LR upsampled to same size")
plt.tight_layout()
plt.savefig("check_patches.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: check_patches.png")


# ── 3. Zoom-in comparison: HR vs LR upsampled ─────────────────────────────────
print("\nSection 3: zoom-in crop comparison")
lr_up, hr, _ = ds[5]
hr_np  = denorm(hr)
lr_np  = denorm(lr_up)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(hr_np)
axes[0].set_title("HR patch (ground truth)")
axes[0].axis("off")
axes[1].imshow(lr_np)
axes[1].set_title("LR upsampled (model input)")
axes[1].axis("off")
diff = np.abs(hr_np.astype(float) - lr_np.astype(float)).astype(np.uint8)
axes[2].imshow(diff)
axes[2].set_title("Absolute difference (what model must recover)")
axes[2].axis("off")
plt.tight_layout()
plt.savefig("check_diff.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: check_diff.png")


# ── 4. Quick dataloader test ──────────────────────────────────────────────────
print("\nSection 4: dataloader batch")
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
lr_batch, hr_batch, _ = next(iter(loader))
print(f"  lr_batch: {lr_batch.shape}  hr_batch: {hr_batch.shape}")
print("  All checks passed.")
