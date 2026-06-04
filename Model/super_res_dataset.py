import os
import random
import glob
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
from typing import Tuple


class DIV2KDataset(Dataset):
    """DIV2K super-resolution dataset using pre-downsampled LR images.

    Returns (lr_upsampled, hr, 0) triples where both tensors are
    (3, hr_patch_size, hr_patch_size) in [-1, 1].

    Args:
        hr_dir:        path to DIV2K_train_HR/ or DIV2K_valid_HR/
        lr_dir:        path to DIV2K_train_LR_bicubic/X2/ or DIV2K_valid_LR_bicubic/X2/
        hr_patch_size: side length of the HR crop (must be divisible by scale)
        scale:         SR scale factor (2 for DIV2K 2x bicubic)
        seed:          if set, crop coordinates and augmentation are deterministic
                       per image index — use for fixed eval patches
    """
    def __init__(self, hr_dir: str, lr_dir: str,
                 hr_patch_size: int = 128, scale: int = 2, seed: int = None):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        assert len(self.hr_paths) == len(self.lr_paths), \
            f"HR/LR count mismatch: {len(self.hr_paths)} vs {len(self.lr_paths)}"
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.scale = scale
        self.seed = seed
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        lr = Image.open(self.lr_paths[idx]).convert("RGB")

        lw, lh = lr.size   # PIL uses (width, height)
        lp = self.lr_patch_size

        # use a per-index RNG when seed is set, otherwise use the global random state
        rng = random.Random(self.seed + idx) if self.seed is not None else random

        x = rng.randint(0, lw - lp)
        y = rng.randint(0, lh - lp)

        lr_patch = lr.crop((x, y, x + lp, y + lp))
        hr_patch = hr.crop((x * self.scale, y * self.scale,
                            (x + lp) * self.scale, (y + lp) * self.scale))

        # augmentation only when not seeded (training mode)
        if self.seed is None:
            if rng.random() < 0.5:
                lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            k = rng.randint(0, 3)
            if k > 0:
                lr_patch = lr_patch.rotate(90 * k)
                hr_patch = hr_patch.rotate(90 * k)

        lr_t = self.normalize(self.to_tensor(lr_patch))   # (3, lp, lp)
        hr_t = self.normalize(self.to_tensor(hr_patch))   # (3, hp, hp)

        # upsample LR to HR size — matches condition_on_lr=True expectation
        lr_up = F.interpolate(lr_t.unsqueeze(0),
                              size=(self.hr_patch_size, self.hr_patch_size),
                              mode="bicubic", align_corners=False).squeeze(0)

        return lr_up, hr_t, torch.tensor(0)


def build_DIV2K(
    hr_dir: str,
    lr_dir: str,
    hr_patch_size: int = 128,
    scale: int = 2,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    """Build a training DataLoader for DIV2K super-resolution.

    Each batch yields (lr_upsampled, hr, 0) of shape (B, 3, hr_patch_size, hr_patch_size).
    For fixed eval patches use DIV2KDataset directly with seed set:

        eval_ds = DIV2KDataset(hr_dir, lr_dir, hr_patch_size=128, scale=2, seed=0)
        fixed_lr, fixed_hr, _ = next(iter(DataLoader(eval_ds, batch_size=N)))

    Args:
        hr_dir:        path to DIV2K_train_HR/
        lr_dir:        path to DIV2K_train_LR_bicubic/X2/
        hr_patch_size: HR crop size (LR crop = hr_patch_size // scale)
        scale:         SR upscaling factor
        batch_size:    samples per batch
        num_workers:   DataLoader worker processes
    """
    train_ds = DIV2KDataset(hr_dir, lr_dir,
                            hr_patch_size=hr_patch_size, scale=scale, seed=None)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)
