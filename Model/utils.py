import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def show_cifar_image(img_tensor):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    img  = img_tensor.permute(1, 2, 0).numpy()
    img  = np.clip(std * img + mean, 0, 1)
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')


def unnormalize_img(img_tensor):
    # DIV2K is normalised with mean=0.5, std=0.5 → inverse: x*0.5 + 0.5
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * 0.5 + 0.5, 0, 1)


def plot_comparison(
    low_r, gen, high_r, epoch,
    show=False, n_img=10,
    save_prefix="SuperRes", output_dir=".",
    col_titles=("Low Res", "Generated", "High Res"),
):
    n_img = min(n_img, low_r.size(0))

    fig, axes = plt.subplots(
        n_img, 3,
        figsize=(3 * 2.5, n_img * 2.5),
        gridspec_kw={'hspace': 0.04, 'wspace': 0.04},
    )
    if n_img == 1:
        axes = axes[np.newaxis, :]

    for col, (title, batch) in enumerate(zip(col_titles, [low_r, gen, high_r])):
        for row in range(n_img):
            axes[row, col].imshow(unnormalize_img(batch[row]), interpolation='nearest')
            axes[row, col].axis('off')
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', pad=6)

    fig.suptitle(f"{save_prefix} — epoch {epoch}", fontsize=13, fontweight='bold', y=1.01)
    fig.savefig(
        os.path.join(output_dir, f"{save_prefix}_epoch_{str(epoch).zfill(3)}.png"),
        bbox_inches='tight', dpi=150,
    )
    if show:
        plt.show()
    plt.close(fig)


def _auto_zoom_box(img_tensor, crop_frac=0.25):
    """Return (r0, c0, zh, zw) of the highest-variance crop in img_tensor."""
    img = unnormalize_img(img_tensor)   # (H, W, 3)
    H, W = img.shape[:2]
    zh = max(32, int(H * crop_frac))
    zw = max(32, int(W * crop_frac))
    step = max(8, min(zh, zw) // 4)
    best_var, best_r, best_c = -1.0, 0, 0
    for r in range(0, H - zh + 1, step):
        for c in range(0, W - zw + 1, step):
            v = img[r:r + zh, c:c + zw].var()
            if v > best_var:
                best_var, best_r, best_c = v, r, c
    return best_r, best_c, zh, zw


def plot_comparison_zoom(
    low_r, gen, epoch,
    high_r=None,
    zoom_box=None,          # (r0, c0, h, w) in pixels; None = auto per image
    crop_frac=0.25,         # fraction of image used for zoom when zoom_box is None
    box_color='red',
    zoom_scale=2,           # how much to magnify the zoom patch for display
    show=False, n_img=4,
    save_prefix="SuperRes", output_dir=".",
):
    """Plot SR results with a zoomed inset crop for each image.

    Layout (per sample):
        top row   — full image(s) with a coloured rectangle marking the crop
        bottom row — zoomed crop, magnified by zoom_scale for easy comparison

    Args:
        low_r:      LR batch  (B, C, H, W) in [-1, 1]
        gen:        SR batch  (B, C, H, W) in [-1, 1]
        high_r:     HR batch  (B, C, H, W) in [-1, 1], or None
        zoom_box:   fixed (r0, c0, h, w) applied to every image;
                    pass None to auto-select per image (highest variance region)
        crop_frac:  size of auto crop as fraction of image dimensions
        zoom_scale: integer upscale applied to the crop patch for display
    """
    batches = [low_r, gen] + ([high_r] if high_r is not None else [])
    titles  = ['Low Res', 'Generated'] + (['High Res'] if high_r is not None else [])
    n_cols  = len(batches)
    n_img   = min(n_img, low_r.size(0))

    # Each sample occupies 2 matplotlib rows (full + zoom)
    fig, axes = plt.subplots(
        n_img * 2, n_cols,
        figsize=(n_cols * 2.8, n_img * 5.6),
        gridspec_kw={'hspace': 0.06, 'wspace': 0.04},
    )
    # Normalise axes shape to (n_img*2, n_cols)
    if n_img * 2 == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_img * 2 == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for img_idx in range(n_img):
        full_row = img_idx * 2
        zoom_row = img_idx * 2 + 1

        # Determine zoom box for this image (use LR as reference)
        if zoom_box is None:
            r0, c0, zh, zw = _auto_zoom_box(low_r[img_idx], crop_frac)
        else:
            r0, c0, zh, zw = zoom_box

        for col_idx, (title, batch) in enumerate(zip(titles, batches)):
            img = unnormalize_img(batch[img_idx])   # (H, W, 3)

            # ── Full image with rectangle ──────────────────────────────────
            ax = axes[full_row, col_idx]
            ax.imshow(img, interpolation='nearest')
            rect = plt.Rectangle(
                (c0, r0), zw, zh,
                linewidth=2, edgecolor=box_color, facecolor='none',
            )
            ax.add_patch(rect)
            ax.axis('off')
            if img_idx == 0:
                ax.set_title(title, fontsize=12, fontweight='bold', pad=5)

            # ── Zoom patch ────────────────────────────────────────────────
            ax_z = axes[zoom_row, col_idx]
            crop = img[r0:r0 + zh, c0:c0 + zw]     # (zh, zw, 3)
            # Nearest-neighbour upscale so pixel structure stays crisp
            crop_big = np.repeat(np.repeat(crop, zoom_scale, axis=0),
                                 zoom_scale, axis=1)
            ax_z.imshow(crop_big, interpolation='nearest')
            ax_z.axis('off')
            # Thin border matching the rectangle colour
            for spine in ax_z.spines.values():
                spine.set_edgecolor(box_color)
                spine.set_linewidth(1.5)
                spine.set_visible(True)

    fig.suptitle(f"{save_prefix} — epoch {epoch}", fontsize=13,
                 fontweight='bold', y=1.005)
    fig.savefig(
        os.path.join(output_dir, f"{save_prefix}_zoom_epoch_{str(epoch).zfill(3)}.png"),
        bbox_inches='tight', dpi=200,
    )
    if show:
        plt.show()
    plt.close(fig)



def ema(source: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
