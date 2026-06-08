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


def _auto_zoom_box(img_tensor, crop_size):
    """Return (r0, c0) of the highest-variance crop of shape (crop_size, crop_size)."""
    img = unnormalize_img(img_tensor)   # (H, W, 3)
    H, W = img.shape[:2]
    ch = cw = crop_size
    step = max(8, crop_size // 4)
    best_var, best_r, best_c = -1.0, 0, 0
    for r in range(0, H - ch + 1, step):
        for c in range(0, W - cw + 1, step):
            v = img[r:r + ch, c:c + cw].var()
            if v > best_var:
                best_var, best_r, best_c = v, r, c
    return best_r, best_c


def plot_sr_patch(
    full_lr,        # (3, H, W) tensor — full original LR image, [-1, 1]
    lr_crop,        # (3, ch, cw) tensor — LR patch at native LR resolution
    sr_crop,        # (3, ch*s, cw*s) tensor — SR model output
    crop_box_lr,    # (r0, c0, ch, cw) location of crop inside full_lr
    scale_factor=4,
    box_color='red',
    show=False,
    save_prefix="SuperRes", output_dir=".",
):
    """Three-panel SR result plot (no HR reference needed).

    Layout:
        [Full original LR + rectangle]  |  [ LR Crop ]
                                           [ SR ×s    ]

    The original image sits on the left (slightly smaller than the crop
    column). The LR crop and SR output are stacked vertically on the right,
    both framed with a red border that matches the rectangle on the left.
    """
    from matplotlib.gridspec import GridSpec

    r0, c0, ch, cw = crop_box_lr
    full_img = unnormalize_img(full_lr)    # (H, W, 3)
    lr_img   = unnormalize_img(lr_crop)    # (ch, cw, 3)
    sr_img   = unnormalize_img(sr_crop)    # (ch*s, cw*s, 3)
    H, W     = full_img.shape[:2]

    # ── Figure sizing ─────────────────────────────────────────────────────
    # Right column width in inches; left column is slightly narrower.
    w_crop  = 3.0
    w_full  = w_crop * 1.3                 # original image a bit smaller
    h_fig   = (H / W) * w_full            # height follows the image aspect ratio

    fig = plt.figure(figsize=(w_full + w_crop + 0.2, h_fig))
    gs  = GridSpec(
        2, 2, figure=fig,
        width_ratios  = [w_full, w_crop],
        height_ratios = [1, 1],
        wspace=0.06, hspace=0.08,
    )

    ax_full = fig.add_subplot(gs[:, 0])   # left: spans both rows
    ax_lr   = fig.add_subplot(gs[0, 1])   # right top: LR crop
    ax_sr   = fig.add_subplot(gs[1, 1])   # right bottom: SR output

    # ── Left: full original image ─────────────────────────────────────────
    ax_full.imshow(full_img, interpolation='nearest')
    ax_full.add_patch(plt.Rectangle(
        (c0, r0), cw, ch,
        linewidth=max(1, int(W / 300)), edgecolor=box_color, facecolor='none',
    ))
    ax_full.axis('off')
    ax_full.set_title('Original', fontsize=11, fontweight='bold', pad=4)

    # ── Right top: LR crop ────────────────────────────────────────────────
    ax_lr.imshow(lr_img, interpolation='nearest')
    ax_lr.axis('off')
    for spine in ax_lr.spines.values():
        spine.set_edgecolor(box_color); spine.set_linewidth(2); spine.set_visible(True)
    ax_lr.set_title('LR Crop', fontsize=11, fontweight='bold', pad=4)

    # ── Right bottom: SR output ───────────────────────────────────────────
    ax_sr.imshow(sr_img, interpolation='nearest')
    ax_sr.axis('off')
    for spine in ax_sr.spines.values():
        spine.set_edgecolor(box_color); spine.set_linewidth(2); spine.set_visible(True)
    ax_sr.set_title(f'SR ×{scale_factor}', fontsize=11, fontweight='bold', pad=4)

    fig.suptitle(save_prefix, fontsize=12, fontweight='bold', y=1.02)
    fig.savefig(
        os.path.join(output_dir, f"{save_prefix}_sr_patch.png"),
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
