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


def _auto_zoom_box(img_tensor, crop_size, avoid=None):
    """Return (r0, c0) of the highest-variance crop of shape (crop_size, crop_size).

    Args:
        avoid: optional (r0, c0, h, w) region to exclude from search
               (used to ensure two crops don't overlap).
    """
    img = unnormalize_img(img_tensor)
    H, W = img.shape[:2]
    ch = cw = crop_size
    step = max(8, crop_size // 4)
    best_var, best_r, best_c = -1.0, 0, 0
    for r in range(0, H - ch + 1, step):
        for c in range(0, W - cw + 1, step):
            if avoid is not None:
                ar, ac, ah, aw = avoid
                overlaps = not (r + ch <= ar or r >= ar + ah or
                                c + cw <= ac or c >= ac + aw)
                if overlaps:
                    continue
            v = img[r:r + ch, c:c + cw].var()
            if v > best_var:
                best_var, best_r, best_c = v, r, c
    return best_r, best_c


def plot_sr_patch(
    full_lr,
    patches,
    scale_factor=4,
    show=False,
    save_prefix="SuperRes", output_dir=".",
):
    """SR result plot with one or two crop regions.

    Args:
        full_lr: (3, H, W) full original LR image tensor in [-1, 1]
        patches: list of (lr_crop, sr_crop, crop_box, color) where
                   lr_crop   — (3, ch, cw) LR patch
                   sr_crop   — (3, ch*s, cw*s) SR output
                   crop_box  — (r0, c0, ch, cw) in full_lr pixel coordinates
                   color     — rectangle / border colour string

    Layout (n = len(patches)):
        [ Original ]  [ Zoom 1 ]  [ SR 1 ]
                      [ Zoom 2 ]  [ SR 2 ]   ← only when n == 2
    Labels below each panel. Aspect ratios are always preserved.
    """
    from matplotlib.gridspec import GridSpec

    n        = len(patches)
    full_img = unnormalize_img(full_lr)
    H, W     = full_img.shape[:2]

    # ── Sizing ────────────────────────────────────────────────────────────
    # Each crop row is h_row inches tall; crop panels are square.
    h_row  = 2.6
    h_fig  = n * h_row
    w_crop = h_row              # square crop panels
    # Full image: natural aspect but capped so it doesn't dwarf the crops
    w_full = min((W / H) * h_fig, h_fig * 1.6)

    fig = plt.figure(figsize=(w_full + 2 * w_crop + 0.3, h_fig + 0.55))
    gs  = GridSpec(
        n, 3, figure=fig,
        width_ratios=[w_full, w_crop, w_crop],
        height_ratios=[1] * n,
        wspace=0.04,
        hspace=0.12,
        left=0.01, right=0.99,
        top=0.88,  bottom=0.08,
    )

    ax_full = fig.add_subplot(gs[:, 0])   # spans all rows

    for row, (lr_crop, sr_crop, crop_box, color) in enumerate(patches):
        r0, c0, ch, cw = crop_box
        lr_img = unnormalize_img(lr_crop)
        sr_img = unnormalize_img(sr_crop)

        # Rectangle on full image
        lw = max(2, int(W / 250))
        ax_full.add_patch(plt.Rectangle(
            (c0, r0), cw, ch,
            linewidth=lw, edgecolor=color, facecolor='none',
        ))

        # Zoom panel
        ax_z = fig.add_subplot(gs[row, 1])
        ax_z.imshow(lr_img, interpolation='nearest')
        ax_z.axis('off')
        for sp in ax_z.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3); sp.set_visible(True)
        if row == 0:
            ax_z.set_title('Zoom', fontsize=11, fontweight='bold', pad=4)

        # SR panel
        ax_s = fig.add_subplot(gs[row, 2])
        ax_s.imshow(sr_img, interpolation='nearest')
        ax_s.axis('off')
        for sp in ax_s.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3); sp.set_visible(True)
        if row == 0:
            ax_s.set_title('SR', fontsize=11, fontweight='bold', pad=4)

    # Full image (drawn last so rectangles are on top)
    ax_full.imshow(full_img, interpolation='nearest')
    ax_full.axis('off')
    ax_full.set_title('Original', fontsize=11, fontweight='bold', pad=4)

    fig.suptitle(save_prefix, fontsize=13, fontweight='bold', y=0.97)
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
