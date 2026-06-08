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

    Layout — all panels the same height, side by side:
        [ Original + rectangle ]  [ Zoom ]  [ SR ]

    Labels sit below each panel. Zoom and SR have a coloured border that
    matches the rectangle drawn on the original image.
    """
    r0, c0, ch, cw = crop_box_lr
    full_img = unnormalize_img(full_lr)
    lr_img   = unnormalize_img(lr_crop)
    sr_img   = unnormalize_img(sr_crop)
    H, W     = full_img.shape[:2]

    # ── Figure sizing ─────────────────────────────────────────────────────
    # Fix figure height; derive widths from each panel's natural aspect ratio.
    h_fig    = 4.0                          # inches
    w_full   = (W / H) * h_fig * 0.75      # original image: 75% of natural width
    w_crop   = (cw / ch) * h_fig * 0.85    # crop panels: 85% of crop aspect
    label_h  = 0.35                         # inches below each panel for label
    fig_h    = h_fig + label_h + 0.4        # +0.4 for suptitle

    fig = plt.figure(figsize=(w_full + 2 * w_crop + 0.3, fig_h))

    # Manual axes: place them so their tops align and heights are equal
    top    = 1.0 - (0.4 / fig_h)           # normalised y of panel top
    height = h_fig / fig_h                  # normalised panel height
    gap    = 0.02                           # normalised gap between panels
    total_w = w_full + 2 * w_crop + 0.3
    x0      = 0.01
    wn_full = w_full  / total_w
    wn_crop = w_crop  / total_w

    ax_full = fig.add_axes([x0,
                            top - height,
                            wn_full - gap,
                            height])
    ax_zoom = fig.add_axes([x0 + wn_full + gap,
                            top - height,
                            wn_crop - gap,
                            height])
    ax_sr   = fig.add_axes([x0 + wn_full + wn_crop + 2 * gap,
                            top - height,
                            wn_crop - gap,
                            height])

    # ── Original image ────────────────────────────────────────────────────
    ax_full.imshow(full_img, interpolation='nearest', aspect='auto')
    ax_full.add_patch(plt.Rectangle(
        (c0, r0), cw, ch,
        linewidth=max(2, int(W / 200)), edgecolor=box_color, facecolor='none',
    ))
    ax_full.axis('off')
    ax_full.text(0.5, -0.03, 'Original', transform=ax_full.transAxes,
                 ha='center', va='top', fontsize=11, fontweight='bold')

    # ── Zoom (LR crop) ────────────────────────────────────────────────────
    ax_zoom.imshow(lr_img, interpolation='nearest', aspect='auto')
    ax_zoom.axis('off')
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(box_color); spine.set_linewidth(3); spine.set_visible(True)
    ax_zoom.text(0.5, -0.03, 'Zoom', transform=ax_zoom.transAxes,
                 ha='center', va='top', fontsize=11, fontweight='bold')

    # ── SR output ─────────────────────────────────────────────────────────
    ax_sr.imshow(sr_img, interpolation='nearest', aspect='auto')
    ax_sr.axis('off')
    for spine in ax_sr.spines.values():
        spine.set_edgecolor(box_color); spine.set_linewidth(3); spine.set_visible(True)
    ax_sr.text(0.5, -0.03, 'SR', transform=ax_sr.transAxes,
               ha='center', va='top', fontsize=11, fontweight='bold')

    fig.suptitle(save_prefix, fontsize=13, fontweight='bold', y=0.98)
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
