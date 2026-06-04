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


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def plot_generation_by_class(generated_by_class, epoch, save_prefix="SuperRes", output_dir=".", n_per_class=2):
    """Plot generated images in a grid with one row per CIFAR10 class.

    Args:
        generated_by_class: list of 10 tensors, each (n_per_class, C, H, W).
    """
    n_cols = n_per_class

    fig, axes = plt.subplots(
        10, n_cols,
        figsize=(n_cols * 1.6, 10 * 1.6),
        gridspec_kw={'hspace': 0.12, 'wspace': 0.04},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, (class_name, imgs) in enumerate(zip(CIFAR10_CLASSES, generated_by_class)):
        for col in range(n_cols):
            axes[row, col].imshow(unnormalize_img(imgs[col]), interpolation='nearest')
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(class_name, fontsize=9, rotation=0, labelpad=52, va='center')

    for col in range(n_cols):
        axes[0, col].set_title(f"sample {col + 1}", fontsize=9, pad=4)

    fig.suptitle(f"{save_prefix} — epoch {epoch}", fontsize=12, fontweight='bold', y=1.01)
    fig.savefig(
        os.path.join(output_dir, f"{save_prefix}_epoch_{str(epoch).zfill(3)}.png"),
        bbox_inches='tight', dpi=150,
    )
    plt.close(fig)


def ema(source: torch.nn.Module, target: torch.nn.Module, decay: float) -> None:
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
