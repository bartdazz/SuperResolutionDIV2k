import torch
from torch.utils.data import Dataset, DataLoader


class SuperResDataset(Dataset):
    """Returns (low_res, high_res) pairs for super-resolution training.

    Args:
        base_dataset: original torchvision dataset (ToTensor already applied).
        scale_factor: downsampling factor.
    """
    def __init__(self, base_dataset: Dataset, scale_factor: int) -> None:
        self.base_dataset = base_dataset
        self.scale_factor = scale_factor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        return self.downsample(image), image, label