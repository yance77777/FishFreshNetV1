from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASS_NAMES = ["Highly Fresh", "Fresh", "Not Fresh"]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class MFEDImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        missing = [class_name for class_name in CLASS_NAMES if not (Path(directory) / class_name).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing MFED class folders: {missing}")

        return CLASS_NAMES, {class_name: index for index, class_name in enumerate(CLASS_NAMES)}


def create_split_indices(dataset_size: int, runs: int = 5, seed: int = 42) -> list[dict[str, list[int]]]:
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    split_indices = []

    for run_index in range(runs):
        generator = torch.Generator().manual_seed(seed + run_index)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        split_indices.append(
            {
                "train": indices[:train_size],
                "val": indices[train_size : train_size + val_size],
                "test": indices[train_size + val_size :],
            }
        )

    return split_indices


def create_dataloaders(
    data_dir: Path,
    split_indices: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    train_dataset = MFEDImageFolder(str(data_dir), transform=TRAIN_TRANSFORM)
    eval_dataset = MFEDImageFolder(str(data_dir), transform=EVAL_TRANSFORM)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    return {
        "train": DataLoader(Subset(train_dataset, split_indices["train"]), shuffle=True, **loader_kwargs),
        "val": DataLoader(Subset(eval_dataset, split_indices["val"]), shuffle=False, **loader_kwargs),
        "test": DataLoader(Subset(eval_dataset, split_indices["test"]), shuffle=False, **loader_kwargs),
    }
