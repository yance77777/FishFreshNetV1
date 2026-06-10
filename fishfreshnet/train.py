import argparse
import copy
import random
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .data import CLASS_NAMES, MFEDImageFolder, create_dataloaders, create_split_indices
from .models import build_model

try:
    from thop import profile
except ImportError:
    profile = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FishFreshNetV1 on MFED.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the MFED root directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fishfreshnet_v1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=min(8, os_cpu_count()))
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained weights.")
    return parser.parse_args()


def os_cpu_count() -> int:
    try:
        import os

        return os.cpu_count() or 4
    except Exception:
        return 4


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.amp.autocast("cuda")
    return nullcontext()


def prepare_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return device


def move_inputs(inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs = inputs.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda" and inputs.ndim == 4:
        return inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def save_learning_curves(history: dict[str, list[float]], run_index: int, output_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Validation")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, np.array(history["train_acc"]) * 100.0, marker="o", label="Train")
    axes[1].plot(epochs, np.array(history["val_acc"]) * 100.0, marker="s", label="Validation")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / f"learning_curves_run_{run_index + 1}.png", bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true: list[int], y_pred: list[int], run_index: int, output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0.0,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title(f"FishFreshNetV1 | Run {run_index + 1} Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_run_{run_index + 1}.png", bbox_inches="tight")
    plt.close(fig)


def model_complexity(model: nn.Module, device: torch.device) -> tuple[float | None, float | None]:
    if profile is None:
        return None, None

    was_training = model.training
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    if device.type == "cuda":
        dummy_input = dummy_input.contiguous(memory_format=torch.channels_last)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    model.train(was_training)
    return params / 1e6, flops / 1e9


def evaluate(model: nn.Module, dataloader, device: torch.device, use_amp: bool) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = move_inputs(inputs, device)
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with autocast_context(use_amp):
                outputs = model(inputs)

            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return (
        {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        },
        y_true,
        y_pred,
    )


def train_one_run(args: argparse.Namespace, split_indices: dict[str, list[int]], run_index: int, device: torch.device) -> dict[str, float]:
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        split_indices=split_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        for phase in ["train", "val"]:
            model.train(phase == "train")
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = move_inputs(inputs, device)
                labels = labels.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast_context(use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    predictions = outputs.argmax(dim=1)
                    if phase == "train":
                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (predictions == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val":
                scheduler.step(epoch_acc)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"Run {run_index + 1} | Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} | "
                f"val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}"
            )

    model.load_state_dict(best_weights)
    metrics, y_true, y_pred = evaluate(model, dataloaders["test"], device, use_amp)
    params_m, flops_g = model_complexity(model, device)

    save_learning_curves(history, run_index, args.output_dir)
    save_confusion_matrix(y_true, y_pred, run_index, args.output_dir)
    torch.save(best_weights, args.output_dir / f"best_model_fishfreshnet_v1_run{run_index + 1}.pth")

    row = {
        "Run": run_index + 1,
        "Params (M)": params_m,
        "FLOPs (G)": flops_g,
        **metrics,
    }
    print(
        f"Run {run_index + 1} Test | "
        f"accuracy={metrics['Accuracy']:.4f} precision={metrics['Precision']:.4f} "
        f"recall={metrics['Recall']:.4f} f1={metrics['F1-Score']:.4f}"
    )
    return row


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    set_seed(args.seed)
    sns.set_theme(style="whitegrid", context="paper")
    device = prepare_device()
    dataset = MFEDImageFolder(str(args.data_dir))
    split_cache = create_split_indices(len(dataset), runs=args.runs, seed=args.seed)

    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Runs: {args.runs}, epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.learning_rate}")

    rows = [train_one_run(args, split_indices, run_index, device) for run_index, split_indices in enumerate(split_cache)]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.output_dir / "all_runs_metrics.csv", index=False)

    summary = metrics_df.drop(columns=["Run"]).agg(["mean", "std"]).T
    summary.to_csv(args.output_dir / "summary_metrics.csv")
    print(summary)


if __name__ == "__main__":
    main()
