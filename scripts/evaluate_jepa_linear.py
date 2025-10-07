#!/usr/bin/env python
"""Evaluate JEPA embeddings with a linear probe on CCD response time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

from cerebro.data import prepare_data_pipeline
from cerebro.logging_utils import create_progress, get_console
from cerebro.data import load_chs_info
from cerebro.constants import SFREQ
from fm.models import EEGJEPAModel

console = get_console()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to JEPA checkpoint")
    parser.add_argument(
        "--release",
        type=str,
        default="R5",
        help="Release to evaluate on (default: R5)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="contrastChangeDetection",
        help="Task to evaluate (default: contrastChangeDetection)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (auto/cpu/cuda[:id])",
    )
    parser.add_argument(
        "--dataset-variant",
        choices=["mini", "full"],
        default="mini",
        help="Dataset variant used for channel metadata",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Ridge regularisation strength for the linear probe (default: 1e-3)",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


@torch.no_grad()
def extract_embeddings(model: EEGJEPAModel, loader, device: torch.device):
    model.eval()
    embeds = []
    targets = []
    with create_progress(transient=True) as progress:
        task_id = progress.add_task("Embedding", total=len(loader))
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X, y = batch[0], batch[1]
            else:
                raise ValueError("Unexpected batch structure")
            X = X.to(device).float()
            embed = model(X).cpu()
            embeds.append(embed)
            targets.append(y.view(y.size(0), -1).cpu())
            progress.advance(task_id)
    return torch.cat(embeds), torch.cat(targets)


def fit_ridge(features: torch.Tensor, targets: torch.Tensor, alpha: float) -> torch.Tensor:
    features = torch.cat([features, torch.ones(features.size(0), 1)], dim=1)
    eye = torch.eye(features.size(1)) * alpha
    eye[-1, -1] = 0.0  # do not regularise bias
    lhs = features.T @ features + eye
    rhs = features.T @ targets
    weights = torch.linalg.solve(lhs, rhs)
    return weights


def predict(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    features = torch.cat([features, torch.ones(features.size(0), 1)], dim=1)
    return features @ weights


@torch.no_grad()
def evaluate_split(model: EEGJEPAModel, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    embeds, targets = extract_embeddings(model, loader, device)
    return embeds, targets.squeeze(-1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    console.rule("JEPA Linear Evaluation")
    console.print(f"Device: [bold]{device}[/bold]")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    try:
        chs_info = load_chs_info(
            args.task,
            args.release,
            variant=args.dataset_variant,
        )
    except Exception as exc:
        console.log(f"[yellow]Could not load channel metadata: {exc}")
        chs_info = None

    n_times = int(2.0 * SFREQ)
    model = EEGJEPAModel(n_times=n_times, chs_info=chs_info).to(device)
    model.load_state_dict(checkpoint["model_state"])
    console.log(f"Loaded checkpoint from {args.checkpoint}")

    train_loader, val_loader, test_loader = prepare_data_pipeline(
        task=args.task,
        release=args.release,
        remove_bad_subjects=True,
        batch_size=128,
        num_workers=8,
    )

    train_embeds, train_targets = evaluate_split(model, train_loader, device)
    weights = fit_ridge(train_embeds, train_targets.view(-1, 1), alpha=args.alpha)

    def compute_metrics(embeds: torch.Tensor, targets: torch.Tensor):
        preds = predict(embeds, weights).squeeze()
        diff = preds - targets
        rmse = torch.sqrt(torch.mean(diff**2)).item()
        mae = torch.mean(torch.abs(diff)).item()
        return rmse, mae

    metrics = {}
    val_embeds, val_targets = evaluate_split(model, val_loader, device)
    metrics["val"] = compute_metrics(val_embeds, val_targets)
    test_embeds, test_targets = evaluate_split(model, test_loader, device)
    metrics["test"] = compute_metrics(test_embeds, test_targets)

    console.print("")
    console.print("[bold]Results (RMSE / MAE)[/bold]")
    for split, (rmse, mae) in metrics.items():
        console.print(f"  {split}: RMSE={rmse:.4f}, MAE={mae:.4f}")


if __name__ == "__main__":
    main()
