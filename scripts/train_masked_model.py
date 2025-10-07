#!/usr/bin/env python
"""Train masked time-channel autoencoder on EEG windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import tomllib
import torch
from torch.cuda.amp import GradScaler, autocast

from cerebro.logging_utils import create_progress, get_console
from cerebro.pretraining import create_masked_modeling_dataloader
from fm.models.masked import EEGMaskedAutoencoder
from fm.tasks import masked_mse_loss

DEFAULT_CONFIG_PATH = Path("fm/config/masked_autoencoder.toml")
console = get_console()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config (default: fm/config/masked_autoencoder.toml)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--releases", nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--dataset-variant", choices=["mini", "full"], default=None)
    parser.add_argument("--time-mask", type=float, default=None)
    parser.add_argument("--channel-mask", type=float, default=None)
    parser.add_argument("--mask-value", type=float, default=None)
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        return tomllib.load(fh)


def merge_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = json.loads(json.dumps(config))
    train_cfg = merged.setdefault("training", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.output_dir is not None:
        train_cfg["output_dir"] = str(args.output_dir)
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.val_fraction is not None:
        train_cfg["val_fraction"] = args.val_fraction
    if args.dataset_variant is not None:
        train_cfg["dataset_variant"] = args.dataset_variant
    if args.time_mask is not None:
        train_cfg["time_mask_fraction"] = args.time_mask
    if args.channel_mask is not None:
        train_cfg["channel_mask_fraction"] = args.channel_mask
    if args.mask_value is not None:
        train_cfg["mask_value"] = args.mask_value
    if args.tasks:
        merged.setdefault("tasks", {})["names"] = list(args.tasks)
    if args.releases:
        merged.setdefault("releases", {})["names"] = list(args.releases)
    return merged


def resolve_device(device_str: str | None) -> torch.device:
    if device_str in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_dataloaders(config: Dict[str, Any]):
    train_cfg = config.get("training", {})
    loaders = create_masked_modeling_dataloader(
        tasks=config.get("tasks", {}).get("names", []),
        releases=config.get("releases", {}).get("names", []),
        window_size_s=train_cfg.get("window_size_s", 4.0),
        stride_s=train_cfg.get("stride_s", 2.0),
        crop_size_s=train_cfg.get("crop_size_s", 2.0),
        batch_size=train_cfg.get("batch_size", 128),
        num_workers=train_cfg.get("num_workers", 8),
        seed=train_cfg.get("seed"),
        time_mask_fraction=train_cfg.get("time_mask_fraction", 0.15),
        channel_mask_fraction=train_cfg.get("channel_mask_fraction", 0.25),
        mask_value=train_cfg.get("mask_value", 0.0),
        val_fraction=train_cfg.get("val_fraction"),
        dataset_variant=train_cfg.get("dataset_variant", "mini"),
    )
    if isinstance(loaders, tuple):
        return loaders
    return loaders, None


def train_one_epoch(
    model: EEGMaskedAutoencoder,
    loader,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    log_every: int,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    step_count = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    with create_progress(transient=True) as progress:
        task_id = progress.add_task(f"Epoch {epoch:03d}", total=total_batches)
        for masked, target, mask in loader:
            masked = masked.to(device)
            target = target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(masked)
                loss = masked_mse_loss(pred, target, mask)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.detach().item()
            total_loss += loss_val
            step_count += 1
            progress.advance(task_id)
            if log_every and step_count % log_every == 0:
                console.log(
                    f"Epoch [bold]{epoch:03d}[/bold] | Step {step_count:06d} | Loss {loss_val:.4f}"
                )
    return total_loss / max(step_count, 1)


def evaluate(model: EEGMaskedAutoencoder, loader, device: torch.device, use_amp: bool) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with create_progress(transient=True) as progress:
        task_id = progress.add_task("Validation", total=len(loader) if hasattr(loader, "__len__") else None)
        with torch.no_grad():
            for masked, target, mask in loader:
                masked = masked.to(device)
                target = target.to(device)
                mask = mask.to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    pred = model(masked)
                    loss = masked_mse_loss(pred, target, mask)
                total_loss += loss.item()
                n_batches += 1
                progress.advance(task_id)
    model.train()
    return total_loss / max(n_batches, 1)


def checkpoint_payload(model, optimizer, epoch: int, global_step: int) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def main() -> None:
    args = parse_args()
    config = merge_overrides(load_config(args.config), args)
    train_cfg = config.setdefault("training", {})

    device = resolve_device(train_cfg.get("device"))
    torch.manual_seed(train_cfg.get("seed", 0))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.get("seed", 0))

    console.rule("Masked Autoencoder Training")
    console.print(f"Device: [bold]{device}[/bold]")
    console.print(f"Tasks: [cyan]{', '.join(config.get('tasks', {}).get('names', []))}[/cyan]")
    console.print(f"Releases: [cyan]{', '.join(config.get('releases', {}).get('names', []))}[/cyan]")
    console.print(f"Variant: [cyan]{train_cfg.get('dataset_variant', 'mini')}[/cyan]")

    train_loader, val_loader = create_dataloaders(config)

    model = EEGMaskedAutoencoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    use_amp = bool(train_cfg.get("use_amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    epochs = int(train_cfg.get("epochs", 10))
    log_every = int(train_cfg.get("log_every_steps", 25))
    grad_clip = float(train_cfg.get("clip_grad_norm", 0.0))
    max_steps = train_cfg.get("max_steps")
    if args.max_steps is not None:
        max_steps = args.max_steps

    output_dir = Path(train_cfg.get("output_dir", "weights/masked_ae"))
    if args.output_dir is not None:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    console.log(f"Resolved config written to {config_path}")

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            log_every,
            use_amp,
            grad_clip,
            epoch,
        )
        console.print(f"Epoch [bold]{epoch:03d}[/bold] | train loss {avg_loss:.4f}")
        global_step += len(train_loader) if hasattr(train_loader, "__len__") else 0

        val_loss = evaluate(model, val_loader, device, use_amp)
        if val_loader is not None:
            console.print(f"  val loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    checkpoint_payload(model, optimizer, epoch, global_step),
                    output_dir / "checkpoint_best.pt",
                )
                console.log("Saved checkpoint_best.pt")

        state = checkpoint_payload(model, optimizer, epoch, global_step)
        if epoch % max(int(train_cfg.get("save_every_epochs", 1)), 1) == 0:
            ckpt = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(state, ckpt)
            torch.save(state, output_dir / "latest.pt")
            console.log(f"Saved checkpoint to {ckpt}")

        if max_steps is not None and global_step >= max_steps:
            console.print("[yellow]Max steps reached, stopping training.[/yellow]")
            break

    console.print("[bold green]Masked modeling training completed.[/bold green]")


if __name__ == "__main__":
    main()
