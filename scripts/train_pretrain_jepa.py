#!/usr/bin/env python
"""Entry point for multitask JEPA-style pretraining."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import tomllib
from torch import autocast
from cerebro.logging_utils import create_progress, get_console

from cerebro.constants import SFREQ
from cerebro.data import load_chs_info
from fm.datamodules import MultiTaskPretrainConfig, MultiTaskPretrainDataModule
from fm.models import EEGJEPAModel
from fm.tasks import jepa_loss

DEFAULT_CONFIG_PATH = Path("fm/config/pretrain_jepa.toml")

console = get_console()


@dataclass
class TrainingState:
    epoch: int
    global_step: int


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config (default: fm/config/pretrain_jepa.toml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Override batch size",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Override task list (space separated)",
    )
    parser.add_argument(
        "--releases",
        nargs="+",
        default=None,
        help="Override release list (space separated)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on total optimizer steps (useful for smoke tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints/logs (overrides config)",
    )
    parser.add_argument(
        "--amp/--no-amp",
        dest="use_amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle mixed precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string override (e.g. cuda, cuda:1, cpu)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Optional fraction of subjects to hold out for a validation contrastive loss."
    )
    parser.add_argument(
        "--dataset-variant",
        choices=["mini", "full"],
        default=None,
        help="Select dataset variant (default: config value, usually mini)",
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(list(argv))


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def merge_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(config))  # deep copy via JSON for nested dicts
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("training", {})["learning_rate"] = args.lr
    if args.tasks:
        cfg.setdefault("tasks", {})["names"] = list(args.tasks)
    if args.releases:
        cfg.setdefault("releases", {})["names"] = list(args.releases)
    if args.use_amp is not None:
        cfg.setdefault("training", {})["use_amp"] = bool(args.use_amp)
    if args.device is not None:
        cfg.setdefault("training", {})["device"] = args.device
    if args.output_dir is not None:
        cfg.setdefault("training", {})["output_dir"] = str(args.output_dir)
    if args.max_steps is not None:
        cfg.setdefault("training", {})["max_steps"] = args.max_steps
    if args.val_fraction is not None:
        cfg.setdefault("training", {})["val_fraction"] = args.val_fraction
    if args.dataset_variant is not None:
        cfg.setdefault("training", {})["dataset_variant"] = args.dataset_variant
    return cfg


def resolve_device(device_str: str | None) -> torch.device:
    if device_str is None or device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(config: Dict[str, Any]):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("wandb is not installed but wandb.enabled is true") from exc

    mode = wandb_cfg.get("mode", "online")
    run = wandb.init(
        project=wandb_cfg.get("project", "eeg2025"),
        entity=wandb_cfg.get("entity") or None,
        name=wandb_cfg.get("run_name"),
        config=config,
        mode=mode,
    )
    return run


def create_dataloader(config: Dict[str, Any]) -> MultiTaskPretrainDataModule:
    training_cfg = config.get("training", {})
    dm_cfg = MultiTaskPretrainConfig(
        tasks=config.get("tasks", {}).get("names", []),
        releases=config.get("releases", {}).get("names", []),
        window_size_s=training_cfg.get("window_size_s", 4.0),
        stride_s=training_cfg.get("stride_s", 2.0),
        crop_size_s=training_cfg.get("crop_size_s", 2.0),
        batch_size=training_cfg.get("batch_size", 128),
        num_workers=training_cfg.get("num_workers", 8),
        seed=training_cfg.get("seed"),
        views=2,
        val_fraction=training_cfg.get("val_fraction"),
        dataset_variant=training_cfg.get("dataset_variant", "mini"),
    )
    return MultiTaskPretrainDataModule(dm_cfg)


def save_checkpoint(
    model: EEGJEPAModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    state: TrainingState,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, optimizer, state)
    ckpt_path = output_dir / f"checkpoint_epoch_{state.epoch:03d}.pt"
    torch.save(payload, ckpt_path)
    torch.save(payload, output_dir / "latest.pt")
    return ckpt_path


def checkpoint_payload(
    model: EEGJEPAModel,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
) -> dict:
    return {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }


def log_to_console(epoch: int, step: int, loss: float) -> None:
    console.log(
        f"Epoch [bold]{epoch:03d}[/bold] | Step [cyan]{step:06d}[/cyan] | Loss {loss:.4f}"
    )


def train_one_epoch(
    model: EEGJEPAModel,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    log_every: int,
    use_amp: bool,
    grad_clip: float,
    start_step: int,
    max_steps: Optional[int],
    wandb_run,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    step_count = 0
    global_step = start_step

    total_batches = len(loader) if hasattr(loader, "__len__") else None

    with create_progress(transient=True) as progress:
        task_id = progress.add_task(
            f"Epoch {epoch:03d}", total=total_batches
        )

        for batch_idx, (views, _info) in enumerate(loader, start=1):
            views = views.to(device)
            if views.dim() != 4 or views.size(1) < 2:
                raise ValueError(
                    "Expected input shape (batch, views, channels, samples) with >=2 views"
                )
            context = views[:, 0]
            target = views[:, 1]

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                context_emb = model(context)
                target_emb = model(target)
                loss = jepa_loss(context_emb, target_emb)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            loss_value = loss.detach().item()
            total_loss += loss_value
            step_count += 1
            global_step += 1

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss_value,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": global_step,
                    },
                    step=global_step,
                )

            progress.advance(task_id)

            if log_every and step_count % log_every == 0:
                log_to_console(epoch, global_step, loss_value)

            if max_steps is not None and global_step >= max_steps:
                break

    avg_loss = total_loss / max(step_count, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: EEGJEPAModel,
    loader: Iterable,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with create_progress(transient=True) as progress:
        task_id = progress.add_task("Validation", total=len(loader) if hasattr(loader, "__len__") else None)
        for views, _info in loader:
            views = views.to(device)
            context = views[:, 0]
            target = views[:, 1]
            with autocast(device_type=device.type, enabled=use_amp):
                loss = jepa_loss(model(context), model(target))
            total_loss += loss.item()
            n_batches += 1
            progress.advance(task_id)
    model.train()
    return total_loss / max(n_batches, 1)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config = merge_overrides(base_config, args)
    training_cfg = config.setdefault("training", {})

    device = resolve_device(training_cfg.get("device"))
    set_global_seed(training_cfg.get("seed"))
    if device.type == "cuda":  # pragma: no cover - device dependent
        torch.backends.cudnn.benchmark = True

    tasks = config.get("tasks", {}).get("names", [])
    releases = config.get("releases", {}).get("names", [])
    dataset_variant = training_cfg.get("dataset_variant", "mini")

    console.rule("JEPA Pretraining")
    console.print(f"Using device: [bold]{device}[/bold]")
    console.print(f"Tasks: [cyan]{', '.join(tasks) or 'n/a'}[/cyan]")
    console.print(f"Releases: [cyan]{', '.join(releases) or 'n/a'}[/cyan]")
    console.print(f"Variant: [cyan]{dataset_variant}[/cyan]")

    dm = create_dataloader(config)
    loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    train_summary = dm.summary.get("train", {})
    if train_summary:
        console.print(
            "Train recordings: [cyan]{}[/cyan] (windows: {}, subjects: {}, batches/epoch: {})".format(
                train_summary.get("n_recordings", "?"),
                train_summary.get("n_windows", "?"),
                len(train_summary.get("subjects", [])),
                train_summary.get("batches_per_epoch", "?"),
            )
        )
    val_summary = dm.summary.get("val", {})
    if val_summary:
        console.print(
            "Val recordings: [cyan]{}[/cyan] (windows: {}, subjects: {}, batches: {})".format(
                val_summary.get("n_recordings", "?"),
                val_summary.get("n_windows", "?"),
                len(val_summary.get("subjects", [])),
                val_summary.get("batches_per_epoch", "?"),
            )
        )

    tasks = config.get("tasks", {}).get("names", [])
    releases = config.get("releases", {}).get("names", [])
    dataset_variant = train_cfg.get("dataset_variant", "mini")
    chs_info = None
    if tasks and releases:
        try:
            chs_info = load_chs_info(
                tasks[0],
                releases[0],
                variant=dataset_variant,
            )
        except Exception as exc:
            console.log(f"[yellow]Failed to load channel info: {exc}; using default layout[/yellow]")

    crop_size_s = train_cfg.get("crop_size_s", 2.0)
    n_times = int(crop_size_s * SFREQ)
    model = EEGJEPAModel(n_times=n_times, chs_info=chs_info).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )

    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_every = int(training_cfg.get("log_every_steps", 25))
    grad_clip = float(training_cfg.get("clip_grad_norm", 0.0))
    epochs = int(training_cfg.get("epochs", 1))
    max_steps = training_cfg.get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
    if args.max_steps is not None:
        max_steps = args.max_steps

    output_dir = Path(training_cfg.get("output_dir", "weights/pretrain_jepa"))
    if args.output_dir is not None:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    console.log(f"Resolved config written to {config_path}")

    wandb_run = maybe_init_wandb(config)

    global_step = 0
    best_val_loss: float | None = None
    try:
        for epoch in range(1, epochs + 1):
            avg_loss, global_step = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                log_every=log_every,
                use_amp=use_amp,
                grad_clip=grad_clip,
                start_step=global_step,
                max_steps=max_steps,
                wandb_run=wandb_run,
            )
            console.print(
                f"Epoch [bold]{epoch:03d}[/bold] completed | avg loss {avg_loss:.4f}"
            )
            if val_loader is not None:
                val_loss = evaluate(model, val_loader, device, use_amp)
                console.print(
                    f"  Validation contrastive loss: [cyan]{val_loss:.4f}[/cyan]"
                )
                if wandb_run is not None:
                    wandb_run.log({"val/loss": val_loss, "train/epoch": epoch}, step=global_step)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        checkpoint_payload(model, optimizer, TrainingState(epoch, global_step)),
                        output_dir / "checkpoint_best.pt",
                    )
                    console.log("Updated checkpoint_best.pt (validation loss improved)")
            state = TrainingState(epoch=epoch, global_step=global_step)
            if epoch % max(int(training_cfg.get("save_every_epochs", 1)), 1) == 0:
                ckpt_path = save_checkpoint(model, optimizer, output_dir, state)
                console.log(f"Saved checkpoint to {ckpt_path}")
            if max_steps is not None and global_step >= max_steps:
                console.print("[yellow]Max steps reached, exiting training loop.[/yellow]")
                break
    finally:
        if wandb_run is not None:  # pragma: no cover - depends on optional dep
            wandb_run.finish()

    console.print("[bold green]Pretraining completed.[/bold green]")


if __name__ == "__main__":
    main()
