import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_pretrain_jepa.py"

spec = importlib.util.spec_from_file_location("train_pretrain_jepa", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module  # type: ignore[union-attr]
spec.loader.exec_module(module)  # type: ignore[arg-type]


def test_jepa_merge_overrides_updates_fields(tmp_path):
    base = {
        "tasks": {"names": ["RestingState"]},
        "releases": {"names": ["R1"]},
        "training": {"epochs": 5, "batch_size": 32, "learning_rate": 1e-3},
    }
    args = module.parse_args([
        "--epochs",
        "2",
        "--batch-size",
        "64",
        "--lr",
        "0.0005",
        "--output-dir",
        str(tmp_path),
        "--device",
        "cpu",
        "--val-fraction",
        "0.1",
    ])
    merged = module.merge_overrides(base, args)
    assert merged["training"]["epochs"] == 2
    assert merged["training"]["batch_size"] == 64
    assert merged["training"]["learning_rate"] == 0.0005
    assert merged["training"]["device"] == "cpu"
    assert merged["training"]["val_fraction"] == 0.1


def test_resolve_device_auto():
    device = module.resolve_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_masked_merge_overrides_updates_fields(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "train_masked_model", Path(__file__).resolve().parents[2] / "scripts" / "train_masked_model.py"
    )
    masked_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = masked_module
    spec.loader.exec_module(masked_module)  # type: ignore[arg-type]

    base = {
        "tasks": {"names": ["RestingState"]},
        "releases": {"names": ["R1"]},
        "training": {"epochs": 5, "batch_size": 32, "learning_rate": 1e-3},
    }
    args = masked_module.parse_args([
        "--epochs",
        "3",
        "--batch-size",
        "16",
        "--lr",
        "0.0001",
        "--dataset-variant",
        "full",
        "--time-mask",
        "0.2",
        "--channel-mask",
        "0.3",
        "--mask-value",
        "-1.0",
        "--tasks",
        "RestingState",
        "surroundSupp",
    ])
    merged = masked_module.merge_overrides(base, args)
    train_cfg = merged["training"]
    assert train_cfg["epochs"] == 3
    assert train_cfg["batch_size"] == 16
    assert train_cfg["learning_rate"] == 0.0001
    assert train_cfg["dataset_variant"] == "full"
    assert train_cfg["time_mask_fraction"] == 0.2
    assert train_cfg["channel_mask_fraction"] == 0.3
    assert train_cfg["mask_value"] == -1.0
    assert merged["tasks"]["names"] == ["RestingState", "surroundSupp"]
