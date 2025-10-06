import torch

from cerebro.backbones import EEGConvBackbone, EEGConvBackboneConfig


def test_backbone_output_shape_matches_config():
    cfg = EEGConvBackboneConfig(
        input_channels=5,
        input_samples=100,
        hidden_channels=32,
        embedding_dim=64,
    )
    model = EEGConvBackbone(cfg)
    x = torch.randn(8, cfg.input_channels, cfg.input_samples)
    out = model(x)
    assert out.shape == (8, cfg.embedding_dim)


def test_backbone_handles_default_dimensions():
    model = EEGConvBackbone()
    x = torch.randn(2, model.config.input_channels, model.config.input_samples)
    out = model(x)
    assert out.shape == (2, model.config.embedding_dim)
