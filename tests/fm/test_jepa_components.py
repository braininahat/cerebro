import torch

from fm.tasks import jepa_loss
from fm.models import EEGJEPAModel


def test_jepa_loss_has_expected_range():
    context = torch.randn(4, 128)
    target = torch.randn(4, 128)
    loss = jepa_loss(context, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_jepa_model_forward_shape():
    model = EEGJEPAModel()
    batch = torch.randn(2, model.encoder.config.input_channels, model.encoder.config.input_samples)
    out = model(batch)
    assert out.shape[0] == batch.size(0)
