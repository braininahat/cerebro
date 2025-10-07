import torch

import scripts.evaluate_jepa_linear as eval_script


def test_fit_ridge_matches_closed_form():
    torch.manual_seed(0)
    X = torch.randn(10, 3)
    true_w = torch.randn(4, 1)  # including bias
    X_with_bias = torch.cat([X, torch.ones(10, 1)], dim=1)
    y = X_with_bias @ true_w

    weights = eval_script.fit_ridge(X, y, alpha=1e-6)
    preds = eval_script.predict(X, weights).squeeze()
    assert torch.allclose(preds, y.squeeze(), atol=1e-4)


def test_resolve_device_auto(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = eval_script.resolve_device("auto")
    assert device.type == "cpu"
