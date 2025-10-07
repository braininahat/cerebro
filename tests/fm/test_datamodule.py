from fm.datamodules import MultiTaskPretrainConfig, MultiTaskPretrainDataModule


def test_datamodule_setup(monkeypatch):
    called = {}

    def fake_loader(**kwargs):
        called.update(kwargs)
        return "loader"

    monkeypatch.setattr("fm.datamodules.multitask.create_pretraining_dataloader", fake_loader)

    cfg = MultiTaskPretrainConfig(
        tasks=["task"], releases=["R1"], batch_size=32, views=2
    )
    dm = MultiTaskPretrainDataModule(cfg)
    loader = dm.train_dataloader()
    assert loader == "loader"
    assert called["batch_size"] == 32
    assert called["views"] == 2
    assert called["dataset_variant"] == "mini"


def test_datamodule_returns_val_loader(monkeypatch):
    def fake_loader(**kwargs):
        assert kwargs["val_fraction"] == 0.2
        assert kwargs["dataset_variant"] == "mini"
        return "train", "val"

    monkeypatch.setattr("fm.datamodules.multitask.create_pretraining_dataloader", fake_loader)

    cfg = MultiTaskPretrainConfig(
        tasks=["task"], releases=["R1"], val_fraction=0.2
    )
    dm = MultiTaskPretrainDataModule(cfg)
    train = dm.train_dataloader()
    val = dm.val_dataloader()
    assert train == "train"
    assert val == "val"
