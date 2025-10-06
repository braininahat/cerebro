from fm.datamodules import MultiTaskPretrainConfig, MultiTaskPretrainDataModule


def test_datamodule_setup(monkeypatch):
    called = {}

    def fake_loader(**kwargs):
        called.update(kwargs)
        return "loader"

    monkeypatch.setattr("fm.datamodules.multitask.create_pretraining_dataloader", fake_loader)

    cfg = MultiTaskPretrainConfig(tasks=["task"], releases=["R1"], batch_size=32)
    dm = MultiTaskPretrainDataModule(cfg)
    loader = dm.train_dataloader()
    assert loader == "loader"
    assert called["batch_size"] == 32
