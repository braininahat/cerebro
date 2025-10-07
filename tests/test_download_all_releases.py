import pathlib

import scripts.download_all_releases as dar


def test_normalise_handles_case_and_order():
    result = dar._normalise(["r2", "R1", "R2"], dar.ALL_RELEASES, label="release")
    assert result == ["R1", "R2"]


def test_main_invokes_cache_release(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_cache_release(
        task,
        release,
        *,
        mini,
        cache_dir,
        description_fields=None,
        dataset_workers=None,
        materialize_raw=False,
    ):
        calls.append((task, release, mini, pathlib.Path(cache_dir)))
        return True, None

    monkeypatch.setattr(dar, "cache_release", fake_cache_release)

    dar.main(
        [
            "--releases",
            "R1",
            "--tasks",
            "RestingState",
            "--skip-full",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert calls == [("RestingState", "R1", True, tmp_path.resolve())]
    out = capsys.readouterr().out
    assert "Cached 1 entries" in out


def test_main_reports_skipped(monkeypatch, tmp_path, capsys):
    def fake_cache_release(*_args, **_kwargs):
        return False, "no recordings"

    monkeypatch.setattr(dar, "cache_release", fake_cache_release)

    dar.main(
        [
            "--releases",
            "R1",
            "--tasks",
            "RestingState",
            "--skip-full",
            "--data-root",
            str(tmp_path),
        ]
    )

    out = capsys.readouterr().out
    assert "Skipped 1 entries" in out
