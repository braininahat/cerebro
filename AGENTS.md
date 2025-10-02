# Repository Guidelines

## Primary References
- Treat `startkit/README.md`, `startkit/challenge_1.py`, and `startkit/challenge_2.py` as ground truth for challenge definitions, data splits, and baselines.
- Use `README.md` for consolidated project docs (data prep, sweep commands, metrics) and surface updates there when behaviour changes.
- `gpt_roadmap.md` tracks upcoming custom work—update it after major planning decisions.

## Project Structure & Module Organization
- `src/cerebro/` remains the public API; keep modules snake_case and group helpers by domain (data, preprocessing, models, training, metrics).
- `notebooks/` contains exploratory work (EDA, benchmark sweep, dataset inspection). Keep notebooks lightweight and refactor reusable logic into `src/`.
- `startkit/` stays vendor code; avoid editing it unless mirroring upstream updates.
- Large artifacts belong under `weights/` (models) and `wandb/` (run logs); both are git-ignored.

## Build, Test, and Execution Commands
- `pip install -e .` installs Cerebro utilities for local development.
- `pip install -r startkit/requirements.txt` ensures parity with the official kits.
- Prefer launching scripts via `uv run python <path>` (mirrors the guidance in `CLAUDE.md`). Examples:
  - `uv run python notebooks/002_benchmark.py --show-configs`
  - `uv run python startkit/challenge_1.py`
- Formatting & linting: `make format` or `black --check . && isort --check .`.
- Tests: `pytest -q tests/` (add coverage for new utilities).

## Coding Style & Naming Conventions
- Follow 4-space indentation, snake_case for functions/variables, CapWords for classes, ALL_CAPS for constants.
- Use concise NumPy-style docstrings (`Args`, `Returns`, `Raises`).
- Keep modules focused; do not expand `__init__.py` beyond exports.
- Prefer ASCII unless existing files require Unicode (e.g., README badges).

## Data & Experiment Tips
- Set the `EEG2025_DATA_DIR` env var to the cache location (defaults to `<repo>/data`); mirror the start kit patterns when overriding per-script `DATA_DIR`.
- Apply the shared exclusion list for problematic subjects before splitting.
- When extending notebooks, sync new behaviours back into `src/` helpers and document the workflow in `README.md`.
- Log long-running experiments to W&B (see README sweep section) and store checkpoints in `weights/` with informative names.

## Testing Guidelines
- Keep tests isolated and deterministic; use fixtures to mock filesystem/network access when needed.
- Mirror library layout in `tests/` (e.g., `tests/test_data_pipeline.py`).
- Validate both success paths and failure modes (e.g., missing anchors, invalid configs).

## Commit & Pull Request Guidelines
- Commit subject lines: ≤50 chars, imperative (e.g., `Add eegnet benchmark`).
- Reference issues (`Closes #123`) and summarise motivation + impact in the body.
- Note formatting/test status in the PR description; include screenshots only for visual diffs.

## Security & Configuration Tips
- Never commit raw data, checkpoints, or secrets. Use env vars or config files listed in `.gitignore`.
- Document new flags or env vars in `README.md` so other agents stay aligned.
- Before merging, verify that notebooks do not embed absolute paths or personal identifiers.
