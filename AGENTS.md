# Repository Guidelines

## Project Structure & Module Organization

This repository is a learning-oriented PyTorch computer-vision project. `train.py` is the main training entry point, while `evaluate.py` loads a saved checkpoint and produces test metrics and a confusion matrix. Keep dataset and `DataLoader` factories in `data/`, model definitions in `models/`, and reusable training, evaluation, plotting, or logging code in `utils/`. Dated exercises and report drafts belong in `notes/`; experiment hypotheses and controlled configurations belong in `configs/`. Store generated metrics and figures under descriptive paths such as `results/augmentation/` or `results/regularization/`.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py
python evaluate.py
python -m compileall train.py evaluate.py data models utils
```

`compileall` is the required quick syntax check. `train.py` trains candidate configurations, selects checkpoints with validation metrics, and appends rows to `results/formal_validation_experiments.csv`. Run `evaluate.py` only after validation has selected one configuration and its checkpoint exists; it performs the final test evaluation and creates a confusion matrix. For model-only checks, execute the module directly, for example `python models/simple_cnn.py`.

## Coding Style & Naming Conventions

Use UTF-8, four-space indentation, and PEP 8 spacing. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_CASE`. Prefer `pathlib.Path` over hard-coded path strings. Keep data loading, model architecture, training logic, and analysis separate. Comments should explain tensor shapes, device movement, index mappings, or experiment decisions—not restate obvious code. No formatter or linter is configured, so run `git diff --check` before committing.

## Testing & Experiment Guidelines

There is no automated test suite or coverage target yet. After behavior changes, run `compileall` and a small synthetic or one-batch smoke test before launching full training. Verify tensor shapes, devices, loss/accuracy values, checkpoint reloads, CSV columns, and output paths. Change one experimental variable at a time, record its value in both the output directory and experiment log, and use fixed seeds for paired comparisons. Future pytest files should live in `tests/` and follow `test_*.py`.

## Commit & Pull Request Guidelines

Follow the repository’s Conventional Commit pattern, such as `feat: add configurable dropout experiments`, `fix: correct metric logging`, or `docs: update experiment notes`. Keep commits focused and review `git status`, `git diff`, and staged changes before committing. Pull requests should state the goal, changed modules, verification commands, and relevant metrics; attach updated plots when results change. Never commit `.venv/`, raw datasets, secrets, caches, or `*.pt`/`*.pth` checkpoints.
