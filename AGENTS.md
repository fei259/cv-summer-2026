# Repository Guidelines

## Project Structure & Module Organization

This repository is a learning-oriented PyTorch computer-vision project. `train.py` is the training entry point and currently runs the CIFAR-10 `SimpleCNN` baseline; `evaluate.py` loads the best checkpoint and produces test metrics and a confusion matrix. Dataset and `DataLoader` factories live in `data/`, architectures in `models/`, and reusable training, evaluation, and experiment-logging code in `utils/`. Keep dated exercises in `notes/`, future experiment configuration in `configs/`, and generated metrics or figures in `results/<experiment>/`. Curated baseline figures belong in `results/baseline/`.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py
python evaluate.py
python -m compileall train.py evaluate.py data models utils
```

`train.py` downloads or reuses CIFAR-10, trains the model, saves `results/cifar10/best_model.pth`, plots curves, and appends metrics to `results/experiments.csv`. Run `evaluate.py` only after a checkpoint exists. `compileall` is the required quick syntax check.

## Coding Style & Naming Conventions

Use UTF-8, four-space indentation, and PEP 8 conventions. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Prefer `pathlib.Path` over hard-coded path strings. Keep data loading, models, and training logic separated. Add comments for tensor shapes, device movement, or non-obvious experiment decisions; avoid comments that merely repeat the code. No formatter or linter is configured, so review formatting manually.

## Testing Guidelines

There is currently no automated test suite or coverage target. Before committing, run `compileall`, then perform a small training/evaluation smoke test when behavior changes. Check tensor shapes, devices, loss/accuracy values, checkpoint creation, and result paths. Add future pytest tests under `tests/` using names such as `test_engine.py`; tests should use synthetic data and must not download full datasets.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat: add experiment tracking`, `fix: correct accuracy aggregation`, or `docs: update repository guidelines`. Keep each commit focused and leave the project runnable. Review `git status` and `git diff` before committing. Pull requests should explain the goal, changed modules, verification commands, and resulting metrics; attach updated plots when experiment behavior changes. Never commit `.venv/`, raw datasets, secrets, caches, or `*.pt`/`*.pth` checkpoints.
