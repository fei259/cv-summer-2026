# Repository Guidelines

## Project Structure & Module Organization

This repository is a learning-oriented PyTorch computer-vision project centered on CIFAR-10. `train.py` trains candidate configurations and selects checkpoints using validation metrics; `evaluate.py` performs the final test evaluation. Keep dataset splits and `DataLoader` factories in `data/`, model definitions in `models/`, and reusable training, plotting, and logging helpers in `utils/`. Store controlled experiment settings in `configs/`, exercises and report drafts in `notes/`, and generated metrics or figures under descriptive `results/` subdirectories. Root-level `analyze_*.py` scripts compare experiments or inspect errors.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall train.py evaluate.py data models utils
python train.py
python evaluate.py
python analyze_errors.py
```

There is no separate build step. `compileall` is the required syntax check. Run `train.py` for training and validation; run `evaluate.py` only after validation has selected a checkpoint. Analysis scripts read recorded experiments or fixed checkpoints and write plots or summaries beneath `results/`.

## Coding Style & Naming Conventions

Use UTF-8, four-space indentation, and PEP 8 spacing. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Prefer `pathlib.Path` over manually concatenated paths. Keep data preparation, model architecture, training, evaluation, and analysis separate. Comments should clarify tensor shapes, experimental decisions, or non-obvious indexing rather than restate code. No formatter or linter is configured; run `git diff --check` before committing.

## Testing & Experiment Guidelines

No automated test suite or coverage target exists yet. After code changes, run `compileall` and a small synthetic or one-batch smoke test before full training. Verify tensor shapes, devices, checkpoint reloads, CSV fields, and output paths. Change one experimental variable at a time and use fixed seeds for paired comparisons. Select epochs and configurations with validation data; reserve test data for the final report. Future pytest files belong in `tests/` and must follow `test_*.py`.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat: add validation-based model selection`, `fix: correct metric logging`, or `docs: update experiment report`. Keep commits focused and inspect `git status`, `git diff`, and staged changes first. Pull requests should describe the goal, changed modules, verification commands, and relevant metrics; attach updated plots when results change. Never commit `.venv/`, caches, raw datasets, secrets, or `*.pt`/`*.pth` checkpoints.
