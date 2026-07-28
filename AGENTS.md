# Repository Guidelines

## Project Structure & Module Organization

This repository is a learning-oriented PyTorch computer-vision project. `train.py` is the current FashionMNIST entry point. Dataset creation and batching live in `data/fashion_mnist.py`; model definitions belong in `models/`; reusable training and evaluation loops belong in `utils/`; and future experiment settings belong in `configs/`. Dated exercises and notebooks are kept in `notes/`. Store generated plots and metrics under `results/<experiment>/`, such as `results/fashion/training_curves.png`. Raw datasets, virtual environments, caches, and model weights are local artifacts and must remain untracked.

## Build, Test, and Development Commands

There is no build step. Run commands from the repository root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py
python -m compileall train.py data models utils
python -c "import torch; print(torch.cuda.is_available())"
git status
```

`python train.py` downloads or reuses FashionMNIST, trains the MLP, saves the best local checkpoint, and writes training curves. `compileall` provides a quick syntax check. For notebooks, restart the kernel and run all cells in order before committing.

## Coding Style & Naming Conventions

Use UTF-8, four-space indentation, and PEP 8 conventions. Name modules, functions, and variables with `snake_case`; classes with `PascalCase`; and constants with `UPPER_CASE`. Prefer `pathlib.Path` for filesystem paths. Keep model architecture, data handling, and training logic in their respective modules. Add concise comments for tensor shapes, device transfers, or non-obvious experiment decisions rather than narrating every statement.

## Testing Guidelines

No automated test suite or coverage target exists yet. Validate changes with small, deterministic runs that check tensor shapes, dtypes, devices, returned metrics, and whether parameters change only during training. Future pytest files should go under `tests/`, use names such as `test_engine.py`, and avoid downloading full datasets.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit pattern, for example `feat: complete FashionMNIST training pipeline`, `fix: correct accuracy aggregation`, or `chore: add project structure`. Keep commits focused and runnable. Before committing, review `git status` and `git diff`. Pull requests should state the goal, changed modules, verification commands, and relevant metrics; include updated plots when experimental results change. Never commit `.venv/`, raw data, secrets, or `*.pt`/`*.pth` checkpoints.
