# Repository Guidelines

## Project Structure & Module Organization

This repository is a learning-driven, reproducible computer-vision project. `notes/` contains dated Python exercises and Jupyter notebooks, such as `day08_tensor.py` and `day07_visualization.ipynb`. Generated plots belong under `results/<experiment>/`, currently `results/week1/`. As the project grows, place model definitions in `models/`, reusable training and evaluation code in `utils/`, and experiment settings in `configs/`. Keep downloaded datasets in `data/`; this directory and model weights (`*.pt`, `*.pth`) are intentionally ignored by Git.

## Build, Test, and Development Commands

The project has no build step. Run commands from the repository root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python notes/day08_tensor.py
python -c "import torch; print(torch.cuda.is_available())"
git status
```

Use VS Code or Jupyter to open notebooks. Before committing a notebook, restart its kernel and run all cells from top to bottom so hidden state or incorrect cell ordering cannot mask errors.

## Coding Style & Naming Conventions

Use Python 3.12, UTF-8, four-space indentation, and PEP 8 conventions. Use `snake_case` for files, variables, and functions; use `PascalCase` for model classes and `UPPER_CASE` for constants. Prefer `pathlib.Path` over manually concatenated paths. Keep functions focused, avoid duplicated training logic, and add short comments only where tensor shapes, devices, or non-obvious decisions need explanation.

## Testing Guidelines

There is no dedicated automated test suite yet. Validate scripts by running them in the project virtual environment and checking outputs, tensor shapes, dtypes, and devices. When reusable modules are introduced, add pytest tests under `tests/` using names such as `test_engine.py` and functions such as `test_evaluate_returns_accuracy()`. Tests should use small synthetic inputs and must not download full datasets.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `feat: complete NumPy fundamentals`, `fix: correct tensor device transfer`, or `chore: initialize project`. Keep each commit runnable and limited to one learning milestone or experiment. Before committing, inspect `git status` and relevant diffs. Pull requests should describe the goal, changed files, verification commands, and experiment environment; include plots or metric comparisons when results change. Never commit secrets, virtual environments, raw datasets, or large model weights.
