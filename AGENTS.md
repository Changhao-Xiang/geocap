# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ project for geometric-perception benchmarks and fossil-image captioning. Shared arguments, inference helpers, and model adapters live in `common/`; synthetic rule, rendering, caption, and VQA pipelines live in `data/`. Evaluation code is under `eval/`, while `stage2_diffusion/` and `stage3/` contain fossil-generation and feature-recognition stages. Use `scripts/data/`, `scripts/eval/`, and `scripts/train/` for repeatable workflows. Model-specific integrations are in `llava/`, `qwen2vlm/`, and `LLaVA-1.5-GeP/`. Documentation and images belong in `docs/`; generated datasets and evaluation fixtures belong in `dataset/` and `eval_data/`.

## Build, Test, and Development Commands

- `python -m pip install -r deploy/requirements.txt` installs the supported environment.
- `./run -m data.rule.generate --num_basic_geo_samples 10` runs a small geometry-generation smoke test. On Windows, use `python run --module data.rule.generate ...`.
- `scripts/data/generate.sh train` builds the full training-data pipeline; use smaller module-level commands while developing.
- `scripts/eval/bench.sh --eval_model <name> --eval_batchsize <n>` evaluates a supported model and writes results under `results/`.
- `black . --skip-magic-trailing-comma --line-length 110` formats Python.
- `isort . --profile black --line-length 110` sorts imports.
- `flake8 . --ignore=E402,E731,W503,E203,F403,F405,E501 --exclude=llava` runs the required style checks.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations for new interfaces, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Keep lines compatible with the 110-column formatter settings. Pyright/Pylance runs in `basic` mode via `pyrightconfig.json`; prefer fixing types or using `typing.cast()` over broad `# type: ignore` comments.

## Testing Guidelines

There is no centralized unit-test suite or coverage threshold. Validate changes with the smallest relevant `./run` invocation, then run the affected data or evaluation script. Use deterministic seeds where available and inspect generated JSON/JSONL and images. Add focused tests near the changed module when practical; name them `test_<behavior>.py`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries in English or Chinese, commonly beginning with `fix`, `add`, `update`, or `remove`; keep each commit focused and optionally append an issue reference such as `(#136)`. Pull requests should explain the problem and approach, list validation commands, link related issues, and include sample output or screenshots for rendering/recognition changes. Do not commit model checkpoints, generated datasets, result directories, or real API credentials; keep secrets in ignored local configuration.
