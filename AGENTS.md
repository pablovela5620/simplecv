# Contributor Guide

## Repository Overview
- `simplecv/` – main library of computer‑vision utilities.
- `tools/` – example scripts and dataset viewers.
- `data/` – stores sample datasets (downloaded on demand).
- `media/` – images and animations for the docs.

## Dev Environment Tips
1. Install [Pixi](https://pixi.sh/latest/#installation).
2. Clone the repo and enter the environment:
   ```bash
   git clone https://github.com/pablovela5620/simplecv.git
   cd simplecv
   pixi shell
   ```
3. Use `pixi task list` to see available tasks.

## Testing Instructions
- Run `ruff check .` and fix any issues.
- Run project specific tasks with `pixi run <task>`.

## PR Instructions
- Title format: `[simplecv] <Title>`
- Include a **Summary** describing the changes.
- Add a **Testing** section with `ruff` results and any commands run.
- If there are placeholders or TODOs, include a **Notes** section.

