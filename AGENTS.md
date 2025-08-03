# SimpleCV Contributor Guide

## Repository Overview
SimpleCV is a computer vision utility library focused on multi-camera 3D reconstruction and visualization:

- `simplecv/` – Core library with typed data structures and operations
- `simplecv/apis/` – High-level workflow functions (view_*, convert_*)  
- `simplecv/data/` – Dataset-specific loaders (polycam, ego4d, assembly101, etc.)
- `simplecv/ops/` – Low-level CV operations (triangulation, TSDF fusion, conventions)
- `tools/` – Command-line scripts that wrap APIs using tyro
- `data/` – Sample datasets (downloaded on demand via HuggingFace)

## Development Environment

### Setup
1. Install [Pixi](https://pixi.sh/latest/#installation) - see [pixi.sh](https://pixi.sh) for more info
2. Clone and setup:
   ```bash
   git clone https://github.com/pablovela5620/simplecv.git
   cd simplecv
   pixi shell -e dev
   ```

### Running Code
**Critical**: To run any Python code with correct dependencies, you must either:
- Be inside `pixi shell -e dev` environment, OR  
- Prefix commands with `pixi run -e dev` (e.g., `pixi run -e devpython tools/view_polycam.py`)
always use dev environment

### Available Tasks
```bash
pixi task list  # Shows all available tasks
pixi run view-polycam-data  # Example task
```
Read the pyproject.toml to get more information about tasks

## Key Design Patterns

### Transformation Matrix Notation
Use `cam_T_world` notation (transformation goes right to left):
```python
cam_points = cam_T_world @ world_points  # world → camera
```

### Type Safety
Runtime type checking via `beartype_this_package()` ensures all code is validated at runtime.

- Every function and variable must include type annotations.
- Use PEP 526–compliant annotated assignments wherever variables are defined.
- No manual `@beartype` decorators are needed.

### Array Type Annotations with JAXTyping
Arrays must be annotated with both dtype and shape using jaxtyping:
```python
from jaxtyping import Float, UInt8
rgb: UInt8[np.ndarray, "H W 3"] = ...
depth_map: Float[np.ndarray, "H W"] = ...
xyzc_stack: Float[ndarray, "num_frames 133 4"] = ...
```
- Include dtype (`Float`, `UInt8`, etc.) and shape string in every array annotation.
- Annotate array variables at assignment time, even for intermediates—the verbosity keeps code self-documenting.

References: [beartype docs](https://beartype.readthedocs.io/en/latest/), [jaxtyping docs](https://docs.kidger.site/jaxtyping/)

### CLI Tools
Use tyro + dataclasses for type-safe configuration:
```python
@dataclass
class ViewConfig:
    data_path: Path
    rr_config: RerunTyroConfig

tyro.cli(ViewConfig)
```

## Testing & Linting
- Run `pixi run -e dev ruff check .` and fix any issues before PRs

## Adding New Datasets
1. Create loader in `simplecv/data/your_dataset.py` with `@serde` classes
2. Add visualization API in `simplecv/apis/view_your_data.py`
3. Create tool script in `tools/view_your_dataset.py` using tyro
4. Add pixi task in `pyproject.toml` with download dependencies

## PR Instructions
- Title format: `[simplecv] <Title>`
- Include a **Summary** describing the changes
- Add a **Testing** section with `ruff` results and any commands run
- If there are placeholders or TODOs, include a **Notes** section

## Key References
- See `.github/copilot-instructions.md` for comprehensive development guide
- `simplecv/camera_parameters.py` - Core camera math and data structures
- `simplecv/apis/view_polycam_data.py` - Example visualization pipeline
- `pyproject.toml` - Pixi task definitions and dependencies

