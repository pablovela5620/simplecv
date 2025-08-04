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

### Test‑Driven Development (TDD)
SimpleCV follows a strict **test‑driven development** workflow:

1. **Write the test first** – create or update a failing test in `simplecv/tests/` that captures the desired behaviour or reproduces a bug.
2. **Make it pass** – implement the minimal code changes required for the test to pass.
3. **Refactor** – clean up the implementation while keeping the test suite green.

### Running the test & lint suite
Run the full test and lint suite locally before every commit:

```bash
pixi run -e dev pytest            # run all tests
pixi run -e dev ruff check .      # static analysis & style
```

Pull requests that introduce new functionality without accompanying tests (or that leave tests failing) will not be accepted.

## Adding New Datasets
1. Create loader in `simplecv/data/your_dataset.py` with `@serde` classes
2. Add visualization API in `simplecv/apis/view_your_data.py`
3. Create tool script in `tools/view_your_dataset.py` using tyro
4. Add pixi task in `pyproject.toml` with download dependencies

## RRD Artifact Workflow

When working with datasets and visualization tools that use `RerunTyroConfig` (from `simplecv.configs.rerun_tyro_config`), you can create shareable RRD artifacts for easy visualization. This workflow is particularly useful for creating demos and debugging visualizations.

### Step 1: Generate RRD Artifact
Use any tool with `--rr-config.save` to generate an RRD file. Name artifacts descriptively with Rerun version and datetime:
```bash
# Example: Create Assembly101 720p artifact
# Format: {dataset}-{description}-rerun{version}-{YYYY-MM-DD-HHMMSS}.rrd
pixi run python tools/view_exoego.py --rr-config.save data/rrd-debug-artifacts/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd assembly101 --root-directory data/assembly101-720-sample --sequence-name nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724 --encoding av1-720-new
```

### Step 2: Upload to HuggingFace
Upload the generated RRD file to the artifacts dataset:
```bash
huggingface-cli upload pablovela5620/rrd-debug-artifacts data/rrd-debug-artifacts/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd --repo-type dataset --commit-message "Add assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd"
```

### Step 3: Create Shareable Rerun Link
The final download URL follows this pattern:
```
https://huggingface.co/datasets/pablovela5620/rrd-debug-artifacts/resolve/main/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd
```

You can then create a direct Rerun viewer link:
```
https://app.rerun.io/version/0.24.0/index.html?url=https://huggingface.co/datasets/pablovela5620/rrd-debug-artifacts/resolve/main/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd
```

### Artifact Naming Convention
Follow this naming pattern for consistency:
```
{dataset}-{description}-rerun{version}-{YYYY-MM-DD-HHMMSS}.rrd
```
Examples:
- `assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd`
- `polycam-room-scan-rerun0.24.0-2025-08-03-143000.rrd`
- `ego4d-hands-demo-rerun0.24.0-2025-08-03-091500.rrd`

This workflow enables easy sharing of complex 3D visualizations without requiring users to download and process the original datasets.

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

