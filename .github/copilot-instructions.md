# SimpleCV Copilot Instructions

## Architecture Overview

SimpleCV is a computer vision utility library focused on multi-camera 3D reconstruction and visualization. The codebase follows a modular design:

- `simplecv/` - Core library with typed data structures and operations
- `simplecv/apis/` - High-level workflow functions (view_*, convert_*)  
- `simplecv/data/` - Dataset-specific loaders (polycam, ego4d, assembly101, etc.)
- `simplecv/ops/` - Low-level CV operations (triangulation, TSDF fusion, conventions)
- `tools/` - Command-line scripts that wrap APIs using tyro

## Key Design Patterns

### Transformation Matrix Notation
**Critical**: Use `cam_T_world` notation where the transformation goes from right to left:
```python
cam_points = cam_T_world @ world_points  # world → camera
world_points = world_T_cam @ cam_points  # camera → world
```
Both directions are stored in `Extrinsics` dataclass and computed automatically.

### Type Safety with JAXTyping
All arrays use shape-annotated types:
```python
from jaxtyping import Float, UInt8
rgb: UInt8[np.ndarray, "H W 3"] = ...  # Height × Width × Channels
intrinsics: Float[np.ndarray, "3 3"] = ...
```

### Runtime Type Checking with Beartype
**Critical**: Everything must be type annotated. The project uses `beartype_this_package()` for automatic runtime type validation:
- All functions, classes, and variables are automatically type-checked at runtime
- Use PEP 526-compliant annotated variable assignments whenever possible
- JAXTyping is preferred for array types to enable shape validation
- No need to manually add `@beartype` decorators - it's applied automatically

References: [beartype docs](https://beartype.readthedocs.io/en/latest/), [jaxtyping docs](https://docs.kidger.site/jaxtyping/)

### Configuration with Tyro + Dataclasses
CLI tools use tyro for type-safe configuration:
```python
@dataclass
class ViewConfig:
    data_path: Path
    rr_config: RerunTyroConfig  # Rerun visualization settings

tyro.cli(ViewConfig)  # Auto-generates CLI from dataclass
```

### Dataset Integration Pattern
Each dataset in `simplecv/data/` follows this structure:
1. Serde dataclass for JSON/config parsing (`@serde`)
2. Main data container with computed properties
3. Iterator/generator for efficient loading
4. Conversion to standard `PinholeParameters` format

### Rerun Visualization
All visualization uses Rerun SDK with standardized logging:
- `RerunTyroConfig` for consistent setup across tools
- `log_pinhole()` for camera visualization with standardized entity paths
- Blueprint configuration for 3D/2D layouts

## Development Workflow

### Environment Setup
```bash
pixi shell  # Enters conda environment (Linux/macOS only)
pixi task list  # Shows all available tasks
```

**Important**: To run any Python code with correct dependencies, you must either:
- Be inside `pixi shell` environment, OR
- Prefix commands with `pixi run` (e.g., `pixi run python tools/view_polycam.py`)

### Testing & Linting
```bash
ruff check .  # Required before PRs
pixi run <task-name>  # Run project-specific tasks
```

### Adding New Datasets
1. Create loader in `simplecv/data/your_dataset.py` with `@serde` classes
2. Add visualization API in `simplecv/apis/view_your_data.py` 
3. Create tool script in `tools/view_your_dataset.py` using tyro
4. Add pixi task in `pyproject.toml` with download dependencies

### Camera Convention Handling
Use `simplecv.ops.conventions` for coordinate system conversions:
- `CameraConventions.CV` (OpenCV: X right, Y down, Z forward)
- `CameraConventions.GL` (OpenGL: X right, Y up, Z backward)
- Always convert to project's standard convention in data loaders

## Key Files for Reference

- `simplecv/camera_parameters.py` - Core camera math and data structures
- `simplecv/apis/view_polycam_data.py` - Example of complete visualization pipeline
- `simplecv/data/polycam.py` - Reference dataset loader implementation
- `pyproject.toml` - Pixi task definitions and dependencies

## Integration Points

- **Rerun SDK**: All 3D visualization and logging
- **Open3D**: Point cloud processing and TSDF fusion
- **HuggingFace Hub**: Dataset storage and download via `pixi` tasks
- **Serde**: JSON/YAML serialization for camera parameters
- **Tyro**: CLI generation from dataclasses
