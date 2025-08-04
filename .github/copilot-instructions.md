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

### Array Type Annotations with JAXTyping
Every array must specify both dtype and shape using jaxtyping, and variables should be annotated at assignment time:
```python
from jaxtyping import Float, UInt8, Int
rgb: UInt8[np.ndarray, "H W 3"] = ...
intrinsics: Float[np.ndarray, "3 3"] = ...
indices: Int[np.ndarray, "N"] = ...
```
- Annotate variables using PEP 526-style assignments, even for intermediate values.
- This verbosity keeps code self-documenting and enables static and runtime shape validation.

### Runtime Type Checking with Beartype
The project uses environment-conditional beartype activation via `PIXI_ENVIRONMENT_NAME`:
```python
if os.environ.get("PIXI_ENVIRONMENT_NAME") == "dev":
    from beartype.claw import beartype_this_package
    beartype_this_package()
```
- Full runtime type checking only in dev environment
- No need to manually add `@beartype` decorators
- Zero overhead in production/default environments

References: [beartype docs](https://beartype.readthedocs.io/en/latest/), [jaxtyping docs](https://docs.kidger.site/jaxtyping/)

### Camera Parameter Structure
All cameras follow the `PinholeParameters` pattern:
```python
PinholeParameters(
    name="cam_01",
    intrinsics=Intrinsics(
        camera_conventions="RDF",  # or "RUB" 
        fl_x=fx, fl_y=fy, cx=cx, cy=cy,
        height=h, width=w
    ),
    extrinsics=Extrinsics(
        world_R_cam=R, world_t_cam=t  # OR cam_R_world, cam_t_world
    )
)
```
- Extrinsics automatically computes both `world_T_cam` and `cam_T_world` matrices
- Convention handling via `simplecv.ops.conventions` for coordinate system conversions

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
pixi shell -e dev  # Development environment (includes beartype, ruff)
pixi shell         # Default environment (production-like)
pixi task list     # Shows all available tasks
```

**Critical**: Always use the dev environment for development:
- `pixi run -e dev` for running commands
- `pixi shell -e dev` for interactive development
- beartype runtime checking only activates in dev environment

### Testing & Linting
```bash
pixi run -e dev ruff check .      # Required before PRs
pixi run -e dev pytest            # Run test suite
```

### Camera Convention Handling
Use `simplecv.ops.conventions` for coordinate system conversions:
- `CameraConventions.CV` (OpenCV: X right, Y down, Z forward)
- `CameraConventions.GL` (OpenGL: X right, Y up, Z backward)
- Always convert to project's standard convention in data loaders

### Adding New Datasets
1. Create loader in `simplecv/data/your_dataset.py` with `@serde` classes
2. Add visualization API in `simplecv/apis/view_your_data.py` 
3. Create tool script in `tools/view_your_dataset.py` using tyro
4. Add pixi task in `pyproject.toml` with download dependencies

### RRD Artifact Creation and Sharing
When tools use `RerunTyroConfig` (from `simplecv.configs.rerun_tyro_config`), you can create shareable RRD artifacts:

#### 1. Generate RRD File
Use `--rr-config.save` with any visualization tool. Name artifacts descriptively with Rerun version and datetime:
```bash
# Format: {dataset}-{description}-rerun{version}-{YYYY-MM-DD-HHMMSS}.rrd
pixi run python tools/view_exoego.py --rr-config.save data/rrd-debug-artifacts/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd [subcommand] [options]
```

#### 2. Upload to HuggingFace
```bash
huggingface-cli upload pablovela5620/rrd-debug-artifacts data/rrd-debug-artifacts/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd --repo-type dataset --commit-message "Add assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd"
```

#### 3. Create Shareable Links
- **Download URL**: `https://huggingface.co/datasets/pablovela5620/rrd-debug-artifacts/resolve/main/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd`
- **Rerun Viewer**: `https://app.rerun.io/version/0.24.0/index.html?url=https://huggingface.co/datasets/pablovela5620/rrd-debug-artifacts/resolve/main/assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd`

#### Artifact Naming Convention
Use this pattern: `{dataset}-{description}-rerun{version}-{YYYY-MM-DD-HHMMSS}.rrd`
- Include dataset name, brief description, Rerun version, and timestamp
- Examples: `assembly101-720p-sample-rerun0.24.0-2025-08-03-212600.rrd`, `polycam-room-scan-rerun0.24.0-2025-08-03-143000.rrd`

This workflow enables easy sharing of complex 3D visualizations without requiring users to download and process original datasets.

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
