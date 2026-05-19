# Repo Instructions

## Package Manager

Use Pixi for all project commands. Do not use `pip` or `uv` in this repo.

## Performance Benchmarking

Use the default Pixi environment for actual performance measurements. Do not use
`pixi run -e dev` for benchmark numbers, because the dev environment enables
runtime checking and can significantly slow down the measured path. Reserve
`pixi run -e dev` for tests and validation.

## Implementation Workflow

Use the `tdd` skill for implementation work. Prefer a failing or characterization
test first, then make the smallest change that proves the behavior.

## Existing Pattern First

Before adding new infrastructure, inspect the closest existing implementation and
preserve its default behavior unless there is a demonstrated blocker.

For catalog/server changes:

- Do not replace the existing `tools/catalog.py` / `exoego_forge_catalog.py` flow
  with a parallel implementation.
- Assembly101-style nested RRD registration must continue to work through
  recursive discovery and explicit RRD file lists.
- Transport-specific workarounds must be opt-in and covered by tests proving the
  normal path is unchanged.

## Python Style

- Use PEP 526-style variable annotations for nontrivial local values.
- Annotate arrays with jaxtyping dtype and shape.
- Follow the repo's existing dataclass field documentation style.

## Pre-release Rerun

Prefer the public `rerun-sdk` release. Use the `rerun-prerelease` Pixi
environment only for a confirmed Rerun bug that is fixed upstream but not yet
released.

When updating it, pin `find-links` to the matching `rerun-io/reality` wheel
commit, match the wheel version string, and leave the upstream issue/PR in a
comment. Keep it opt-in; do not make prerelease Rerun the default. Once the fix
ships publicly, move back to the public release pin.
