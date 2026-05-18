# Repo Instructions

## Package Manager

Use Pixi for all project commands. Do not use `pip` or `uv` in this repo.

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
