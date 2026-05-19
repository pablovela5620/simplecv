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

The default environment uses the public conda-forge / PyPI `rerun-sdk`
release pinned in `pyproject.toml` (see `dependencies` and
`[tool.pixi.feature.dev]`). **Always prefer the public release.**

A separate `rerun-prerelease` pixi feature exists that installs an
unreleased build of `rerun-sdk` from
`https://build.rerun.io/commit/<reality-sha>/wheels/`. **Only opt into
this when you hit a Rerun bug that has been fixed on `main` but not yet
shipped in a release.** Workflow:

1. Reproduce the bug against the public release with a small script and
   capture the symptom (timing curve, error message, partial-state
   behavior).
2. Locate the upstream fix: the PR on `rerun-io/rerun` and its
   corresponding merge commit on `rerun-io/reality` (private). The
   wheel index key is the *reality* commit, not the public one — try
   the 7-char truncation first (e.g. `5f732f2`); if HTTP 404, the wheel
   isn't built yet, pick the next reality commit that returns HTTP 200.
3. Update the SHA in `[tool.pixi.feature.rerun-prerelease.pypi-options]`
   `find-links` and the version string in `pypi-dependencies` to match
   the wheel filename at that commit (e.g. `==0.33.0a1+dev`).
4. Use the env via `pixi run -e rerun-prerelease …` for verification.
   Do **not** make `rerun-prerelease` the default for any task in
   `pyproject.toml`; leave it as an opt-in.
5. Once a public release ships with the fix, bump the public version
   pin and revert the `rerun-prerelease` feature back to its previous
   commented-out / stub state.

When committing a switch to the prerelease feature, include a comment
naming the upstream issue + PR (e.g. `rerun-io/rerun#12778` /
`rerun-io/rerun#12774`) so the next engineer to look at this knows why
we're off the public release.
