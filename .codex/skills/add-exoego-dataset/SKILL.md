---
name: add-exoego-dataset
description: Add or update SimpleCV ExoEgo dataset adapters end to end. Use when implementing a dataset loader, mapping labels to COCO-133, storing MANO/depth metadata, saving RRDs, registering ExoEgo Forge catalog entries, serving the Rerun catalog/web viewer, or validating screenshots.
---

# Add ExoEgo Dataset

Keep this skill short: read the repo, copy local patterns, prove the result visually, and avoid inventing parallel tooling.

## Read First

- Task/goal doc, if present.
- `AGENTS.md`, `pyproject.toml`, and `docs/exoego_schema.md`.
- Closest existing adapter: usually HOCap for ego+exo+MANO, Assembly101 for multi-view hand keypoints, HOT3D for headset+MANO, or Aria/EgoDex for ego-only.
- Existing entrypoints before adding new ones: `view_exoego.py`, `batch_raw_to_rrd.py`, and `exoego_forge_catalog.py`.

## Workflow

- Use Pixi and TDD. Build vertical slices: identity/config, discovery count, `load_labels=False`, labels, optional MANO/depth/mesh, single RRD, catalog, full conversion, screenshots.
- Inspect real files early. Trust headers and sample arrays over docs for names, capitalization, timestamps, units, and shapes.
- Prefer the established three-file split for ego+exo datasets: `data/exoego/<dataset>.py`, `data/ego/<dataset>_ego.py`, `data/exo/<dataset>_exo.py`.
- Add `SequenceIdentity`, config `_target`, stream timestamps, `load_labels`, dataset iteration/counting, `dataset_defaults`, and catalog defaults when full registration is in scope.
- Do not add dataset-specific viewer/catalog scripts. Use generic view, batch conversion, and `tools/catalog.py`.

## Labels And MANO

- End state for labels is `ExoEgoLabels.xyzc_stack: Float[np.ndarray, "num_frames 133 4"]`; missing points are `NaN xyz` with `0.0` confidence.
- COCO-133 rows: body `0:17`, feet `17:23`, face `23:91`, left hand `91:112`, right hand `112:133`.
- Use released 3D keypoints when available. Store MANO/SMPL parameters separately; do not generate COCO labels from MANO/SMPL unless this repo already has a tested conversion for that exact source.
- `ManoStack` convention is hand `0=right`, `1=left`. If a dataset has one shared shape requirement but per-hand shapes differ, follow the user/local precedent and warn rather than fail.
- Verify hand order from upstream code or links before reordering. Apply MANO-to-COCO reorder only to raw MANO joints, not released COCO-ordered hand keypoints.

## Rerun And Catalog Lessons

- Log video through the shared `log_video` path so recordings use video streams, not ad hoc asset-video logging.
- Tune `image_plane_distance` from screenshots; overlapping camera frusta usually mean it is too large.
- For browser sharing, use one SimpleCV catalog command but expose two endpoints: web viewer port and catalog server port. Tailscale raw TCP forwarding worked where HTTPS proxying caused malformed catalog responses.
- Rerun directory dataset registration expects direct child `.rrd` files. For nested datasets, register a flat hardlink mirror so built-in dataset tables show rows.
- If AV1/Hugging Face mirroring is in scope, transcode videos first, prefer NVENC when available, register against AV1 paths, and skip sidecars made obsolete by the mirror.

## Done Means

- Focused tests pass, including catalog tests for defaults, camera names, table names, and nested RRD discovery.
- A representative sequence opens with no labels, then with labels, then as an RRD.
- Full conversion covers every supported session; every skip has a written reason.
- Native Rerun screenshots show direct viewer, labels/MANO when present, catalog table rows, and clicked recordings.
