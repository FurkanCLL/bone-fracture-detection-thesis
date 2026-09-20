# Canonical Thesis Evidence

This directory keeps compact, version-controlled evidence needed to understand and reproduce important thesis decisions.

Large or routinely generated artifacts remain under the ignored `outputs/` tree. This includes full manifests, visual review collections, temporary files, training runs, evaluation exports, and model weights. Those files must not be force-added to Git.

Each completed phase should copy only the small canonical summaries or tables needed for traceability:

- `phase2a/` records the approved prepared-dataset build summary;
- `phase2b/` records conversion validation and occupancy summaries;
- `phase2c/` records the frozen training protocol and detected environment;
- `phase2d/` records the deterministic CLAHE build, intensity summary, and manual visual QA decision;
- later phases should follow the same compact-evidence principle.

Generated evidence may contain paths that describe its original run location. The surrounding phase report and current tool defaults define the active local output structure.
