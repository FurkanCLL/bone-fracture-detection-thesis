# Project working instructions

## Scope and evidence

- Treat this repository as a reproducible RTU bachelor thesis project, not a tutorial.
- Read `docs/THESIS.md`, `docs/ROADMAP.md`, and `docs/DECISIONS.md` before substantial work.
- Do not silently change the research scope, class mapping, split strategy, or evaluation protocol.
- Separate verified facts, interpretations, unresolved questions, and proposed decisions.
- Do not invent dataset facts, metrics, results, or references.
- Describe the system as an experimental fracture-localization pipeline, not a clinically validated diagnostic tool.

## Raw data and experiments

- Treat `data/raw/` as immutable. Never rename, rewrite, relabel, move, or preprocess raw files in place.
- Put derived datasets and generated evidence in clearly separate locations with traceability to the source files.
- Do not commit raw data, model weights, secrets, virtual environments, or large generated outputs.
- Keep train, validation, and test roles separate. Do not use the test set for iterative decisions.
- Record settings needed to reproduce an experiment, including the seed and software versions.
- Keep comparisons fair and document consequential choices in `DECISIONS.md`.

## Python style and comments

- Write clear, natural, student-level Python. Prefer readable functions and descriptive names over abstraction for its own sake.
- Add short English comments where they explain an important function, a non-obvious check, or why a choice exists.
- Prefer a short comment directly above an important function when it helps:

```python
# Checks YOLO labels and reports malformed bounding boxes.
def validate_labels(...):
    ...
```

- Add an inline comment only when a block would otherwise be difficult to understand.
- Do not comment every function or restate obvious code.
- Avoid long, robotic, tutorial-like, or AI-sounding comments.
- Keep comments accurate when code changes.

## Testing and repository structure

- Add focused tests for deterministic or error-prone logic such as label parsing, coordinate validation, and configuration handling.
- Run relevant tests and validation commands after critical changes.
- Add directories or dependencies only when they solve a clear project need.
- Prefer reusable scripts for thesis-relevant calculations; use notebooks only when interaction is genuinely useful.

## Git discipline

- Keep commits logically scoped and avoid mixing unrelated refactors with experimental work.
- Write short, natural commit subjects in the present tense.
- Capitalize the first word and follow normal Git subject conventions: use the imperative form, omit a trailing period, and keep the subject concise.
- Explain why in the commit body only when the change needs context.

Good examples:

- `Add dataset provenance checks`
- `Create empty-label review sample`
- `Clarify Phase 1 findings`

Avoid vague or mechanical subjects such as `updates`, `fixed stuff`, or a file list.

## When uncertain

Stop and make the uncertainty explicit before any change that would materially affect dataset semantics, raw data, patient grouping, class mapping, split independence, evaluation, or experiment comparability.
