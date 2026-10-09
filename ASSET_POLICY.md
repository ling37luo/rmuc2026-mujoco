# Asset policy

The repository and releases contain code, documentation and original synthetic
examples. Official CAD/rulebook sources and derived field assets stay local.

## Repository contents

Allowed: source URLs and hashes, schemas, synthetic test geometry, and code that
builds or loads user-supplied assets.

Do not upload official STEP/PDF files or their meshes, heightfields, textures
and screenshots without separate redistribution permission. External robots,
policies, checkpoints and generated experiment reports are also excluded.
This applies to Git, Git LFS, release attachments and Python packages.

`source` prints the pinned official identity. `download` requires the explicit
reference-only acknowledgement; `setup` performs that download as part of local
construction. Downloads are checked against the pinned size and SHA-256.
These commands do not grant redistribution rights or accept third-party terms.

CI uses synthetic fixtures. Release checks inspect source and wheel contents
for generated assets and run directories.

## Physical scope

The base collision model is a single-heightfield top-surface proxy. Source ray
misses may be filled with ground height; experimental outer-edge masks use a
finite void surrogate. Neither represents general holes, underpasses or stacked
surfaces. Optional source-wall collision covers selected static parts only.

`DRAFT_BLOCKED` means whole-field physical validation is incomplete. Matching
file hashes establishes pack identity, not physical accuracy or official
endorsement. See the [current scope](README.md#scenarios-and-current-scope).
