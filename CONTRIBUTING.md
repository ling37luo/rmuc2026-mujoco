# Contributing

This is an unofficial community field simulator. Contributions should improve
reusable field geometry, interaction, diagnostics or documentation.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Keep the core independent of robot policies, RL frameworks and machine-specific
paths. Public examples should use included synthetic models or clearly state
which robot and controller the user must supply.

## Changes and validation

- Test changed behavior with small synthetic fixtures. Documentation-only
  changes need link and example checks, not a new physics campaign.
- Preserve asset identity checks and compatibility with supported pack schemas.
- Describe whether visuals, collision or public interfaces change. Record
  user-visible changes in [CHANGELOG.md](CHANGELOG.md).
- For collision changes, report the tested routes and results separately from
  whole-field validation. Keep detailed local evidence under `runs/`.

Do not commit official sources, derived field assets, external robot/policy
files or generated experiment reports. See [ASSET_POLICY.md](ASSET_POLICY.md).
CI uses synthetic assets and does not download official material.

Bug reports should include versions, the command and exception, plus the pack
manifest hash for asset-specific failures. Omit restricted assets and private
paths from public reports.
