# Contributing

Thank you for helping make this loader safer and easier to reuse. This is an
unofficial community project, not an official RoboMaster simulator.

## Development setup

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Keep the runtime independent of RL-Lab, reinforcement-learning frameworks,
robot policies, and project-specific absolute paths. Public APIs should accept
ordinary paths and MuJoCo/Python objects.

## Asset-free contributions

Do not commit, attach to a pull request, or upload to a GitHub Release:

- official RMUC STEP or rulebook files;
- OBJ, GLB, heightfield, texture, or screenshot derivatives;
- Fudan robot assets, policies, or checkpoints;
- training runs, videos, or other large generated evidence.

Tests must generate their own tiny synthetic OBJ and heightfield fixture in a
temporary directory. Never make CI download official or derived field assets.
See [ASSET_POLICY.md](ASSET_POLICY.md) before changing an asset schema or
download-related feature.

## Change checklist

1. Add or update a synthetic test that fails before the change.
2. Preserve fail-closed path containment and SHA-256 verification.
3. Run the complete test suite on a clean environment.
4. Document user-visible API or manifest changes in `CHANGELOG.md`.
5. State whether the change affects visual geometry, collision geometry, or
   neither. Never infer physical validation from a successful file load.

Bug reports should include the package and MuJoCo versions, operating system,
the command used, and the full exception. For asset-specific failures, include
the manifest SHA-256 and validation status, but do not attach restricted files.
