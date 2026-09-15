# Security policy

## Supported versions

Until the first stable release, security fixes are applied to the newest
published `0.x` release only.

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** form in the repository Security tab so
path-traversal, archive-extraction, integrity-check, or dependency issues are
not disclosed before a fix is available. If private vulnerability reporting is
not enabled, open a minimal issue asking the maintainers to enable a private
channel; do not publish exploit details or restricted assets.

Please include the affected version, platform, reproduction steps, expected
and observed behavior, and whether untrusted input is required. Do not attach
official RMUC files, their derivatives, Fudan assets, credentials, or private
URLs.

## Security boundary

Asset manifests and their paths are untrusted input. The loader verifies path
containment, rejects symlinks where required, and checks declared SHA-256
digests before model construction. A matching digest establishes identity, not
safety, copyright permission, physical correctness, or endorsement.

The project downloads official CAD material only after the user selects the
`download` or `setup` command and provides the explicit reference-only
acknowledgement. It verifies the pinned size, STEP header, and SHA-256 before
conversion. MuJoCo XML and mesh parsing occur in third-party native code, so
only load asset packs you trust and keep MuJoCo updated.
