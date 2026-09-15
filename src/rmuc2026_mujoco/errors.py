"""Public exceptions raised by :mod:`rmuc2026_mujoco`."""


class Rmuc2026Error(RuntimeError):
    """Base class for package errors."""


class ManifestError(Rmuc2026Error):
    """The asset manifest is missing, malformed, or outside the supported scope."""


class AssetIntegrityError(ManifestError):
    """A declared asset is missing, escapes the asset root, or has the wrong hash."""


class MujocoModelError(Rmuc2026Error):
    """The field could not be represented by the expected MuJoCo model contract."""


class OutOfBoundsError(Rmuc2026Error):
    """A terrain query lies outside the declared heightfield."""
