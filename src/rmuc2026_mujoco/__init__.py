"""Public API for the standalone, unofficial RMUC 2026 MuJoCo field loader."""

from .builder import BuilderUnavailable, build_from_official_step, build_runtime_asset_pack
from .download import (
    OFFICIAL_SOURCE_PAGE,
    OFFICIAL_STEP_PRODUCT,
    OFFICIAL_STEP_SHA256,
    OFFICIAL_STEP_SIZE,
    OFFICIAL_STEP_URL,
    DownloadedStep,
    DownloadError,
    download_official_step,
    verify_official_step,
)
from .errors import (
    AssetIntegrityError,
    ManifestError,
    MujocoModelError,
    OutOfBoundsError,
    Rmuc2026Error,
)
from .manifest import (
    DEFAULT_RUNTIME_PROFILE,
    RUNTIME_ARTIFACT_TYPE,
    RUNTIME_PROFILE_NAMES,
    FieldAsset,
    ValidationReport,
    verify_asset,
)
from .mjcf import (
    FIELD_ATTACH_PREFIX,
    FIELD_COLLISION_GEOM_NAME,
    HFIELD_NAME,
    UNOFFICIAL_FRICTION_PRESETS,
    apply_friction_preset,
    compose_with_robot,
    inject_exact_heightfield,
    load_model,
)
from .query import HeightFieldData, field_bounds, height_at, load_heightfield

__all__ = [
    "AssetIntegrityError",
    "BuilderUnavailable",
    "DownloadedStep",
    "DownloadError",
    "DEFAULT_RUNTIME_PROFILE",
    "FIELD_ATTACH_PREFIX",
    "FIELD_COLLISION_GEOM_NAME",
    "FieldAsset",
    "HFIELD_NAME",
    "HeightFieldData",
    "ManifestError",
    "MujocoModelError",
    "OFFICIAL_SOURCE_PAGE",
    "OFFICIAL_STEP_PRODUCT",
    "OFFICIAL_STEP_SHA256",
    "OFFICIAL_STEP_SIZE",
    "OFFICIAL_STEP_URL",
    "OutOfBoundsError",
    "RUNTIME_ARTIFACT_TYPE",
    "RUNTIME_PROFILE_NAMES",
    "Rmuc2026Error",
    "ValidationReport",
    "UNOFFICIAL_FRICTION_PRESETS",
    "apply_friction_preset",
    "build_from_official_step",
    "build_runtime_asset_pack",
    "compose_with_robot",
    "download_official_step",
    "field_bounds",
    "height_at",
    "inject_exact_heightfield",
    "load_heightfield",
    "load_model",
    "verify_asset",
    "verify_official_step",
]

__version__ = "0.1.0"
