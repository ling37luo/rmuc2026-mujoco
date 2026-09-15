"""Runtime-only display switches declared by an RMUC 2026 field pack."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import Any

from .errors import ManifestError, MujocoModelError
from .manifest import (
    FILL_LIGHT_NAME,
    KEY_LIGHT_NAME,
    SURFACE_GUIDE_GEOM_GROUP,
    SURFACE_GUIDE_GEOM_NAME,
    SURFACE_GUIDE_KIND,
    FieldAsset,
)
from .livery import GROUND_MARKING_OVERLAY_KIND


_LIGHTING_MODES = ("flat", "shadow")
_LIVERY_STATES = ("off", "on")


class FieldDisplayController:
    """Apply field-owned lighting and livery visibility without changing physics.

    A controller only operates controls explicitly declared by the runtime-pack
    manifest.  This makes schema-1 and early packs safe to load: their missing
    display contracts are reported as unavailable and are never guessed from
    model layout.
    """

    def __init__(
        self,
        asset: FieldAsset,
        *,
        lighting: str | None = "flat",
        livery: str | bool | None = "off",
    ) -> None:
        self._key_light_name: str | None = None
        self._fill_light_name: str | None = None
        self._lighting_toggle_key = "L"
        self._livery_geom_name: str | None = None
        self._livery_geom_group: int | None = None
        self._livery_toggle_key = "G"

        lighting_contract = self._lighting_contract(asset.manifest)
        if lighting_contract is not None:
            self._key_light_name = _required_string(
                lighting_contract, "key_light_name", label="visual_display.lighting"
            )
            if self._key_light_name != KEY_LIGHT_NAME:
                raise ManifestError("visual_display.lighting.key_light_name is unsupported")
            self._fill_light_name = _required_string(
                lighting_contract, "fill_light_name", label="visual_display.lighting"
            )
            if self._fill_light_name != FILL_LIGHT_NAME:
                raise ManifestError("visual_display.lighting.fill_light_name is unsupported")
            modes = lighting_contract.get("modes")
            if not isinstance(modes, list) or tuple(modes) != _LIGHTING_MODES:
                raise ManifestError("visual_display.lighting.modes must be ['flat', 'shadow']")
            default_mode = lighting_contract.get("default_mode")
            if default_mode != "flat":
                raise ManifestError("visual_display.lighting.default_mode must be 'flat'")
            if lighting_contract.get("toggle_key") != "L":
                raise ManifestError("visual_display.lighting.toggle_key must be 'L'")
            if lighting_contract.get("physics_changed") is not False:
                raise ManifestError("visual_display.lighting must not claim a physics change")
            self._lighting_toggle_key = "L"

        livery_contract = self._livery_contract(asset.manifest)
        if livery_contract is not None:
            if livery_contract.get("kind") not in {
                SURFACE_GUIDE_KIND,
                GROUND_MARKING_OVERLAY_KIND,
            }:
                raise ManifestError("visual_layers.livery.kind is unsupported")
            self._livery_geom_name = _required_string(
                livery_contract, "geom_name", label="visual_layers.livery"
            )
            if self._livery_geom_name != SURFACE_GUIDE_GEOM_NAME:
                raise ManifestError("visual_layers.livery.geom_name is unsupported")
            group = livery_contract.get("geom_group")
            if group != SURFACE_GUIDE_GEOM_GROUP or isinstance(group, bool):
                raise ManifestError(
                    f"visual_layers.livery.geom_group must be {SURFACE_GUIDE_GEOM_GROUP}"
                )
            self._livery_geom_group = group
            if livery_contract.get("default_visible") is not False:
                raise ManifestError("visual_layers.livery.default_visible must be false")
            if livery_contract.get("toggle_key") != "G":
                raise ManifestError("visual_layers.livery.toggle_key must be 'G'")
            if livery_contract.get("physics") is not False:
                raise ManifestError("visual_layers.livery.physics must be false")
            self._livery_toggle_key = "G"

        requested_lighting = "flat" if lighting is None else str(lighting).lower()
        if requested_lighting not in _LIGHTING_MODES:
            raise ValueError("lighting must be 'flat' or 'shadow'")
        if not self.lighting_available and requested_lighting != "flat":
            raise ValueError("shadow lighting is unavailable in this runtime pack")
        self._lighting_mode = requested_lighting

        requested_livery = _livery_state(livery)
        if not self.livery_available and requested_livery:
            raise ValueError("livery is unavailable in this runtime pack")
        self._livery_visible = requested_livery

    @property
    def lighting_available(self) -> bool:
        """Whether the pack declares a uniquely named field lighting control."""

        return self._key_light_name is not None

    @property
    def lighting_mode(self) -> str:
        """Current requested lighting mode: ``flat`` or ``shadow``."""

        return self._lighting_mode

    @property
    def livery_available(self) -> bool:
        """Whether the pack declares a field-owned livery display group."""

        return self._livery_geom_name is not None

    @property
    def livery_visible(self) -> bool:
        """Current requested livery visibility."""

        return self._livery_visible

    @property
    def status(self) -> dict[str, object]:
        """Return the current controller state as a JSON-safe mapping."""

        return {
            "lighting_available": self.lighting_available,
            "lighting_mode": self._lighting_mode,
            "livery_available": self.livery_available,
            "livery_visible": self._livery_visible,
        }

    def press_name(self, name: str) -> bool:
        """Handle a case-insensitive key name and report whether it was consumed."""

        key = str(name).strip().upper()
        if self.lighting_available and key == self._lighting_toggle_key:
            self.toggle_lighting()
            return True
        if self.livery_available and key == self._livery_toggle_key:
            self.toggle_livery()
            return True
        return False

    def toggle_lighting(self) -> str:
        """Toggle the requested field lighting mode and return the new mode."""

        if self.lighting_available:
            self._lighting_mode = "shadow" if self._lighting_mode == "flat" else "flat"
        return self._lighting_mode

    def toggle_livery(self) -> bool:
        """Toggle requested livery visibility and return the new value."""

        if self.livery_available:
            self._livery_visible = not self._livery_visible
        return self._livery_visible

    def apply(self, model: Any, geomgroup: MutableSequence[int] | Any) -> dict[str, object]:
        """Synchronize requested display state to a model and viewer geom groups.

        All model lookups and group checks complete before either setting is
        changed.  Ambiguous attached names therefore fail closed without a
        partial update.
        """

        mujoco = _mujoco()
        key_light_id: int | None = None
        livery_geom_id: int | None = None

        if self._key_light_name is not None:
            key_light_id = _unique_object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_LIGHT,
                int(model.nlight),
                self._key_light_name,
                label="field key light",
            )
            assert self._fill_light_name is not None
            _unique_object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_LIGHT,
                int(model.nlight),
                self._fill_light_name,
                label="field fill light",
            )
        if self._livery_geom_name is not None:
            livery_geom_id = _optional_unique_object_id(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(model.ngeom),
                self._livery_geom_name,
                label="field livery geom",
            )
            if livery_geom_id is None and self._livery_visible:
                raise MujocoModelError(
                    "cannot enable the field livery because this runtime profile "
                    f"does not contain {self._livery_geom_name!r}"
                )
        if livery_geom_id is not None:
            assert self._livery_geom_group is not None
            actual_group = int(model.geom_group[livery_geom_id])
            if actual_group != self._livery_geom_group:
                raise MujocoModelError(
                    "field livery geom group disagrees with manifest: "
                    f"expected={self._livery_geom_group}, actual={actual_group}"
                )
            group_conflicts = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom[{geom_id}]"
                for geom_id in range(int(model.ngeom))
                if geom_id != livery_geom_id
                and int(model.geom_group[geom_id]) == self._livery_geom_group
            ]
            if group_conflicts:
                raise MujocoModelError(
                    "field livery display group is also used by other geoms: "
                    + ", ".join(group_conflicts)
                )
            try:
                group_count = len(geomgroup)
            except TypeError as exc:
                raise MujocoModelError("viewer geomgroup must be a mutable sequence") from exc
            if self._livery_geom_group >= group_count:
                raise MujocoModelError(
                    "viewer geomgroup does not contain the declared livery group "
                    f"{self._livery_geom_group}"
                )
            if not hasattr(geomgroup, "__setitem__"):
                raise MujocoModelError("viewer geomgroup is not mutable")

        if key_light_id is not None:
            model.light_castshadow[key_light_id] = self._lighting_mode == "shadow"
        if livery_geom_id is not None:
            try:
                geomgroup[self._livery_geom_group] = int(self._livery_visible)
            except (IndexError, TypeError, ValueError) as exc:
                raise MujocoModelError("viewer geomgroup is not mutable") from exc

        return self.status

    @staticmethod
    def _lighting_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
        visual_display = manifest.get("visual_display")
        if visual_display is None:
            return None
        if not isinstance(visual_display, Mapping):
            raise ManifestError("visual_display must be an object")
        lighting = visual_display.get("lighting")
        if lighting is None:
            return None
        if not isinstance(lighting, Mapping):
            raise ManifestError("visual_display.lighting must be an object")
        return lighting

    @staticmethod
    def _livery_contract(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
        visual_layers = manifest.get("visual_layers")
        if visual_layers is None:
            return None
        if not isinstance(visual_layers, Mapping):
            raise ManifestError("visual_layers must be an object")
        livery = visual_layers.get("livery")
        if livery is None:
            return None
        if not isinstance(livery, Mapping):
            raise ManifestError("visual_layers.livery must be an object")
        return livery


def _required_string(contract: Mapping[str, Any], key: str, *, label: str) -> str:
    value = contract.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{label}.{key} must be a non-empty string")
    return value


def _livery_state(value: str | bool | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized not in _LIVERY_STATES:
        raise ValueError("livery must be 'off' or 'on'")
    return normalized == "on"


def _unique_object_id(
    mujoco: Any,
    model: Any,
    object_type: Any,
    count: int,
    declared_name: str,
    *,
    label: str,
) -> int:
    suffix = f"/{declared_name}"
    matches = []
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        if name == declared_name or (name is not None and name.endswith(suffix)):
            matches.append((object_id, name))
    if not matches:
        raise MujocoModelError(f"MuJoCo model does not contain {label} {declared_name!r}")
    if len(matches) != 1:
        names = ", ".join(repr(name) for _object_id, name in matches)
        raise MujocoModelError(f"MuJoCo model contains ambiguous {label} names: {names}")
    return matches[0][0]


def _optional_unique_object_id(
    mujoco: Any,
    model: Any,
    object_type: Any,
    count: int,
    declared_name: str,
    *,
    label: str,
) -> int | None:
    """Return a unique declared object, allowing an omitted runtime-profile layer."""

    suffix = f"/{declared_name}"
    matches = []
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        if name == declared_name or (name is not None and name.endswith(suffix)):
            matches.append((object_id, name))
    if not matches:
        return None
    if len(matches) != 1:
        names = ", ".join(repr(name) for _object_id, name in matches)
        raise MujocoModelError(f"MuJoCo model contains ambiguous {label} names: {names}")
    return matches[0][0]


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise MujocoModelError("MuJoCo is not installed") from exc
    return mujoco
