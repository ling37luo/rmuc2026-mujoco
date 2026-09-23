from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from rmuc2026_mujoco.errors import AssetIntegrityError, ManifestError
from rmuc2026_mujoco.fly_batch import run_fly_batch
from rmuc2026_mujoco.isaac import (
    export_isaac_training_region,
    load_isaac_training_region,
    load_isaac_training_region_export,
)
from rmuc2026_mujoco.manifest import sha256_file
from rmuc2026_mujoco.query import HeightFieldData
from rmuc2026_mujoco.training_region import (
    TrainingRegion,
    _heightfield_png,
    _profile_hash,
    compose_training_region_with_robot,
    export_training_region,
    load_training_region_model,
)


def _fake_source(tmp_path: Path, monkeypatch) -> tuple[SimpleNamespace, HeightFieldData]:
    source_xml = tmp_path / "source.xml"
    source_xml.write_text(
        '<mujoco model="source"><option timestep="0.002" solver="Newton"/>'
        '<asset><hfield name="rmuc2026_collision" file="unused.png" '
        'nrow="801" ncol="801" size="4 4 1 0.05"/></asset>'
        '<worldbody><geom name="rmuc2026_field_collision" type="hfield" '
        'hfield="rmuc2026_collision" pos="0 0 0" contype="2" conaffinity="1" '
        'friction="1 0.005 0.0001" solref="0.02 1"/>'
        '<geom name="rmuc2026_perimeter_top" type="box" pos="0 0.7 0.5" '
        'size="4 0.02 0.5" contype="2" conaffinity="1" '
        'friction="1 0.005 0.0001" solref="0.04 1"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    axis = np.arange(-4.0, 4.001, 0.01)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    world = HeightFieldData(axis, axis, 0.10 + 0.01 * xx + 0.005 * yy)
    asset = SimpleNamespace(
        hashes_verified=True,
        manifest_sha256="a" * 64,
        manifest={"validation_status": "DRAFT_BLOCKED", "schema_version": 3},
        collision={
            "samples_sha256": "b" * 64,
            "minimum_height_m": 0.0,
            "maximum_height_m": 1.0,
        },
        recommended_spawn={"terrain_height_m": 0.0},
        entrypoint_for=lambda _profile: source_xml,
    )
    route = {
        "route_id": "test_route",
        "uphill_unit_xy": [1.0, 0.0],
        "heading_yaw_rad": 0.0,
        "surface_width_m": 0.8,
        "approach_distance_m": 0.9,
        "approach_xyz_m": [-0.5, 0.0, 0.08],
        "low_seam_xyz_m": [-0.3, 0.0, 0.08],
        "takeoff_xyz_m": [0.0, 0.0, 0.1],
        "landing_edge_xyz_m": [0.2, 0.0, 0.12],
        "landing_target_xyz_m": [0.5, 0.0, 0.12],
        "spawn": {"xyz_m": [-0.5, 0.0, 0.08], "heading_yaw_rad": 0.0},
    }
    monkeypatch.setattr("rmuc2026_mujoco.training_region.fly_route_descriptor", lambda *_: route)
    monkeypatch.setattr("rmuc2026_mujoco.training_region.load_heightfield", lambda *_: world)
    monkeypatch.setattr(
        "rmuc2026_mujoco.training_region.scenario_descriptor",
        lambda *_args, **_kwargs: {"profile_hash": "c" * 64},
    )
    return asset, world


def test_training_region_exports_exact_source_grid_and_composes(tmp_path, monkeypatch):
    asset, source = _fake_source(tmp_path, monkeypatch)
    output = tmp_path / "region"
    manifest = export_training_region(asset, output, scenario_id="fly_ramp_north")
    region = TrainingRegion.open(output)
    rows, columns = manifest["source"]["source_grid_slice_yx"]
    local = region.heightfield()
    np.testing.assert_array_equal(local.x_m, source.x_m[columns[0] : columns[1]])
    np.testing.assert_array_equal(local.y_m, source.y_m[rows[0] : rows[1]])
    np.testing.assert_array_equal(
        local.height_m, source.height_m[rows[0] : rows[1], columns[0] : columns[1]]
    )
    assert local.height_m.size < source.height_m.size / 2
    assert manifest["validation_status"] == "DRAFT_BLOCKED"
    assert local.x_m[-1] >= 3.0  # landing target plus 2.5 m runout
    (lower_x, lower_y), (upper_x, upper_y) = local.bounds_xy_m
    assert region.contains_footprint(0.0, 0.0, radius_m=0.2)
    assert region.contains_footprint(lower_x + 0.2, lower_y + 0.2, radius_m=0.2)
    assert not region.contains_footprint(lower_x + 0.19, 0.0, radius_m=0.2)
    assert not region.contains_footprint(upper_x + 0.01, upper_y, radius_m=0.0)
    assert not region.contains_footprint(float("nan"), 0.0, radius_m=0.2)
    with pytest.raises(ValueError, match="footprint radius"):
        region.contains_footprint(0.0, 0.0, radius_m=-0.1)

    model, _ = load_training_region_model(region)
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.nhfield == 1
    assert model.ngeom == 2  # source heightfield and unchanged nearby perimeter
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "rmuc2026_collision")
    address = int(model.hfield_adr[hfield_id])
    injected = model.hfield_data[address : address + local.height_m.size].reshape(
        local.height_m.shape
    )
    np.testing.assert_allclose(
        injected,
        local.height_m,
        atol=2e-7,
        rtol=0,
    )
    assert manifest["source"]["source_vertical_normalization"]["maximum_source_height_m"] == 1.0

    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco model="robot"><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".06"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    composed, _ = compose_training_region_with_robot(region, robot)
    assert composed.nhfield == 1
    assert composed.ngeom == 3
    isaac = load_isaac_training_region(region)
    assert isaac.scenario["profile_hash"] == manifest["profile_hash"]
    assert isaac.scenario["source_manifest_sha256"] == asset.manifest_sha256
    np.testing.assert_allclose(isaac.height_m, local.height_m, atol=1e-7, rtol=0)
    assert len(isaac.static_boxes) == 1
    assert isaac.heightfield_contact == {
        "name": "rmuc2026_field_collision",
        "contype": 2,
        "conaffinity": 1,
        "friction": [1.0, 0.005, 0.0001],
        "solref": [0.02, 1.0],
    }
    assert isaac.to_dict()["heightfield_contact"] == isaac.heightfield_contact

    with pytest.raises(ValueError, match="already exists"):
        export_training_region(asset, output, scenario_id="fly_ramp_north")


def test_isaac_offline_export_preserves_exact_grid_scale_and_identity(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    region_path = tmp_path / "region"
    export_training_region(asset, region_path, scenario_id="fly_ramp_north")
    region = TrainingRegion.open(region_path)
    export_path = tmp_path / "offline"
    descriptor = export_isaac_training_region(region, export_path)
    imported = load_isaac_training_region_export(export_path, source_region=region)
    original = region.heightfield()

    assert descriptor["identity"]["region_profile_hash"] == region.manifest["profile_hash"]
    assert descriptor["identity"]["source_manifest_sha256"] == asset.manifest_sha256
    assert descriptor["identity"]["source_field_xml_sha256"] == sha256_file(
        region_path / "field.xml"
    )
    assert descriptor["schema_version"] == 3
    assert descriptor["grid_file_sha256"] == sha256_file(region_path / "collision/heightfield.npz")
    assert descriptor["grid"]["height_values"] == "absolute_world_z_m_no_extra_scale_or_offset"
    np.testing.assert_array_equal(imported.x_m, original.x_m)
    np.testing.assert_array_equal(imported.y_m, original.y_m)
    np.testing.assert_array_equal(imported.height_m, original.height_m)
    assert imported.bounds_xy_m == original.bounds_xy_m
    assert imported.scenario["profile_hash"] == region.manifest["profile_hash"]
    assert imported.scenario["heightfield_contact_status"] == "source_mjcf_recorded"
    assert imported.static_boxes == load_isaac_training_region(region).static_boxes
    assert descriptor["collision"]["heightfield"] == imported.heightfield_contact
    assert imported.heightfield_contact == load_isaac_training_region(region).heightfield_contact
    box = imported.static_boxes[0]
    assert box["name"] == "rmuc2026_perimeter_top"
    assert box["pos_m"] == pytest.approx([(original.x_m[0] + original.x_m[-1]) / 2, 0.7, 0.5])
    assert box["size_m"] == pytest.approx([(original.x_m[-1] - original.x_m[0]) / 2, 0.02, 0.5])
    assert box["contype"] == 2 and box["conaffinity"] == 1
    assert box["friction"] == [1.0, 0.005, 0.0001]
    assert box["solref"] == [0.04, 1.0]
    with pytest.raises(ValueError, match="already exists"):
        export_isaac_training_region(region, export_path)


def test_isaac_heightfield_contact_follows_source_mjcf(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    source_xml = asset.entrypoint_for("collision_only")
    tree = ET.parse(source_xml)
    geom = tree.getroot().find("./worldbody/geom[@name='rmuc2026_field_collision']")
    assert geom is not None
    geom.set("contype", "4")
    geom.set("conaffinity", "3")
    geom.set("friction", "0.73 0.004 0.0002")
    geom.set("solref", "0.031 0.8")
    tree.write(source_xml, encoding="utf-8")

    region_path = tmp_path / "region"
    export_training_region(asset, region_path, scenario_id="fly_ramp_north")
    expected = {
        "name": "rmuc2026_field_collision",
        "contype": 4,
        "conaffinity": 3,
        "friction": [0.73, 0.004, 0.0002],
        "solref": [0.031, 0.8],
    }
    assert load_isaac_training_region(region_path).heightfield_contact == expected
    export_path = tmp_path / "offline"
    descriptor = export_isaac_training_region(region_path, export_path)
    assert descriptor["collision"]["heightfield"] == expected
    assert (
        load_isaac_training_region_export(
            export_path, source_region=region_path
        ).heightfield_contact
        == expected
    )


def test_isaac_offline_export_rejects_grid_or_source_identity_change(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    region_path = tmp_path / "region"
    export_training_region(asset, region_path, scenario_id="fly_ramp_south")
    export_path = tmp_path / "offline"
    export_isaac_training_region(region_path, export_path)
    grid_path = export_path / "heightfield.npz"
    grid_path.write_bytes(grid_path.read_bytes() + b"tampered")
    with pytest.raises(AssetIntegrityError, match="hash mismatch"):
        load_isaac_training_region_export(export_path)

    grid_path.write_bytes((region_path / "collision/heightfield.npz").read_bytes())
    descriptor_path = export_path / "descriptor.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["grid"]["bounds_xy_m"][0][0] += 0.01
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ManifestError, match="geometry disagrees"):
        load_isaac_training_region_export(export_path)

    descriptor["grid"]["bounds_xy_m"][0][0] -= 0.01
    descriptor["identity"]["source_profile_hash"] = "0" * 64
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ManifestError, match="verified source region"):
        load_isaac_training_region_export(export_path, source_region=region_path)


def test_isaac_contact_is_source_bound_and_legacy_schemas_remain_readable(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    region_path = tmp_path / "region"
    export_training_region(asset, region_path, scenario_id="fly_ramp_north")
    export_path = tmp_path / "offline"
    descriptor = export_isaac_training_region(region_path, export_path)
    descriptor_path = export_path / "descriptor.json"
    assert len(descriptor["collision"]["static_boxes"]) == 1

    changed = json.loads(descriptor_path.read_text(encoding="utf-8"))
    changed["collision"]["static_boxes"][0]["pos_m"][1] -= 0.001
    descriptor_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ManifestError, match="verified source region"):
        load_isaac_training_region_export(export_path, source_region=region_path)

    changed = json.loads(descriptor_path.read_text(encoding="utf-8"))
    changed["identity"]["source_field_xml_sha256"] = "0" * 64
    descriptor_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ManifestError, match="verified source region"):
        load_isaac_training_region_export(export_path, source_region=region_path)

    for key, replacement in (
        ("contype", 3),
        ("conaffinity", 2),
        ("friction", [0.7, 0.005, 0.0001]),
        ("solref", [0.04, 1.0]),
    ):
        changed = json.loads(json.dumps(descriptor))
        changed["collision"]["heightfield"][key] = replacement
        descriptor_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ManifestError, match="verified source region"):
            load_isaac_training_region_export(export_path, source_region=region_path)

    changed = json.loads(json.dumps(descriptor))
    changed["collision"]["heightfield"]["name"] = "wrong_heightfield"
    descriptor_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ManifestError, match="heightfield contact"):
        load_isaac_training_region_export(export_path, source_region=region_path)

    changed = json.loads(json.dumps(descriptor))
    changed["collision"].pop("heightfield")
    descriptor_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ManifestError, match="collision descriptor"):
        load_isaac_training_region_export(export_path, source_region=region_path)

    schema_two = json.loads(json.dumps(descriptor))
    schema_two["schema_version"] = 2
    schema_two["collision"].pop("heightfield")
    descriptor_path.write_text(json.dumps(schema_two), encoding="utf-8")
    imported = load_isaac_training_region_export(export_path, source_region=region_path)
    assert len(imported.static_boxes) == 1
    assert imported.heightfield_contact is None
    assert imported.scenario["heightfield_contact_status"] == "not_recorded_legacy_schema"
    assert imported.to_dict()["heightfield_contact"] is None

    legacy = dict(descriptor)
    legacy["schema_version"] = 1
    legacy["scope"] = "offline_heightfield_data_only_no_physx_contact_validation"
    legacy.pop("collision")
    legacy["identity"] = dict(legacy["identity"])
    legacy["identity"].pop("source_field_xml_sha256")
    descriptor_path.write_text(json.dumps(legacy), encoding="utf-8")
    imported = load_isaac_training_region_export(export_path, source_region=region_path)
    assert imported.static_boxes == ()
    assert imported.heightfield_contact is None
    assert imported.scenario["heightfield_contact_status"] == "not_recorded_legacy_schema"


def test_training_region_rejects_tampered_collision_file(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    output = tmp_path / "region"
    export_training_region(asset, output, scenario_id="fly_ramp_south")
    path = output / "collision/heightfield.png"
    data = bytearray(path.read_bytes())
    data[-10] ^= 1
    path.write_bytes(data)
    with pytest.raises(AssetIntegrityError, match="hash mismatch"):
        TrainingRegion.open(output)


def test_training_region_rejects_rehashed_inconsistent_bootstrap(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    output = tmp_path / "region"
    export_training_region(asset, output, scenario_id="fly_ramp_north")
    region = TrainingRegion.open(output)
    image = output / "collision/heightfield.png"
    image.write_bytes(_heightfield_png(np.zeros_like(region.heightfield().height_m)))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["collision/heightfield.png"] = {
        "sha256": sha256_file(image),
        "size_bytes": image.stat().st_size,
    }
    manifest["profile_hash"] = _profile_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssetIntegrityError, match="PNG disagrees"):
        TrainingRegion.open(output)


def test_training_region_profile_identity_covers_mjcf_contact_parameters(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    output = tmp_path / "region"
    export_training_region(asset, output, scenario_id="fly_ramp_north")
    original_hash = TrainingRegion.open(output).manifest["profile_hash"]
    xml_path = output / "field.xml"
    xml_path.write_text(
        xml_path.read_text(encoding="utf-8").replace(
            'friction="1 0.005 0.0001"', 'friction="0.05 0.005 0.0001"'
        ),
        encoding="utf-8",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["field.xml"] = {
        "sha256": sha256_file(xml_path),
        "size_bytes": xml_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="profile hash"):
        TrainingRegion.open(output)
    manifest["profile_hash"] = _profile_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert TrainingRegion.open(output).manifest["profile_hash"] != original_hash


def test_training_region_rejects_npz_axes_shifted_from_mjcf(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    output = tmp_path / "region"
    export_training_region(asset, output, scenario_id="fly_ramp_south")
    region = TrainingRegion.open(output)
    samples = region.heightfield()
    samples_path = output / "collision/heightfield.npz"
    np.savez_compressed(
        samples_path,
        x_m=samples.x_m + 1.0,
        y_m=samples.y_m,
        height_m=samples.height_m,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["collision/heightfield.npz"] = {
        "sha256": sha256_file(samples_path),
        "size_bytes": samples_path.stat().st_size,
    }
    manifest["source"]["heightfield_samples_sha256"] = sha256_file(samples_path)
    manifest["profile_hash"] = _profile_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManifestError, match="NPZ XY axes disagree"):
        TrainingRegion.open(output)


def test_training_region_runs_parallel_fly_episodes_with_fixed_route(tmp_path, monkeypatch):
    asset, _ = _fake_source(tmp_path, monkeypatch)
    output = tmp_path / "region"
    manifest = export_training_region(asset, output, scenario_id="fly_ramp_north")
    region = TrainingRegion.open(output)
    result = run_fly_batch(
        region,
        speeds=[0.3],
        repeats=2,
        workers=2,
        duration_s=0.02,
        footprint_radius_m=0.1,
    )
    assert result["status"] == "PASS"
    assert result["profile"] == "fly_ramp_training_region"
    assert result["source_manifest_sha256"] == asset.manifest_sha256
    assert result["training_region_manifest_sha256"] == region.manifest_sha256
    assert result["profile_hashes"] == {"fly_ramp_north": manifest["profile_hash"]}
    assert result["approach_distances_m"] == [0.9]
    assert result["footprint_radius_m"] == 0.1
    assert result["workers"] == 2 and result["summary"]["episodes"] == 2
    assert all(episode["physics_status"] == "PASS" for episode in result["episodes"])
    assert all(episode["footprint_radius_m"] == 0.1 for episode in result["episodes"])
    with pytest.raises(ValueError, match="one fixed scenario"):
        run_fly_batch(region, scenarios=["fly_ramp_south"], speeds=[0.3])
    with pytest.raises(ValueError, match="one fixed scenario"):
        run_fly_batch(region, approach_distances=[0.6], speeds=[0.3])
