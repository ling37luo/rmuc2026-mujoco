import pytest

from rmuc2026_mujoco.fly_progress import FlyRampProgress


ROUTE = {
    "low_seam_xyz_m": [0.0, 0.0, 0.2],
    "takeoff_xyz_m": [1.15, 0.0, 0.55],
    "landing_edge_xyz_m": [1.80, 0.0, 0.2],
    "uphill_unit_xy": [1.0, 0.0],
    "surface_width_m": 0.86,
}


def sample(progress, time_s, along, contacts=(), *, speed=2.2, tilt=3.0):
    progress.update(
        time_s=time_s,
        position_xyz_m=[along, 0.0, 0.3],
        forward_speed_mps=speed,
        contact_points_xyz_m=contacts,
        tilt_deg=tilt,
    )


def launch(progress):
    sample(progress, 0.1, 0.05, [[0.05, 0, 0.2]])
    sample(progress, 0.7, 1.12, [[1.10, 0, 0.54]])
    sample(progress, 0.8, 1.20)
    sample(progress, 0.808, 1.23)
    assert progress.phase == "flight"


def test_complete_flight_requires_airborne_top_contact_and_stability():
    progress = FlyRampProgress(ROUTE)
    launch(progress)
    sample(progress, 1.05, 1.90, [[1.84, 0, 0.2]])
    assert progress.phase == "landing"
    assert progress.status == "INCOMPLETE"
    sample(progress, 1.10, 2.01, [[1.98, 0, 0.2]])
    sample(progress, 1.61, 2.08, [[2.02, 0, 0.2]])
    report = progress.report()
    assert report["flight_status"] == "COMPLETE"
    assert report["actual_takeoff_speed_mps"] == pytest.approx(2.2)
    assert report["first_recontact"]["surface"] == "landing_top"
    assert report["events"]["takeoff_s"] == pytest.approx(0.8)


def test_gap_floor_recontact_is_not_landing_success():
    progress = FlyRampProgress(ROUTE)
    launch(progress)
    sample(progress, 1.0, 1.65, [[1.65, 0, 0.0]])
    assert progress.status == "FAIL"
    assert progress.failure_reason == "short_or_lip_impact"
    assert progress.first_recontact["surface"] == "gap_floor_or_lip"


def test_side_recontact_is_reported_separately():
    progress = FlyRampProgress(ROUTE)
    launch(progress)
    sample(progress, 1.0, 1.9, [[1.85, 0.50, 0.2]])
    assert progress.failure_reason == "side_impact"


def test_mixed_lip_and_top_contacts_do_not_count_as_clean_landing():
    progress = FlyRampProgress(ROUTE)
    launch(progress)
    sample(progress, 1.0, 1.9, [[1.75, 0, 0.12], [1.84, 0, 0.2]])
    assert progress.failure_reason == "mixed_lip_top_impact"
    assert progress.first_recontact["surface"] == "mixed_lip_top"


def test_obstacle_contact_prevents_false_flight_and_landing():
    progress = FlyRampProgress(ROUTE)
    sample(progress, 0.1, 0.05, [[0.05, 0, 0.2]])
    progress.update(
        time_s=0.8,
        position_xyz_m=[1.2, 0, 0.5],
        forward_speed_mps=2.2,
        contact_points_xyz_m=[],
        obstacle_contacts=1,
        tilt_deg=3,
    )
    assert progress.phase == "ramp"
    sample(progress, 0.9, 1.2)
    sample(progress, 0.908, 1.23)
    assert progress.phase == "flight"
    progress.update(
        time_s=1.1,
        position_xyz_m=[1.9, 0, 0.3],
        forward_speed_mps=2.0,
        contact_points_xyz_m=[[1.85, 0, 0.2]],
        obstacle_contacts=1,
        tilt_deg=4,
    )
    assert progress.failure_reason == "obstacle_impact"
    assert progress.first_recontact["surface"] == "obstacle"


def test_early_takeoff_records_actual_loss_of_contact():
    progress = FlyRampProgress(ROUTE)
    sample(progress, 0.1, 0.05, [[0.05, 0, 0.2]])
    sample(progress, 0.5, 0.70, speed=1.9)
    sample(progress, 0.65, 1.10, speed=2.1)
    assert progress.phase == "flight"
    report = progress.report()
    assert report["events"]["takeoff_s"] == pytest.approx(0.5)
    assert report["actual_takeoff_speed_mps"] == pytest.approx(1.9)
    assert report["takeoff_before_lip_m"] == pytest.approx(0.45)


def test_crossing_without_flight_does_not_complete_jump():
    progress = FlyRampProgress(ROUTE)
    sample(progress, 0.1, 0.05, [[0.05, 0, 0.2]])
    sample(progress, 1.0, 1.95, [[1.90, 0, 0.2]])
    assert progress.status == "FAIL"
    assert progress.failure_reason == "no_airborne_phase"


def test_landing_contact_loss_and_timeout_are_separate():
    progress = FlyRampProgress(ROUTE)
    assert progress.timeout_reason() == "did_not_reach_ramp"
    launch(progress)
    assert progress.timeout_reason() == "no_landing_contact"
    sample(progress, 1.05, 1.90, [[1.84, 0, 0.2]])
    assert progress.timeout_reason() == "landing_not_stable"
    sample(progress, 1.35, 2.00)
    assert progress.status == "FAIL"
    assert progress.failure_reason == "lost_landing_contact"
