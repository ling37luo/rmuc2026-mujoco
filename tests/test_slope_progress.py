import pytest

from rmuc2026_mujoco.cli import build_parser
from rmuc2026_mujoco.scenarios import get_scenario
from rmuc2026_mujoco.slope_progress import SlopeProgress


ROUTE = {"length_m": 1.69}


def sample(progress, t, along, **kwargs):
    progress.update(
        **dict(time_s=t, along_m=along, contacts=2, tilt_deg=3.0, penetration_m=0.001, **kwargs)
    )


def test_fast_uphill_pass_records_only_uphill_without_stopping():
    progress = SlopeProgress(ROUTE)
    sample(progress, 0.02, 0)
    sample(progress, 1.0, 1.55)
    sample(progress, 1.2, 1.80)
    report = progress.report()
    assert report["traversal_status"] == "COMPLETE"
    assert report["leg_results"]["uphill"]["completed_at_s"] == 1.2
    assert report["leg_results"]["downhill"]["status"] == "NOT_REQUESTED"
    assert report["roundtrip_status"] == "NOT_REQUESTED"
    # Driving back in an uphill-only task does not invent a roundtrip result.
    sample(progress, 3.0, -0.1)
    assert progress.report() == report


def test_downhill_starts_at_high_end_and_never_credits_uphill():
    progress = SlopeProgress(ROUTE, "downhill")
    sample(progress, 0.02, 1.69)
    assert progress.status == "INCOMPLETE"
    sample(progress, 2.0, -0.1)
    report = progress.report()
    assert report["leg_results"]["downhill"]["completed_at_s"] == 2.0
    assert report["leg_results"]["uphill"]["status"] == "NOT_REQUESTED"
    assert report["roundtrip_status"] == "NOT_REQUESTED"


def test_roundtrip_requires_both_legs_and_keeps_separate_metrics():
    progress = SlopeProgress(ROUTE, "roundtrip")
    sample(progress, 0.1, 0)
    sample(progress, 2.0, 1.7)
    uphill_report = progress.report()
    assert uphill_report["leg_results"]["uphill"]["status"] == "COMPLETE"
    assert uphill_report["leg_results"]["downhill"]["status"] == "INCOMPLETE"
    assert uphill_report["roundtrip_status"] == "INCOMPLETE"
    # Continuing past the crest and waiting there is not downhill success.
    sample(progress, 3.0, 1.9)
    assert progress.status == "INCOMPLETE"
    progress.update(time_s=4.0, along_m=0.1, contacts=4, tilt_deg=5, penetration_m=0.002)
    report = progress.report()
    assert report["roundtrip_status"] == report["traversal_status"] == "COMPLETE"
    up, down = report["leg_results"]["uphill"], report["leg_results"]["downhill"]
    assert up["started_at_s"] == 0 and up["completed_at_s"] == 2
    assert down["started_at_s"] == 2 and down["completed_at_s"] == 4
    assert up["duration_s"] == down["duration_s"] == 2
    assert up["max_tilt_deg"] == 3 and down["max_tilt_deg"] == 5
    assert up["max_penetration_m"] == 0.001 and down["max_penetration_m"] == 0.002
    assert up["field_contact_steps"] == down["field_contact_steps"] == 2
    assert uphill_report["leg_results"]["downhill"]["status"] == "INCOMPLETE"


@pytest.mark.parametrize("reason", ["outside_route", "robot_tipped", "numerical_instability"])
def test_downhill_failure_preserves_completed_uphill(reason):
    progress = SlopeProgress(ROUTE, "roundtrip")
    sample(progress, 2.0, 1.8)
    failure = {"reason": reason, "time_s": 3.0}
    sample(progress, 3.0, 0.0, failure=failure)
    sample(progress, 4.0, 0.0)
    report = progress.report()
    assert report["leg_results"]["uphill"]["status"] == "COMPLETE"
    assert report["leg_results"]["downhill"]["status"] == "FAIL"
    assert report["leg_results"]["downhill"]["completed_at_s"] is None
    assert report["leg_results"]["downhill"]["first_failure"] == failure
    assert report["roundtrip_status"] == "FAIL"


def test_uphill_failure_does_not_start_downhill():
    progress = SlopeProgress(ROUTE, "roundtrip")
    sample(progress, 1.0, 1.8, failure={"reason": "outside_route", "time_s": 1.0})
    report = progress.report()
    assert report["leg_results"]["uphill"]["status"] == "FAIL"
    assert report["leg_results"]["downhill"]["started_at_s"] is None
    assert report["roundtrip_status"] == "FAIL"


def test_crossing_in_air_waits_for_contact_beyond_endpoint():
    progress = SlopeProgress(ROUTE)
    progress.update(time_s=1, along_m=1.8, contacts=0, tilt_deg=3, penetration_m=0)
    assert progress.status == "INCOMPLETE"
    sample(progress, 1.1, 1.9)
    assert progress.legs["uphill"]["completed_at_s"] == 1.1


def test_new_episode_does_not_combine_two_separate_runs():
    first = SlopeProgress(ROUTE, "roundtrip")
    sample(first, 1, 1.8)
    saved = first.report()
    second = SlopeProgress(ROUTE, "downhill")
    sample(second, 1, 0)
    assert saved["roundtrip_status"] == "INCOMPLETE"
    assert second.report()["roundtrip_status"] == "NOT_REQUESTED"
    assert SlopeProgress(ROUTE, "roundtrip").legs["uphill"]["completed_at_s"] is None


def test_slope_cli_and_registry_default_to_uphill():
    args = build_parser().parse_args(["view", "field", "--scenario", "slope_basic"])
    assert args.direction == "uphill"
    spec = get_scenario("slope_basic")
    assert spec.spawn["default_direction"] == "uphill"
    assert spec.spawn["directions"] == ["uphill", "downhill", "roundtrip"]
    assert spec.spawn["traversal_rule"]["endpoint_dwell_s"] == 0
