"""Independent uphill/downhill results for one ordinary-slope episode."""

from copy import deepcopy


SLOPE_TRAVERSAL_RULE = {
    "version": 2,
    "endpoint_tolerance_m": 0.12,
    "endpoint_dwell_s": 0.0,
    "endpoint_requires_contact": True,
    "roundtrip": "uphill_then_downhill_in_same_episode",
}


class SlopeProgress:
    """Record each leg once; resets and different runs never combine into a trip."""

    def __init__(self, route, direction="uphill"):
        if direction not in {"uphill", "downhill", "roundtrip"}:
            raise ValueError("direction must be uphill, downhill or roundtrip")
        self.length = route["length_m"]
        self.direction = direction
        self.phase = "downhill" if direction == "downhill" else "uphill"
        self.legs = {
            leg: {
                "status": "INCOMPLETE" if direction in {leg, "roundtrip"} else "NOT_REQUESTED",
                "started_at_s": 0.0 if leg == self.phase else None,
                "completed_at_s": None,
                "duration_s": None,
                "field_contact_steps": 0,
                "max_tilt_deg": 0.0,
                "max_penetration_m": 0.0,
                "first_failure": None,
            }
            for leg in ("uphill", "downhill")
        }

    def update(self, *, time_s, along_m, contacts, tilt_deg, penetration_m, failure=None):
        """Consume a physics step after the session's existing route/physics checks.

        The 12 cm endpoint footprint also accepts passing beyond its target.
        Contact is required when recording arrival, but stopping is not.
        Completed results remain intact if subsequent driving has a problem.
        """
        leg = self.legs[self.phase]
        if leg["status"] != "INCOMPLETE":
            return
        leg["duration_s"] = time_s - leg["started_at_s"]
        leg["field_contact_steps"] += bool(contacts)
        leg["max_tilt_deg"] = max(leg["max_tilt_deg"], tilt_deg)
        leg["max_penetration_m"] = max(leg["max_penetration_m"], penetration_m)
        if failure:
            leg["status"] = "FAIL"
            leg["first_failure"] = dict(failure)
            return
        tolerance = SLOPE_TRAVERSAL_RULE["endpoint_tolerance_m"]
        arrived = (
            along_m >= self.length - tolerance if self.phase == "uphill" else along_m <= tolerance
        )
        if not (arrived and contacts and tilt_deg < 60):
            return
        leg["status"] = "COMPLETE"
        leg["completed_at_s"] = time_s
        if self.direction == "roundtrip" and self.phase == "uphill":
            self.phase = "downhill"
            self.legs["downhill"]["started_at_s"] = time_s

    @property
    def status(self):
        requested = [
            leg["status"] for leg in self.legs.values() if leg["status"] != "NOT_REQUESTED"
        ]
        if "FAIL" in requested:
            return "FAIL"
        return "COMPLETE" if all(status == "COMPLETE" for status in requested) else "INCOMPLETE"

    def report(self):
        return {
            "leg_results": deepcopy(self.legs),
            "traversal_status": self.status,
            "roundtrip_status": self.status if self.direction == "roundtrip" else "NOT_REQUESTED",
            "traversal_rule": dict(SLOPE_TRAVERSAL_RULE),
        }

    def status_line(self, time_s):
        roundtrip = self.status if self.direction == "roundtrip" else "NOT_REQUESTED"
        return (
            f"SLOPE_PROGRESS t={time_s:.2f}s direction={self.direction} "
            f"uphill={self.legs['uphill']['status']} "
            f"downhill={self.legs['downhill']['status']} roundtrip={roundtrip}"
        )
