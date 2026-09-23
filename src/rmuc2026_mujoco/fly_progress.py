"""Task progress for a complete robot flight over one audited fly ramp."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class FlyRampProgress:
    """Separate launch, flight and landing outcomes from physics health."""

    def __init__(self, route: dict[str, Any], *, stable_duration_s: float = 0.5):
        self.route = route
        self.stable_duration_s = stable_duration_s
        self.origin = np.asarray(route["low_seam_xyz_m"][:2], dtype=float)
        self.uphill = np.asarray(route["uphill_unit_xy"], dtype=float)
        self.lateral = np.array([-self.uphill[1], self.uphill[0]])
        self.takeoff_along_m = self._along(route["takeoff_xyz_m"])
        self.landing_edge_along_m = self._along(route["landing_edge_xyz_m"])
        self.landing_top_height_m = float(route["landing_edge_xyz_m"][2])
        self.width_m = float(route["surface_width_m"])
        self.phase = "approach"
        self.events: dict[str, float] = {}
        self.takeoff_speed_mps: float | None = None
        self.takeoff_lateral_m: float | None = None
        self.first_recontact: dict[str, float | str] | None = None
        self.failure_reason: str | None = None
        self._contact_loss_since: float | None = None
        self._contact_loss_speed: float | None = None
        self._contact_loss_lateral: float | None = None
        self._contact_loss_along: float | None = None
        self._stable_since: float | None = None
        self._last_top_contact_s: float | None = None

    def _along(self, position_xyz_m: Sequence[float]) -> float:
        return float((np.asarray(position_xyz_m[:2]) - self.origin) @ self.uphill)

    def _across(self, position_xyz_m: Sequence[float]) -> float:
        return float((np.asarray(position_xyz_m[:2]) - self.origin) @ self.lateral)

    def _top_contact(self, point_xyz_m: Sequence[float]) -> bool:
        return (
            self._along(point_xyz_m) >= self.landing_edge_along_m + 0.01
            and abs(self._across(point_xyz_m)) <= self.width_m / 2
            and float(point_xyz_m[2]) >= self.landing_top_height_m - 0.03
        )

    @property
    def status(self) -> str:
        if self.phase == "complete":
            return "COMPLETE"
        if self.phase == "failed":
            return "FAIL"
        return "INCOMPLETE"

    def fail(self, reason: str, time_s: float) -> None:
        if self.status == "INCOMPLETE":
            self.failure_reason = reason
            self.events["failed_s"] = time_s
            self.phase = "failed"

    def update(
        self,
        *,
        time_s: float,
        position_xyz_m: Sequence[float],
        forward_speed_mps: float,
        contact_points_xyz_m: Sequence[Sequence[float]],
        tilt_deg: float,
        obstacle_contacts: int = 0,
        failure: str | None = None,
    ) -> None:
        if self.status != "INCOMPLETE":
            return
        if failure is not None:
            self.fail(failure, time_s)
            return
        along = self._along(position_xyz_m)
        contacts = bool(contact_points_xyz_m) or obstacle_contacts > 0
        if self.phase == "approach" and along >= 0.0 and contacts:
            self.events["ramp_entry_s"] = time_s
            self.phase = "ramp"
        if self.phase == "ramp":
            if not contacts:
                if self._contact_loss_since is None:
                    self._contact_loss_since = time_s
                    self._contact_loss_speed = float(forward_speed_mps)
                    self._contact_loss_lateral = self._across(position_xyz_m)
                    self._contact_loss_along = along
                if (
                    along >= self.takeoff_along_m - 0.10
                    and time_s - self._contact_loss_since >= 0.006
                ):
                    self.events["takeoff_s"] = self._contact_loss_since
                    self.takeoff_speed_mps = self._contact_loss_speed
                    self.takeoff_lateral_m = self._contact_loss_lateral
                    self.phase = "flight"
            else:
                self._contact_loss_since = None
                self._contact_loss_speed = None
                self._contact_loss_lateral = None
                self._contact_loss_along = None
            if self.phase == "ramp" and along > self.landing_edge_along_m + 0.10 and contacts:
                self.fail("no_airborne_phase", time_s)
        if self.phase == "flight" and contacts:
            if obstacle_contacts:
                self.first_recontact = {"time_s": time_s, "surface": "obstacle"}
                self.events["first_recontact_s"] = time_s
                self.fail("obstacle_impact", time_s)
                return
            first = np.asarray(contact_points_xyz_m[0], dtype=float)
            top_contacts = [self._top_contact(point) for point in contact_points_xyz_m]
            top = all(top_contacts)
            mixed = any(top_contacts) and not top
            self.first_recontact = {
                "time_s": time_s,
                "along_m": self._along(first),
                "lateral_m": self._across(first),
                "height_m": float(first[2]),
                "surface": (
                    "landing_top" if top else "mixed_lip_top" if mixed else "gap_floor_or_lip"
                ),
            }
            self.events["first_recontact_s"] = time_s
            if not top:
                side = not mixed and any(
                    self._along(point) >= self.landing_edge_along_m
                    and abs(self._across(point)) > self.width_m / 2
                    for point in contact_points_xyz_m
                )
                reason = (
                    "mixed_lip_top_impact"
                    if mixed
                    else "side_impact"
                    if side
                    else "short_or_lip_impact"
                )
                self.fail(reason, time_s)
            else:
                self.events["landing_s"] = time_s
                self._last_top_contact_s = time_s
                self.phase = "landing"
        if self.phase == "landing":
            top_contact = any(self._top_contact(point) for point in contact_points_xyz_m)
            if obstacle_contacts:
                self.fail("obstacle_impact", time_s)
                return
            if top_contact:
                self._last_top_contact_s = time_s
            if top_contact and along >= self.landing_edge_along_m + 0.10 and tilt_deg < 45.0:
                if self._stable_since is None:
                    self._stable_since = time_s
                if time_s - self._stable_since >= self.stable_duration_s:
                    self.events["stable_landing_s"] = time_s
                    self.phase = "complete"
            else:
                self._stable_since = None
            if (
                self.phase == "landing"
                and self._last_top_contact_s is not None
                and time_s - self._last_top_contact_s > 0.25
            ):
                self.fail("lost_landing_contact", time_s)

    def timeout_reason(self) -> str:
        return {
            "approach": "did_not_reach_ramp",
            "ramp": "did_not_take_off",
            "flight": "no_landing_contact",
            "landing": "landing_not_stable",
        }.get(self.phase, self.failure_reason or "timeout")

    def report(self) -> dict[str, Any]:
        return {
            "flight_status": self.status,
            "phase": self.phase,
            "events": dict(self.events),
            "actual_takeoff_speed_mps": self.takeoff_speed_mps,
            "takeoff_lateral_m": self.takeoff_lateral_m,
            "takeoff_before_lip_m": (
                max(0.0, self.takeoff_along_m - self._contact_loss_along)
                if self.takeoff_speed_mps is not None and self._contact_loss_along is not None
                else None
            ),
            "flight_duration_s": (
                self.events["first_recontact_s"] - self.events["takeoff_s"]
                if "first_recontact_s" in self.events and "takeoff_s" in self.events
                else None
            ),
            "first_recontact": None if self.first_recontact is None else dict(self.first_recontact),
            "failure_reason": self.failure_reason,
            "stable_duration_required_s": self.stable_duration_s,
        }


__all__ = ["FlyRampProgress"]
