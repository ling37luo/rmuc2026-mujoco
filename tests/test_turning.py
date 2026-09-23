from __future__ import annotations

import mujoco

from rmuc2026_mujoco import (
    NoOpTurnController,
    TurnSpawn,
    run_turn_episode,
    turn_phase,
    turn_phase_registry,
)
from rmuc2026_mujoco.cli import build_parser


def test_turn_phase_command_grids_are_deterministic() -> None:
    spin = turn_phase("spin")
    arc = turn_phase("arc")
    reversal = turn_phase("reversal")
    assert len(spin.commands) == 12
    assert len(arc.commands) == 16
    assert len(reversal.commands) == 36
    assert sorted({command.yaw_rad_s for command in spin.commands}) == [
        -0.6,
        -0.4,
        -0.2,
        0.2,
        0.4,
        0.6,
    ]
    assert sorted({command.vx_mps for command in arc.commands}) == [-0.3, 0.3]
    assert tuple(reversal.commands[index].yaw_rad_s for index in range(3)) == (-0.6, 0.0, 0.6)
    assert turn_phase_registry() == turn_phase_registry()


def test_run_cli_parser_exposes_parallel_turn_contract() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "pack",
            "--robot",
            "robot.xml",
            "--controller",
            "controller:make",
            "--phase",
            "reversal",
            "--backend",
            "mujoco",
            "--workers",
            "4",
            "--envs-per-worker",
            "8",
            "--duration",
            "8",
            "--seed",
            "20260922",
        ]
    )
    assert args.phase == "reversal"
    assert args.workers == 4
    assert args.envs_per_worker == 8
    assert args.duration == 8.0
    assert args.seed == 20260922


def test_generic_freejoint_episode_has_zero_warnings() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="turn_test">
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="10 10 .1"/>
            <body name="robot" pos="0 0 1">
              <freejoint/>
              <geom type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    result = run_turn_episode(
        model,
        data,
        NoOpTurnController(),
        turn_phase("spin"),
        TurnSpawn(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=3,
        duration_s=0.02,
    )
    assert result.failed is False
    assert result.steps == 10
    assert result.warnings == {}
    assert result.nonfinite_count == 0
    assert result.min_height_m is not None
