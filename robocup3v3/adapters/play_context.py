"""Serialize simulator state with MyAgent PlayContext/GameController semantics."""

from __future__ import annotations

import math

from ..actions import world_to_team
from ..types import MatchState, Team


def _team_theta(team, theta):
    value = theta if team is Team.BLUE else theta + math.pi
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def to_play_context_dict(state: MatchState, team, timestamp=None, match_duration_sec=600.0):
    """Return a dependency-free dictionary matching ``PlayContext`` fields.

    ``timestamp`` defaults to ``1 + elapsed`` because the deployed freshness
    filters treat zero-valued BallState/RobotState timestamps as stale.
    """

    team = Team(team)
    now = float(1.0 + state.elapsed if timestamp is None else timestamp)

    def robot_record(robot, logical_id):
        x, y = world_to_team(team, robot.pose.x, robot.pose.y)
        return {
            "player_id": logical_id,
            "pose": {"x": x, "y": y, "theta": _team_theta(team, robot.pose.theta)},
            "last_seen_at": now,
        }

    teammates = {
        robot.player_id: robot_record(robot, robot.player_id)
        for robot in state.team_robots(team)
    }
    opponents = {
        robot.player_id: robot_record(robot, robot.player_id)
        for robot in state.team_robots(team.opponent)
    }
    ball_x, ball_y = world_to_team(team, state.ball.x, state.ball.y)
    teams = []
    for side in (Team.BLUE, Team.RED):
        teams.append({
            "team_number": side.team_id,
            "goalkeeper": 3,
            "score": state.score[side],
            "players": [
                {
                    "penalty": robot.penalty.value,
                    "secs_till_unpenalised": int(math.ceil(robot.penalty_remaining)),
                    "warnings": 0,
                    "cautions": 0,
                }
                for robot in state.team_robots(side)
            ],
        })
    return {
        "game_state": {
            "version": 19,
            "packet_number": state.packet_number,
            "players_per_team": 3,
            "game_phase": state.game_phase.value,
            "state": state.game_state.value,
            "set_play": state.set_play.value,
            "stopped": state.stopped,
            "kicking_team": state.kicking_team.team_id if state.kicking_team else 255,
            "secs_remaining": max(0, int(match_duration_sec - state.elapsed)),
            "secondary_time": max(0, int(state.state_elapsed)),
            "teams": teams,
            "last_seen_at": now,
        },
        "teammates": teammates,
        "opponents": opponents,
        "ball": {"x": ball_x, "y": ball_y, "last_seen_at": now, "confidence": 1.0},
    }
