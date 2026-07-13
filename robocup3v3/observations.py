"""Actor-local and centralized-critic observations.

Actor features use only fields available from deployed ``PlayContext``.  True
velocities are deliberately restricted to the centralized critic.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .actions import world_to_team
from .config import EnvConfig
from .types import GameState, MatchState, PlannerIntent, RobotSimState, SetPlay, Team


GAME_STATES = tuple(GameState)
SET_PLAYS = tuple(SetPlay)


def _one_hot(value, values):
    return [1.0 if value is candidate else 0.0 for candidate in values]


def _team_pose(team: Team, robot: RobotSimState):
    x, y = world_to_team(team, robot.pose.x, robot.pose.y)
    theta = robot.pose.theta if team is Team.BLUE else robot.pose.theta + math.pi
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return x, y, theta


def local_observation(state: MatchState, robot: RobotSimState, config: EnvConfig):
    """Fixed local vector reproducible from one team's PlayContext snapshot."""

    half_l, half_w = config.field_length / 2.0, config.field_width / 2.0
    sx, sy, stheta = _team_pose(robot.team, robot)
    bx, by = world_to_team(robot.team, state.ball.x, state.ball.y)
    values = [
        sx / half_l, sy / half_w, math.cos(stheta), math.sin(stheta),
        1.0 if robot.active else 0.0,
        min(1.0, robot.penalty_remaining / max(config.penalty_duration_sec, 1e-6)),
        (bx - sx) / config.field_length, (by - sy) / config.field_width,
        bx / half_l, by / half_w, 1.0,
    ]

    teammates = [r for r in state.team_robots(robot.team) if r.player_id != robot.player_id]
    for other in teammates:
        ox, oy, otheta = _team_pose(robot.team, other)
        values.extend([
            (ox - sx) / config.field_length,
            (oy - sy) / config.field_width,
            math.cos(otheta), math.sin(otheta),
            1.0 if other.active else 0.0,
        ])
    for other in state.team_robots(robot.team.opponent):
        ox, oy, otheta = _team_pose(robot.team, other)
        values.extend([
            (ox - sx) / config.field_length,
            (oy - sy) / config.field_width,
            math.cos(otheta), math.sin(otheta),
            1.0 if other.active else 0.0,
        ])

    values.extend(_one_hot(state.game_state, GAME_STATES))
    values.extend(_one_hot(state.set_play, SET_PLAYS))
    values.extend([
        1.0 if state.kicking_team is robot.team else 0.0,
        1.0 if state.kicking_team is robot.team.opponent else 0.0,
        1.0 if state.kicking_team is None else 0.0,
        max(0.0, 1.0 - state.elapsed / max(config.match_duration_sec, 1e-6)),
        float(np.clip((state.score[robot.team] - state.score[robot.team.opponent]) / 5.0, -1.0, 1.0)),
    ])
    values.extend(_one_hot(robot.last_action.intent, tuple(PlannerIntent)))
    return np.asarray(values, dtype=np.float32)


def global_state(state: MatchState, perspective: Team, config: EnvConfig):
    """Centralized critic state; may include simulator-only velocities."""

    half_l, half_w = config.field_length / 2.0, config.field_width / 2.0
    values = []
    for team in (perspective, perspective.opponent):
        for robot in state.team_robots(team):
            x, y, theta = _team_pose(perspective, robot)
            vx, vy = world_to_team(perspective, robot.vx, robot.vy)
            values.extend([
                x / half_l, y / half_w, math.cos(theta), math.sin(theta),
                vx / config.max_robot_speed, vy / config.max_robot_speed,
                1.0 if robot.active else 0.0,
            ])
    bx, by = world_to_team(perspective, state.ball.x, state.ball.y)
    bvx, bvy = world_to_team(perspective, state.ball.vx, state.ball.vy)
    values.extend([
        bx / half_l, by / half_w,
        bvx / max(config.shot_speed, 1e-6), bvy / max(config.shot_speed, 1e-6),
    ])
    values.extend(_one_hot(state.game_state, GAME_STATES))
    values.extend(_one_hot(state.set_play, SET_PLAYS))
    values.extend([
        1.0 if state.kicking_team is perspective else 0.0,
        1.0 if state.kicking_team is perspective.opponent else 0.0,
        1.0 if state.kicking_team is None else 0.0,
        state.score[perspective] / 10.0,
        state.score[perspective.opponent] / 10.0,
        max(0.0, 1.0 - state.elapsed / max(config.match_duration_sec, 1e-6)),
    ])
    return np.asarray(values, dtype=np.float32)


def observation_size(config=None):
    # 6 self + 5 ball + 2*5 teammates + 3*5 opponents + 5 game + 7 setplay
    # + 3 kicking-team + 2 clock/score + 6 previous intent
    return 59


def global_state_size(config=None):
    # 6 robots * 7 + ball 4 + game 5 + setplay 7 + match 6
    return 64


BIRDVIEW_CHANNELS = 10


def birdview_state(state: MatchState, perspective: Team, config: EnvConfig, height=32, width=48):
    """Compact team-oriented raster for the training-only centralized critic.

    Channels are own/opponent occupancy, ball occupancy, own/opponent x/y
    velocity fields, ball x/y velocity fields, and an active-player mask.
    Entities are splatted over a 3x3 neighbourhood to remain visible after
    convolutional downsampling.
    """
    grid = np.zeros((BIRDVIEW_CHANNELS, int(height), int(width)), dtype=np.float32)
    half_l, half_w = config.field_length / 2.0, config.field_width / 2.0

    def pixel(x, y):
        col = int(np.clip(round((x / half_l + 1.0) * 0.5 * (width - 1)), 0, width - 1))
        row = int(np.clip(round((1.0 - (y / half_w + 1.0) * 0.5) * (height - 1)), 0, height - 1))
        return row, col

    def splat(channel, row, col, value=1.0):
        for dr, dc, weight in ((0, 0, 1.0), (-1, 0, .5), (1, 0, .5),
                               (0, -1, .5), (0, 1, .5), (-1, -1, .25),
                               (-1, 1, .25), (1, -1, .25), (1, 1, .25)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width:
                grid[channel, rr, cc] += float(value) * weight

    for team, occ, vx_ch, vy_ch in ((perspective, 0, 3, 4),
                                     (perspective.opponent, 1, 5, 6)):
        for robot in state.team_robots(team):
            x, y, _ = _team_pose(perspective, robot)
            vx, vy = world_to_team(perspective, robot.vx, robot.vy)
            row, col = pixel(x, y)
            splat(occ, row, col, 1.0 if robot.active else 0.25)
            splat(vx_ch, row, col, np.clip(vx / config.max_robot_speed, -1.0, 1.0))
            splat(vy_ch, row, col, np.clip(vy / config.max_robot_speed, -1.0, 1.0))
            if robot.active:
                splat(9, row, col, 1.0 if team is perspective else -1.0)

    bx, by = world_to_team(perspective, state.ball.x, state.ball.y)
    bvx, bvy = world_to_team(perspective, state.ball.vx, state.ball.vy)
    row, col = pixel(bx, by)
    splat(2, row, col)
    splat(7, row, col, np.clip(bvx / max(config.shot_speed, 1e-6), -1.0, 1.0))
    splat(8, row, col, np.clip(bvy / max(config.shot_speed, 1e-6), -1.0, 1.0))
    return grid


def hybrid_global_state(state, perspective, config, height=32, width=48):
    vector = global_state(state, perspective, config)
    raster = birdview_state(state, perspective, config, height, width).reshape(-1)
    return np.concatenate((raster, vector)).astype(np.float32)


def hybrid_global_state_size(height=32, width=48):
    return BIRDVIEW_CHANNELS * int(height) * int(width) + global_state_size()
