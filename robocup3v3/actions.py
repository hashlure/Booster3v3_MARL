"""Portable discrete planning catalogue used by MAPPO and deployment.

The supplied on-policy implementation handles ``Discrete`` actions and action
masks reliably.  Action IDs are therefore decoded into the same field-coordinate
``PlannerAction`` contract that a deployed behavior-tree node can consume.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .config import EnvConfig
from .types import MatchState, PlannerAction, PlannerIntent, RobotSimState, Team


ACTION_NAMES = (
    "hold",
    "move_forward",
    "move_forward_left",
    "move_left",
    "move_back_left",
    "move_back",
    "move_back_right",
    "move_right",
    "move_forward_right",
    "dribble_goal",
    "dribble_left",
    "dribble_right",
    "dribble_center",
    "pass_teammate_1",
    "pass_teammate_2",
    "pass_teammate_3",
    "shoot_center",
    "shoot_left",
    "shoot_right",
    "guard_goal",
    "support_left",
    "support_right",
    "shoot_best_gap",
    "precision_forward",
    "precision_forward_left",
    "precision_left",
    "precision_back_left",
    "precision_back",
    "precision_back_right",
    "precision_right",
    "precision_forward_right",
)
N_ACTIONS = len(ACTION_NAMES)


def best_shot_target_y(ball_x, ball_y, opponents, config):
    """Return the exact goal-line y at the centre of the widest unblocked gap.

    Inputs are in team view. ``opponents`` contains ``(x, y)`` or
    ``(x, y, radius)`` records. Obstacles are expanded by the ball radius and a
    small execution margin before their tangent-angle shadows are subtracted
    from the usable goal mouth.
    """

    goal_x = config.field_length / 2.0
    forward = goal_x - ball_x
    safe_half = config.goal_width / 2.0 - config.ball_radius - 0.04
    if forward <= 1e-6:
        return 0.0
    low = math.atan2(-safe_half - ball_y, forward)
    high = math.atan2(safe_half - ball_y, forward)
    blocked = []
    for record in opponents:
        ox, oy = float(record[0]), float(record[1])
        radius = float(record[2]) if len(record) > 2 else config.robot_radius
        dx, dy = ox - ball_x, oy - ball_y
        distance = math.hypot(dx, dy)
        if dx <= 0.0 or ox >= goal_x or distance <= 1e-6:
            continue
        expanded = radius + config.ball_radius + 0.08
        half_angle = math.asin(min(0.999, expanded / distance))
        centre = math.atan2(dy, dx)
        start, end = max(low, centre - half_angle), min(high, centre + half_angle)
        if start < end:
            blocked.append((start, end))
    blocked.sort()
    merged = []
    for start, end in blocked:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    gaps = []
    cursor = low
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < high:
        gaps.append((cursor, high))
    if not gaps:
        # Fully occluded: aim away from the closest side of the largest shadow.
        return float(np.clip(-ball_y * 0.25, -safe_half, safe_half))
    start, end = max(gaps, key=lambda gap: gap[1] - gap[0])
    angle = 0.5 * (start + end)
    return float(np.clip(ball_y + forward * math.tan(angle), -safe_half, safe_half))


def team_to_world(team: Team, x: float, y: float):
    sign = team.attack_sign
    return sign * x, sign * y


def world_to_team(team: Team, x: float, y: float):
    sign = team.attack_sign
    return sign * x, sign * y


def _clip_target(config: EnvConfig, x: float, y: float):
    return (
        float(np.clip(x, -config.field_length / 2.0 + 0.35, config.field_length / 2.0 - 0.35)),
        float(np.clip(y, -config.field_width / 2.0 + 0.35, config.field_width / 2.0 - 0.35)),
    )


def decode_action(
    action_id: int,
    robot: RobotSimState,
    state: MatchState,
    config: EnvConfig,
) -> PlannerAction:
    """Decode one portable action ID using only deployable match state."""

    tx, ty = world_to_team(robot.team, robot.pose.x, robot.pose.y)
    ball_x, ball_y = world_to_team(robot.team, state.ball.x, state.ball.y)
    teammates = {
        teammate.player_id: world_to_team(robot.team, teammate.pose.x, teammate.pose.y)
        for teammate in state.team_robots(robot.team)
    }
    opponents = [
        (*world_to_team(robot.team, opponent.pose.x, opponent.pose.y), opponent.radius)
        for opponent in state.team_robots(robot.team.opponent)
        if opponent.active
    ]
    return decode_team_view_action(
        action_id,
        tx,
        ty,
        ball_x,
        ball_y,
        teammates,
        config,
        opponents,
    )


def decode_team_view_action(
    action_id: int,
    self_x: float,
    self_y: float,
    ball_x: float,
    ball_y: float,
    teammates: Dict[int, Tuple[float, float]],
    config: EnvConfig,
    opponents=(),
) -> PlannerAction:
    """Shared decoder for simulator state and deployed PlayContext state."""

    action_id = int(np.asarray(action_id).reshape(-1)[0])
    if action_id < 0 or action_id >= N_ACTIONS:
        action_id = 0

    if action_id == 0:
        return PlannerAction(PlannerIntent.HOLD, self_x, self_y, ball_x, ball_y)

    if 1 <= action_id <= 8:
        directions = (
            (1.0, 0.0), (0.707, 0.707), (0.0, 1.0), (-0.707, 0.707),
            (-1.0, 0.0), (-0.707, -0.707), (0.0, -1.0), (0.707, -0.707),
        )
        dx, dy = directions[action_id - 1]
        target = _clip_target(config, self_x + 1.2 * dx, self_y + 1.2 * dy)
        return PlannerAction(PlannerIntent.MOVE, target[0], target[1], ball_x, ball_y)

    if 9 <= action_id <= 12:
        if action_id == 9:
            kx, ky = config.field_length / 2.0, 0.0
        elif action_id == 10:
            kx, ky = ball_x + 1.0, ball_y + 0.8
        elif action_id == 11:
            kx, ky = ball_x + 1.0, ball_y - 0.8
        else:
            kx, ky = ball_x + 1.0, ball_y * 0.5
        kx, ky = _clip_target(config, kx, ky)
        return PlannerAction(PlannerIntent.DRIBBLE, ball_x, ball_y, kx, ky)

    if 13 <= action_id <= 15:
        target_id = action_id - 12
        kx, ky = teammates.get(target_id, (ball_x, ball_y))
        return PlannerAction(PlannerIntent.PASS, ball_x, ball_y, kx, ky)

    if 16 <= action_id <= 18:
        goal_y = (0.0, 0.85, -0.85)[action_id - 16]
        return PlannerAction(
            PlannerIntent.SHOOT,
            ball_x,
            ball_y,
            config.field_length / 2.0,
            goal_y,
        )

    if action_id == 19:
        return PlannerAction(PlannerIntent.GUARD, -config.field_length / 2.0 + 0.65, 0.0, ball_x, ball_y)

    if action_id in (20, 21):
        side = 1.0 if action_id == 20 else -1.0
        sx, sy = _clip_target(config, ball_x + 1.25, ball_y + side * 1.65)
        return PlannerAction(PlannerIntent.MOVE, sx, sy, ball_x, ball_y)

    if action_id == 22:
        goal_y = best_shot_target_y(ball_x, ball_y, opponents, config)
        return PlannerAction(
            PlannerIntent.SHOOT, ball_x, ball_y,
            config.field_length / 2.0, goal_y,
        )

    directions = (
        (1.0, 0.0), (0.707, 0.707), (0.0, 1.0), (-0.707, 0.707),
        (-1.0, 0.0), (-0.707, -0.707), (0.0, -1.0), (0.707, -0.707),
    )
    dx, dy = directions[action_id - 23]
    target = _clip_target(config, self_x + 0.45 * dx, self_y + 0.45 * dy)
    return PlannerAction(PlannerIntent.MOVE, target[0], target[1], ball_x, ball_y)


def available_actions(robot: RobotSimState, state: MatchState, config: EnvConfig) -> np.ndarray:
    """Return a Discrete action mask using only legal/deployable information."""

    mask = np.ones(N_ACTIONS, dtype=np.float32)
    if not robot.active or state.game_state.value not in ("READY", "PLAYING") or state.stopped:
        mask[:] = 0.0
        mask[0] = 1.0
        return mask

    distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
    ball_bearing = math.atan2(state.ball.y - robot.pose.y, state.ball.x - robot.pose.x)
    facing_error = (ball_bearing - robot.pose.theta + math.pi) % (2.0 * math.pi) - math.pi
    facing_ball = abs(facing_error) <= config.kick_facing_tolerance_rad
    # TiZero masks possession-dependent actions aggressively.  In our
    # simultaneous 3-agent interface this also prevents two nearby teammates
    # from issuing conflicting kicks in the same simulator tick: only the
    # nearest active teammate is treated as the current ball controller.
    active_team = [r for r in state.team_robots(robot.team) if r.active]
    nearest = min(
        active_team,
        key=lambda r: (math.hypot(r.pose.x - state.ball.x, r.pose.y - state.ball.y), r.player_id),
        default=None,
    )
    can_kick = (
        state.game_state.value == "PLAYING"
        and distance <= config.kick_distance
        and nearest is robot
        and facing_ball
    )
    if not can_kick:
        mask[9:19] = 0.0
        mask[22] = 0.0
    # Passing to oneself is always invalid.
    mask[12 + robot.player_id] = 0.0
    # Do not let the policy waste probability mass on penalized/sent-off
    # receivers.  The mapping remains stable: action 13/14/15 -> player 1/2/3.
    for teammate in state.team_robots(robot.team):
        if not teammate.active:
            mask[12 + teammate.player_id] = 0.0
    if state.set_play.value != "NONE" and state.kicking_team is not robot.team:
        mask[9:19] = 0.0
        mask[22] = 0.0
    return mask
