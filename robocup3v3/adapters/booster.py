"""Deployment-side duck-typed adapters for MyAgent PlayContext.

The functions accept either dataclasses from ``soccer_framework`` or dictionaries
from :func:`to_play_context_dict`, allowing byte-for-byte observation parity to
be tested without importing ROS or Booster SDK packages.
"""

from __future__ import annotations

import math

import numpy as np

from ..actions import N_ACTIONS, decode_team_view_action
from ..config import EnvConfig
from ..types import GameState, PlannerIntent, SetPlay


def _get(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value, default):
    return getattr(value, "value", value) if value is not None else default


def _pose(record):
    pose = _get(record, "pose") if record is not None else None
    if pose is None:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(_get(pose, "x", 0.0)),
        float(_get(pose, "y", 0.0)),
        float(_get(pose, "theta", 0.0)),
        1.0,
    )


def _mapping_get(mapping, key):
    if mapping is None:
        return None
    if isinstance(mapping, dict):
        return mapping.get(key, mapping.get(str(key)))
    return None


def _team(game, team_id):
    for item in _get(game, "teams", []) or []:
        if int(_get(item, "team_number", -1)) == int(team_id):
            return item
    return None


def _player(team, player_id):
    players = _get(team, "players", []) or []
    return players[player_id - 1] if 0 < player_id <= len(players) else None


def observation_from_play_context(
    context,
    player_id,
    team_id,
    config=None,
    previous_intent=PlannerIntent.HOLD,
):
    """Create the exact 59-float Actor observation from a PlayContext snapshot."""

    config = config or EnvConfig()
    game = _get(context, "game_state")
    teammates = _get(context, "teammates", {}) or {}
    opponents = _get(context, "opponents", {}) or {}
    self_record = _mapping_get(teammates, player_id)
    sx, sy, stheta, self_valid = _pose(self_record)
    own_team = _team(game, team_id)
    own_player = _player(own_team, player_id)
    penalty = _enum_value(_get(own_player, "penalty", "SUBSTITUTE"), "SUBSTITUTE")
    penalty_remaining = float(_get(own_player, "secs_till_unpenalised", 0.0))
    active = self_valid and penalty == "NONE" and penalty_remaining <= 0.0

    half_l, half_w = config.field_length / 2.0, config.field_width / 2.0
    ball = _get(context, "ball")
    if ball is None:
        bx = by = ball_valid = 0.0
    else:
        bx = float(_get(ball, "x", 0.0))
        by = float(_get(ball, "y", 0.0))
        ball_valid = 1.0
    values = [
        sx / half_l, sy / half_w, math.cos(stheta), math.sin(stheta),
        1.0 if active else 0.0,
        min(1.0, penalty_remaining / max(config.penalty_duration_sec, 1e-6)),
        (bx - sx) / config.field_length if ball_valid and self_valid else 0.0,
        (by - sy) / config.field_width if ball_valid and self_valid else 0.0,
        bx / half_l if ball_valid else 0.0,
        by / half_w if ball_valid else 0.0,
        ball_valid,
    ]
    for other_id in (1, 2, 3):
        if other_id == player_id:
            continue
        ox, oy, otheta, valid = _pose(_mapping_get(teammates, other_id))
        other_player = _player(own_team, other_id)
        other_penalty = _enum_value(_get(other_player, "penalty", "SUBSTITUTE"), "SUBSTITUTE")
        other_active = valid and other_penalty == "NONE" and float(_get(other_player, "secs_till_unpenalised", 0)) <= 0
        values.extend([
            (ox - sx) / config.field_length if valid and self_valid else 0.0,
            (oy - sy) / config.field_width if valid and self_valid else 0.0,
            math.cos(otheta) if valid else 0.0,
            math.sin(otheta) if valid else 0.0,
            1.0 if other_active else 0.0,
        ])
    opponent_team = next((item for item in (_get(game, "teams", []) or []) if int(_get(item, "team_number", -1)) != int(team_id)), None)
    for other_id in (1, 2, 3):
        ox, oy, otheta, valid = _pose(_mapping_get(opponents, other_id))
        other_player = _player(opponent_team, other_id)
        other_penalty = _enum_value(_get(other_player, "penalty", "SUBSTITUTE"), "SUBSTITUTE")
        other_active = valid and other_penalty == "NONE" and float(_get(other_player, "secs_till_unpenalised", 0)) <= 0
        values.extend([
            (ox - sx) / config.field_length if valid and self_valid else 0.0,
            (oy - sy) / config.field_width if valid and self_valid else 0.0,
            math.cos(otheta) if valid else 0.0,
            math.sin(otheta) if valid else 0.0,
            1.0 if other_active else 0.0,
        ])

    game_state = _enum_value(_get(game, "state", "INITIAL"), "INITIAL")
    set_play = _enum_value(_get(game, "set_play", "NONE"), "NONE")
    values.extend([1.0 if game_state == value.value else 0.0 for value in GameState])
    values.extend([1.0 if set_play == value.value else 0.0 for value in SetPlay])
    kicking_team = int(_get(game, "kicking_team", 255))
    opponent_team_id = int(_get(opponent_team, "team_number", 255)) if opponent_team is not None else 255
    own_score = float(_get(own_team, "score", 0.0))
    opponent_score = float(_get(opponent_team, "score", 0.0))
    values.extend([
        1.0 if kicking_team == int(team_id) else 0.0,
        1.0 if kicking_team == opponent_team_id else 0.0,
        1.0 if kicking_team == 255 else 0.0,
        float(np.clip(float(_get(game, "secs_remaining", config.match_duration_sec)) / max(config.match_duration_sec, 1e-6), 0.0, 1.0)),
        float(np.clip((own_score - opponent_score) / 5.0, -1.0, 1.0)),
    ])
    previous_intent = PlannerIntent(int(previous_intent))
    values.extend([1.0 if previous_intent is intent else 0.0 for intent in PlannerIntent])
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (59,):
        raise RuntimeError("observation contract error: expected (59,), got %r" % (result.shape,))
    return result


def action_from_play_context(action_id, context, player_id, config=None):
    """Decode a policy action into a team-view PlannerAction for the BT layer."""

    config = config or EnvConfig()
    teammates = _get(context, "teammates", {}) or {}
    self_x, self_y, _, self_valid = _pose(_mapping_get(teammates, player_id))
    if not self_valid:
        action_id = 0
    ball = _get(context, "ball")
    ball_x = float(_get(ball, "x", 0.0)) if ball is not None else 0.0
    ball_y = float(_get(ball, "y", 0.0)) if ball is not None else 0.0
    positions = {}
    opponent_positions = []
    for other_id in (1, 2, 3):
        x, y, _, valid = _pose(_mapping_get(teammates, other_id))
        if valid:
            positions[other_id] = (x, y)
    opponents = _get(context, "opponents", {}) or {}
    for other_id in (1, 2, 3):
        x, y, _, valid = _pose(_mapping_get(opponents, other_id))
        if valid:
            opponent_positions.append((x, y, config.robot_radius))
    return decode_team_view_action(
        action_id, self_x, self_y, ball_x, ball_y, positions, config,
        opponent_positions,
    )


def action_mask_from_play_context(context, player_id, team_id, config=None):
    """Deployment-equivalent Discrete mask without simulator-only state."""

    config = config or EnvConfig()
    mask = np.ones(N_ACTIONS, dtype=np.float32)
    game = _get(context, "game_state")
    state = _enum_value(_get(game, "state", "INITIAL"), "INITIAL")
    stopped = bool(_get(game, "stopped", False))
    team = _team(game, team_id)
    player = _player(team, player_id)
    penalty = _enum_value(_get(player, "penalty", "SUBSTITUTE"), "SUBSTITUTE")
    active = penalty == "NONE" and float(_get(player, "secs_till_unpenalised", 0)) <= 0
    if not active or state not in ("READY", "PLAYING") or stopped:
        mask[:] = 0.0
        mask[0] = 1.0
        return mask
    teammates = _get(context, "teammates", {}) or {}
    sx, sy, _, valid = _pose(_mapping_get(teammates, player_id))
    ball = _get(context, "ball")
    distance = float("inf") if not valid or ball is None else math.hypot(sx - float(_get(ball, "x", 0.0)), sy - float(_get(ball, "y", 0.0)))
    bx = float(_get(ball, "x", 0.0)) if ball is not None else 0.0
    by = float(_get(ball, "y", 0.0)) if ball is not None else 0.0
    ball_bearing = math.atan2(by - sy, bx - sx) if valid and ball is not None else 0.0
    facing_error = (ball_bearing - _pose(_mapping_get(teammates, player_id))[2] + math.pi) % (2.0 * math.pi) - math.pi
    facing_ball = valid and ball is not None and abs(facing_error) <= config.kick_facing_tolerance_rad
    candidates = []
    for other_id in (1, 2, 3):
        other_player = _player(team, other_id)
        other_penalty = _enum_value(_get(other_player, "penalty", "SUBSTITUTE"), "SUBSTITUTE")
        other_active = other_penalty == "NONE" and float(_get(other_player, "secs_till_unpenalised", 0)) <= 0
        ox, oy, _, other_valid = _pose(_mapping_get(teammates, other_id))
        if other_active and other_valid and ball is not None:
            candidates.append((math.hypot(ox - bx, oy - by), other_id))
        if not other_active:
            mask[12 + other_id] = 0.0
    nearest_id = min(candidates, default=(float("inf"), -1))[1]
    if state != "PLAYING" or distance > config.kick_distance or nearest_id != player_id or not facing_ball:
        mask[9:19] = 0.0
        mask[22] = 0.0
    mask[12 + player_id] = 0.0
    kicking_team = int(_get(game, "kicking_team", 255))
    set_play = _enum_value(_get(game, "set_play", "NONE"), "NONE")
    if set_play != "NONE" and kicking_team != int(team_id):
        mask[9:19] = 0.0
        mask[22] = 0.0
    return mask
