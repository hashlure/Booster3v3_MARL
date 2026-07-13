"""Deterministic scripted opponent for curriculum and smoke tests."""

from __future__ import annotations

import math
import numpy as np

from .actions import best_shot_target_y, world_to_team
from .config import EnvConfig
from .types import MatchState, PlannerAction, PlannerIntent, Team


class RuleTreeOpponent:
    """Behavior-tree-style opponent mirroring the deployed default tactics."""

    DIFFICULTIES = ("stationary", "novice", "standard", "expert")

    def __init__(self, team: Team, config: EnvConfig, difficulty="standard"):
        self.team = team
        self.config = config
        self.set_difficulty(difficulty)

    def set_difficulty(self, difficulty):
        if difficulty not in self.DIFFICULTIES:
            raise ValueError("unknown behavior-tree difficulty: %s" % difficulty)
        self.difficulty = difficulty

    def actions(self, state: MatchState):
        if self.difficulty == "stationary":
            return {robot.name: PlannerAction() for robot in state.team_robots(self.team)}
        if state.game_state.value != "PLAYING" or state.stopped:
            return {robot.name: PlannerAction() for robot in state.team_robots(self.team)}
        # During an opponent restart, holding position avoids repeatedly
        # violating the mandatory distance and causing endless retakes.
        if state.restart_pending and state.kicking_team is self.team.opponent:
            return {robot.name: PlannerAction() for robot in state.team_robots(self.team)}
        if self.difficulty == "novice":
            return self._novice_actions(state)
        robots = state.team_robots(self.team)
        field = [robot for robot in robots if robot.player_id != 3 and robot.active]
        chaser = min(
            field,
            key=lambda r: math.hypot(r.pose.x - state.ball.x, r.pose.y - state.ball.y),
        ) if field else None
        result = {}
        bx, by = world_to_team(self.team, state.ball.x, state.ball.y)
        for robot in robots:
            if not robot.active:
                result[robot.name] = PlannerAction()
            elif robot.player_id == 3:
                danger = bx < -self.config.field_length / 2.0 + self.config.penalty_area_length
                distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
                if danger and distance <= self.config.kick_distance:
                    if not self._facing_ball(robot, state):
                        rx, ry = world_to_team(self.team, robot.pose.x, robot.pose.y)
                        result[robot.name] = PlannerAction(PlannerIntent.MOVE, rx, ry, bx, by)
                    else:
                        result[robot.name] = PlannerAction(PlannerIntent.PASS, bx, by, 0.0, math.copysign(2.8, by or 1.0))
                elif danger:
                    tx, ty = self._safe_target(bx, by)
                    result[robot.name] = PlannerAction(PlannerIntent.MOVE, tx, ty, bx, by)
                else:
                    result[robot.name] = PlannerAction(
                        PlannerIntent.GUARD, -self.config.field_length / 2.0 + 0.65,
                        max(-1.20, min(1.20, by * 0.75)), bx, by)
            elif robot is chaser:
                distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
                if distance > self.config.kick_distance:
                    tx, ty = self._safe_target(bx, by)
                    result[robot.name] = PlannerAction(PlannerIntent.MOVE, tx, ty, bx, by)
                    continue
                if not self._facing_ball(robot, state):
                    rx, ry = world_to_team(self.team, robot.pose.x, robot.pose.y)
                    result[robot.name] = PlannerAction(PlannerIntent.MOVE, rx, ry, bx, by)
                    continue
                if state.restart_pending and state.kicking_team is self.team and not state.restart_touches:
                    teammate = self._best_pass_target(state, robot, bx, by)
                    if teammate is not None:
                        tx, ty = world_to_team(self.team, teammate.pose.x, teammate.pose.y)
                        result[robot.name] = PlannerAction(PlannerIntent.PASS, bx, by, tx, ty)
                        continue
                blockers = self._lane_blockers(state, bx, by, self.config.field_length / 2.0, 0.0)
                teammate = self._best_pass_target(state, robot, bx, by)
                forward_pass = False
                if teammate is not None:
                    teammate_x, _ = world_to_team(self.team, teammate.pose.x, teammate.pose.y)
                    forward_pass = teammate_x > bx + 0.25
                # Standard AI passes only when the central shot lane is
                # blocked. Expert AI additionally avoids low-quality shots
                # when closely pressured.
                pressured = self.difficulty == "expert" and any(
                    opponent.active and math.hypot(opponent.pose.x - robot.pose.x,
                                                   opponent.pose.y - robot.pose.y) < 1.35
                    for opponent in state.team_robots(self.team.opponent)
                )
                if (blockers or pressured) and teammate is not None and forward_pass:
                    tx, ty = world_to_team(self.team, teammate.pose.x, teammate.pose.y)
                    result[robot.name] = PlannerAction(PlannerIntent.PASS, bx, by, tx, ty)
                else:
                    opponents = [
                        (*world_to_team(self.team, item.pose.x, item.pose.y), item.radius)
                        for item in state.team_robots(self.team.opponent) if item.active
                    ]
                    goal_y = best_shot_target_y(bx, by, opponents, self.config)
                    result[robot.name] = PlannerAction(PlannerIntent.SHOOT, bx, by, self.config.field_length / 2.0, goal_y)
            else:
                side = 1.0 if robot.player_id % 2 == 0 else -1.0
                # Stay wide and ahead; kick target points at the carrier so the
                # deployment motion layer can orient the receiver toward play.
                sx, sy = self._safe_target(bx + 1.1, by + side * 1.6)
                result[robot.name] = PlannerAction(PlannerIntent.MOVE, sx, sy, bx, by)
        return result

    def _novice_actions(self, state):
        """Single chaser, conservative keeper, no passing or coordinated press."""
        robots = state.team_robots(self.team)
        field = [r for r in robots if r.active and r.player_id != 3]
        chaser = min(field, key=lambda r: math.hypot(r.pose.x - state.ball.x,
                                                     r.pose.y - state.ball.y)) if field else None
        bx, by = world_to_team(self.team, state.ball.x, state.ball.y)
        result = {}
        for robot in robots:
            if not robot.active:
                result[robot.name] = PlannerAction()
            elif robot.player_id == 3:
                result[robot.name] = PlannerAction(
                    PlannerIntent.GUARD, -self.config.field_length / 2.0 + 0.65,
                    max(-0.9, min(0.9, by * 0.50)), bx, by)
            elif robot is chaser:
                distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
                if distance <= self.config.kick_distance:
                    if not self._facing_ball(robot, state):
                        rx, ry = world_to_team(self.team, robot.pose.x, robot.pose.y)
                        result[robot.name] = PlannerAction(PlannerIntent.MOVE, rx, ry, bx, by)
                    elif state.restart_pending and state.kicking_team is self.team and not state.restart_touches:
                        teammate = self._best_pass_target(state, robot, bx, by)
                        if teammate is not None:
                            tx, ty = world_to_team(self.team, teammate.pose.x, teammate.pose.y)
                            result[robot.name] = PlannerAction(PlannerIntent.PASS, bx, by, tx, ty)
                        else:
                            result[robot.name] = PlannerAction()
                    else:
                        result[robot.name] = PlannerAction(
                            PlannerIntent.SHOOT, bx, by, self.config.field_length / 2.0, 0.0)
                else:
                    tx, ty = self._safe_target(bx, by)
                    result[robot.name] = PlannerAction(PlannerIntent.MOVE, tx, ty, bx, by)
            else:
                result[robot.name] = PlannerAction()
        return result

    def _safe_target(self, x, y):
        return (
            float(np.clip(x, -self.config.field_length / 2.0 + .55,
                          self.config.field_length / 2.0 - .55)),
            float(np.clip(y, -self.config.field_width / 2.0 + .55,
                          self.config.field_width / 2.0 - .55)),
        )

    def _facing_ball(self, robot, state):
        bearing = math.atan2(state.ball.y - robot.pose.y, state.ball.x - robot.pose.x)
        error = (bearing - robot.pose.theta + math.pi) % (2.0 * math.pi) - math.pi
        return abs(error) <= self.config.kick_facing_tolerance_rad

    def teacher_action_ids(self, state: MatchState):
        """Discrete labels used for behavior-cloning the rule-tree policy."""

        if self.difficulty == "stationary":
            return {robot.name: 0 for robot in state.team_robots(self.team)}
        if self.difficulty == "novice":
            return self._novice_teacher_action_ids(state)

        robots = state.team_robots(self.team)
        field = [robot for robot in robots if robot.player_id != 3 and robot.active]
        chaser = min(field, key=lambda r: math.hypot(r.pose.x - state.ball.x, r.pose.y - state.ball.y)) if field else None
        bx, by = world_to_team(self.team, state.ball.x, state.ball.y)
        labels = {}
        for robot in robots:
            if (not robot.active or state.game_state.value != "PLAYING" or state.stopped
                    or (state.restart_pending and state.kicking_team is self.team.opponent)):
                labels[robot.name] = 0
                continue
            rx, ry = world_to_team(self.team, robot.pose.x, robot.pose.y)
            if robot.player_id == 3:
                danger = bx < -self.config.field_length / 2.0 + self.config.penalty_area_length
                distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
                if danger and distance <= self.config.kick_distance:
                    target = min(field, default=None, key=lambda r: math.hypot(r.pose.x - robot.pose.x, r.pose.y - robot.pose.y))
                    labels[robot.name] = 12 + target.player_id if target is not None else 16
                elif danger:
                    labels[robot.name] = self._direction_action(rx, ry, bx, by, precision=True)
                else:
                    labels[robot.name] = 19
            elif robot is chaser:
                distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
                if distance > self.config.kick_distance:
                    labels[robot.name] = self._direction_action(rx, ry, bx, by)
                else:
                    if state.restart_pending and state.kicking_team is self.team and not state.restart_touches:
                        teammate = self._best_pass_target(state, robot, bx, by)
                        if teammate is not None:
                            labels[robot.name] = 12 + teammate.player_id
                            continue
                    blockers = self._lane_blockers(state, bx, by, self.config.field_length / 2.0, 0.0)
                    teammate = self._best_pass_target(state, robot, bx, by)
                    teammate_x = (world_to_team(self.team, teammate.pose.x, teammate.pose.y)[0]
                                  if teammate is not None else -1e9)
                    labels[robot.name] = (12 + teammate.player_id
                                          if blockers and teammate is not None and teammate_x > bx + .25
                                          else 22)
            else:
                labels[robot.name] = 20 if robot.player_id % 2 == 0 else 21
        return labels

    def _novice_teacher_action_ids(self, state):
        robots = state.team_robots(self.team)
        field = [r for r in robots if r.active and r.player_id != 3]
        chaser = min(field, key=lambda r: math.hypot(r.pose.x - state.ball.x,
                                                     r.pose.y - state.ball.y)) if field else None
        bx, by = world_to_team(self.team, state.ball.x, state.ball.y)
        labels = {}
        for robot in robots:
            if (not robot.active or state.game_state.value != "PLAYING" or state.stopped
                    or (state.restart_pending and state.kicking_team is self.team.opponent)):
                labels[robot.name] = 0
            elif robot.player_id == 3:
                labels[robot.name] = 19
            elif robot is not chaser:
                labels[robot.name] = 0
            else:
                rx, ry = world_to_team(self.team, robot.pose.x, robot.pose.y)
                distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
                if distance > self.config.kick_distance:
                    labels[robot.name] = self._direction_action(rx, ry, bx, by)
                elif state.restart_pending and state.kicking_team is self.team and not state.restart_touches:
                    teammate = self._best_pass_target(state, robot, bx, by)
                    labels[robot.name] = 12 + teammate.player_id if teammate is not None else 0
                else:
                    labels[robot.name] = 16
        return labels

    @staticmethod
    def _direction_action(x1, y1, x2, y2, precision=False):
        angle = math.atan2(y2 - y1, x2 - x1)
        sector = int(round(angle / (math.pi / 4.0))) % 8
        # Catalogue order: E, NE, N, NW, W, SW, S, SE.
        return (23 if precision else 1) + sector

    def _lane_blockers(self, state, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        length2 = max(dx * dx + dy * dy, 1e-8)
        blockers = []
        for opponent in state.team_robots(self.team.opponent):
            ox, oy = world_to_team(self.team, opponent.pose.x, opponent.pose.y)
            t = max(0.0, min(1.0, ((ox - x1) * dx + (oy - y1) * dy) / length2))
            if 0.08 < t < 0.95 and math.hypot(ox - (x1 + t * dx), oy - (y1 + t * dy)) < 0.75:
                blockers.append(opponent)
        return blockers

    def _best_pass_target(self, state, carrier, bx, by):
        candidates = [r for r in state.team_robots(self.team) if r is not carrier and r.player_id != 3 and r.active]
        clear = []
        for candidate in candidates:
            tx, ty = world_to_team(self.team, candidate.pose.x, candidate.pose.y)
            if not self._lane_blockers(state, bx, by, tx, ty):
                clear.append((tx + 0.15 * abs(ty - by), candidate))
        return max(clear, default=(0.0, None), key=lambda item: item[0])[1]



# Backward-compatible name for older configs/imports.
HeuristicOpponent = RuleTreeOpponent
