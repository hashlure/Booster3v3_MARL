"""Deterministic lightweight 2D dynamics for upper-level planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from .actions import team_to_world
from .config import EnvConfig
from .types import MatchState, PlannerAction, PlannerIntent, RobotSimState, RuleEvent


def normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PhysicsResult:
    events: List[RuleEvent] = field(default_factory=list)
    illegal_actions: List[str] = field(default_factory=list)


class PhysicsEngine:
    def __init__(self, config: EnvConfig):
        self.config = config

    def step(
        self,
        state: MatchState,
        actions: Dict[str, PlannerAction],
        rng: np.random.RandomState,
        can_touch: Callable[[RobotSimState], bool],
    ) -> PhysicsResult:
        result = PhysicsResult()
        if state.game_state.value not in ("READY", "PLAYING") or state.stopped:
            self._stop_robots(state)
            return result

        for robot in state.robots.values():
            action = actions.get(robot.name, PlannerAction())
            robot.last_action = action
            if not robot.active:
                robot.vx = robot.vy = robot.vyaw = 0.0
                continue
            self._move_robot(robot, action, rng)

        self._resolve_robot_collisions(state)

        if state.game_state.value == "PLAYING":
            self._process_ball_contacts(state, actions, can_touch, result)
            self._integrate_ball(state)
        else:
            state.ball.vx = state.ball.vy = 0.0
        return result

    def _move_robot(self, robot: RobotSimState, action: PlannerAction, rng: np.random.RandomState):
        if action.intent is PlannerIntent.HOLD:
            desired_vx = desired_vy = 0.0
            desired_theta = robot.pose.theta
        else:
            wx, wy = team_to_world(robot.team, action.target_x, action.target_y)
            dx, dy = wx - robot.pose.x, wy - robot.pose.y
            distance = math.hypot(dx, dy)
            motion_theta = math.atan2(dy, dx) if distance > 1e-8 else robot.pose.theta
            # Upper-level MOVE/GUARD commands carry a look-at point in the
            # kick target. This lets a receiver run to support space while
            # gradually turning its chest toward the ball carrier.
            if action.intent in (PlannerIntent.MOVE, PlannerIntent.GUARD):
                look_x, look_y = team_to_world(
                    robot.team, action.kick_target_x, action.kick_target_y)
                look_dx, look_dy = look_x - robot.pose.x, look_y - robot.pose.y
                desired_theta = (math.atan2(look_dy, look_dx)
                                 if math.hypot(look_dx, look_dy) > 1e-8 else motion_theta)
            else:
                desired_theta = motion_theta
            speed = min(self.config.max_robot_speed, distance / max(self.config.dt, 1e-6))
            desired_vx = speed * math.cos(motion_theta)
            desired_vy = speed * math.sin(motion_theta)

        max_delta = self.config.robot_acceleration * self.config.dt
        robot.vx += float(np.clip(desired_vx - robot.vx, -max_delta, max_delta))
        robot.vy += float(np.clip(desired_vy - robot.vy, -max_delta, max_delta))
        angle_error = normalize_angle(desired_theta - robot.pose.theta)
        desired_vyaw = float(np.clip(
            angle_error / self.config.dt,
            -self.config.max_angular_speed,
            self.config.max_angular_speed,
        ))
        max_yaw_delta = self.config.angular_acceleration * self.config.dt
        robot.vyaw += float(np.clip(desired_vyaw - robot.vyaw, -max_yaw_delta, max_yaw_delta))
        robot.vyaw = float(np.clip(robot.vyaw, -self.config.max_angular_speed,
                                  self.config.max_angular_speed))
        if self.config.action_noise > 0.0:
            robot.vx += float(rng.normal(0.0, self.config.action_noise))
            robot.vy += float(rng.normal(0.0, self.config.action_noise))
        robot.pose.x += robot.vx * self.config.dt
        robot.pose.y += robot.vy * self.config.dt
        robot.pose.theta = normalize_angle(robot.pose.theta + robot.vyaw * self.config.dt)

    def _process_ball_contacts(self, state, actions, can_touch, result):
        robots = sorted(
            state.robots.values(),
            key=lambda r: math.hypot(r.pose.x - state.ball.x, r.pose.y - state.ball.y),
        )
        for robot in robots:
            distance = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
            if not robot.active or distance > self.config.kick_distance:
                continue
            action = actions.get(robot.name, PlannerAction())
            is_kick = action.intent in (PlannerIntent.DRIBBLE, PlannerIntent.PASS, PlannerIntent.SHOOT)
            if is_kick and not can_touch(robot):
                result.illegal_actions.append(robot.name)
                continue
            if is_kick:
                ball_bearing = math.atan2(state.ball.y - robot.pose.y,
                                          state.ball.x - robot.pose.x)
                facing_error = normalize_angle(ball_bearing - robot.pose.theta)
                if abs(facing_error) > self.config.kick_facing_tolerance_rad:
                    result.illegal_actions.append(robot.name)
                    continue
            ball_speed = math.hypot(state.ball.vx, state.ball.vy)
            if (
                is_kick
                and ball_speed > self.config.max_control_ball_speed
                and state.ball.last_touch_team is not robot.team
            ):
                # A high-speed opponent shot cannot be instantaneously trapped
                # and kicked back merely because it crosses the control radius.
                result.illegal_actions.append(robot.name)
                continue
            if is_kick:
                kx, ky = team_to_world(robot.team, action.kick_target_x, action.kick_target_y)
                dx, dy = kx - state.ball.x, ky - state.ball.y
                length = math.hypot(dx, dy)
                if length <= 1e-8:
                    continue
                speed = {
                    PlannerIntent.DRIBBLE: self.config.dribble_speed,
                    PlannerIntent.PASS: self.config.pass_speed,
                    PlannerIntent.SHOOT: self.config.shot_speed,
                }[action.intent]
                state.ball.vx = speed * dx / length
                state.ball.vy = speed * dy / length
                self._record_touch(state, robot, result, action.intent.name.lower())
                return

            # Passive contact prevents robots walking through a stationary ball.
            if distance < robot.radius + state.ball.radius:
                dx, dy = state.ball.x - robot.pose.x, state.ball.y - robot.pose.y
                length = max(math.hypot(dx, dy), 1e-6)
                state.ball.x = robot.pose.x + dx / length * (robot.radius + state.ball.radius)
                state.ball.y = robot.pose.y + dy / length * (robot.radius + state.ball.radius)
                state.ball.vx += robot.vx * 0.55
                state.ball.vy += robot.vy * 0.55
                if can_touch(robot):
                    self._record_touch(state, robot, result, "contact")
                return

    @staticmethod
    def _record_touch(state, robot, result, detail):
        state.ball.last_touch_team = robot.team
        state.ball.last_touch_player = robot.player_id
        result.events.append(RuleEvent("touch", robot.team, robot.player_id, detail=detail))

    def _integrate_ball(self, state):
        state.previous_ball_x = state.ball.x
        state.previous_ball_y = state.ball.y
        state.ball.x += state.ball.vx * self.config.dt
        state.ball.y += state.ball.vy * self.config.dt
        decay = math.exp(-self.config.ball_friction * self.config.dt)
        state.ball.vx *= decay
        state.ball.vy *= decay
        if math.hypot(state.ball.vx, state.ball.vy) < 0.015:
            state.ball.vx = state.ball.vy = 0.0

    def _resolve_robot_collisions(self, state):
        robots = list(state.robots.values())
        for i, first in enumerate(robots):
            for second in robots[i + 1:]:
                dx, dy = second.pose.x - first.pose.x, second.pose.y - first.pose.y
                distance = math.hypot(dx, dy)
                minimum = first.radius + second.radius
                if distance >= minimum:
                    continue
                if distance <= 1e-8:
                    dx, dy, distance = 1.0, 0.0, 1.0
                correction = (minimum - distance) * 0.5
                ux, uy = dx / distance, dy / distance
                first.pose.x -= ux * correction
                first.pose.y -= uy * correction
                second.pose.x += ux * correction
                second.pose.y += uy * correction
                first.vx *= 0.25
                first.vy *= 0.25
                second.vx *= 0.25
                second.vy *= 0.25

    @staticmethod
    def _stop_robots(state):
        for robot in state.robots.values():
            robot.vx = robot.vy = robot.vyaw = 0.0
