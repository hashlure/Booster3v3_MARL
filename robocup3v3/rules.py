"""3v3 match state machine, restarts, penalties, goals, and reset semantics."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .config import EnvConfig
from .types import (
    BallSimState, GameState, MatchState, Penalty, Pose2D, RobotSimState,
    RuleEvent, SetPlay, Team,
)


def standard_robots(config: EnvConfig, rng: np.random.RandomState, randomize: bool):
    positions = {
        # Both front players start outside the mandatory 1.45 m kickoff
        # avoidance radius. The awarded team approaches after PLAYING begins.
        Team.BLUE: ((-1.8, 0.0), (-2.2, 1.65), (-6.15, 0.0)),
        Team.RED: ((1.8, 0.0), (2.2, -1.65), (6.15, 0.0)),
    }
    robots = {}
    for team in (Team.BLUE, Team.RED):
        for player_id, (x, y) in enumerate(positions[team], start=1):
            noise_x = rng.uniform(-config.position_noise, config.position_noise) if randomize else 0.0
            noise_y = rng.uniform(-config.position_noise, config.position_noise) if randomize else 0.0
            theta = 0.0 if team is Team.BLUE else math.pi
            robot = RobotSimState(
                team=team,
                player_id=player_id,
                pose=Pose2D(x + noise_x, y + noise_y, theta),
                radius=config.robot_radius,
            )
            robots[robot.name] = robot
    return robots


def create_match_state(config: EnvConfig, rng: np.random.RandomState, kickoff: Team, randomize: bool):
    return MatchState(
        robots=standard_robots(config, rng, randomize),
        ball=BallSimState(radius=config.ball_radius),
        game_state=GameState.READY,
        set_play=SetPlay.NONE,
        kicking_team=kickoff,
        stopped=False,
        restart_pending=True,
        restart_awarded_team=kickoff,
        restart_started_at=0.0,
        restart_expires_at=config.kickoff_expiry_sec,
        direct_goal_allowed=False,
    )


class RulesEngine:
    def __init__(self, config: EnvConfig):
        self.config = config

    def begin_step(self, state: MatchState):
        state.events = []
        state.packet_number += 1
        self._tick_penalties(state)
        self._advance_state_machine(state)
        self._expire_restart(state)

    def can_touch(self, state: MatchState, robot: RobotSimState) -> bool:
        if state.game_state is not GameState.PLAYING or state.stopped or not robot.active:
            return False
        if state.kicking_team is not None and not state.restart_touches:
            return robot.team is state.kicking_team
        return True

    def accept_physics_events(self, state: MatchState, events):
        for event in events:
            state.events.append(event)
            if event.kind != "touch" or event.team is None or event.player_id is None:
                continue
            touch = (event.team, event.player_id)
            robot = state.robots.get("%s_%d" % (event.team.value, event.player_id))
            # The avoidance-distance check applies before the restart's first
            # legal touch, not again at the receiving player's second touch.
            if (state.restart_pending and state.restart_awarded_team is not None
                    and not state.restart_touches):
                if event.team is not state.restart_awarded_team:
                    if robot is not None:
                        self._penalize(state, robot, Penalty.ILLEGAL_POSITIONING)
                    state.events.append(RuleEvent("restart_retake", state.restart_awarded_team, set_play=state.restart_origin, detail="defender_touched_first"))
                    self._retake_restart(state)
                    continue
                violator = self._restart_distance_violator(state)
                if violator is not None:
                    self._penalize(state, violator, Penalty.ILLEGAL_POSITIONING)
                    state.events.append(RuleEvent("restart_retake", state.restart_awarded_team, violator.player_id, state.restart_origin, "avoidance_distance"))
                    self._retake_restart(state)
                    continue
            if not state.restart_touches or state.restart_touches[-1] != touch:
                state.restart_touches.append(touch)
            state.last_touch_at = state.elapsed
            if len(set(state.restart_touches)) >= 2:
                state.direct_goal_allowed = True
                state.restart_pending = False
            if state.kicking_team is not None:
                # A legal first touch releases normal play. Keep the touch list so
                # indirect-free-kick validation and telemetry can inspect it.
                state.kicking_team = None
                state.set_play = SetPlay.NONE

    def start_restart(self, state: MatchState, set_play: SetPlay, team: Team, x: float, y: float):
        """Public deterministic restart hook for referee scenarios and tests."""

        self._local_restart(state, set_play, team, x, y, "external_restart")

    def end_step(self, state: MatchState):
        self._evaluate_robot_rules(state)
        boundary = self._evaluate_ball_boundary(state)
        if not boundary:
            state.elapsed += self.config.dt if state.game_state is GameState.PLAYING else 0.0
            state.state_elapsed += self.config.dt
        self._check_stalemate(state)
        state.step_count += 1
        if state.elapsed >= self.config.match_duration_sec or max(state.score.values()) >= self.config.score_limit:
            state.game_state = GameState.FINISHED
            state.stopped = True
            state.terminated = True
            state.termination_reason = "match_finished"
            state.events.append(RuleEvent("match_finished"))
        elif state.step_count >= self.config.max_episode_steps:
            state.truncated = True
            state.termination_reason = "time_limit"
            state.events.append(RuleEvent("time_limit"))
        self._check_finite(state)

    def _advance_state_machine(self, state):
        if state.game_state is GameState.READY and state.state_elapsed >= self.config.ready_duration_sec:
            state.game_state = GameState.SET
            state.state_elapsed = 0.0
            state.stopped = True
            state.events.append(RuleEvent("state_changed", detail="SET"))
        elif state.game_state is GameState.SET and state.state_elapsed >= self.config.set_duration_sec:
            state.game_state = GameState.PLAYING
            state.state_elapsed = 0.0
            state.stopped = False
            state.events.append(RuleEvent("state_changed", detail="PLAYING"))
        elif (
            state.game_state is GameState.PLAYING
            and state.stopped
            and state.set_play is not SetPlay.NONE
            and state.state_elapsed >= self.config.restart_placing_duration_sec
        ):
            state.stopped = False
            state.state_elapsed = 0.0
            state.events.append(RuleEvent("restart_active", state.kicking_team, set_play=state.set_play))

    def _evaluate_ball_boundary(self, state):
        ball = state.ball
        half_l = self.config.field_length / 2.0
        half_w = self.config.field_width / 2.0
        px, py = state.previous_ball_x, state.previous_ball_y
        dx, dy = ball.x - px, ball.y - py
        boundaries = (
            ("right", half_l + ball.radius, px, dx, 0),
            ("left", -half_l - ball.radius, px, dx, 1),
            ("top", half_w + ball.radius, py, dy, 2),
            ("bottom", -half_w - ball.radius, py, dy, 3),
        )
        candidates = []
        for name, threshold, previous, delta, priority in boundaries:
            beyond = previous + delta > threshold if threshold > 0 else previous + delta < threshold
            if not beyond:
                continue
            if (threshold > 0 and previous >= threshold) or (threshold < 0 and previous <= threshold):
                crossing_t = 0.0
            elif abs(delta) > 1e-12:
                crossing_t = (threshold - previous) / delta
            else:
                crossing_t = 0.0
            if 0.0 <= crossing_t <= 1.0:
                candidates.append((crossing_t, priority, name))
        if not candidates:
            return False
        crossing_t, _, boundary = min(candidates)
        crossing_x = px + crossing_t * dx
        crossing_y = py + crossing_t * dy

        if boundary in ("right", "left") and abs(crossing_y) + ball.radius < self.config.goal_width / 2.0:
            scoring = Team.BLUE if boundary == "right" else Team.RED
            if (
                state.restart_origin in (SetPlay.THROW_IN, SetPlay.GOAL_KICK, SetPlay.CORNER_KICK)
                and state.restart_awarded_team is scoring.opponent
                and state.restart_pending
            ):
                state.events.append(RuleEvent("disallowed_restart_own_goal", scoring.opponent, set_play=state.restart_origin))
                y = math.copysign(half_w - self.config.restart_inset, crossing_y if crossing_y else 1.0)
                x = half_l - self.config.restart_inset if boundary == "right" else -half_l + self.config.restart_inset
                self._local_restart(state, SetPlay.CORNER_KICK, scoring, x, y, "restart_own_goal")
            elif state.direct_goal_allowed:
                self._goal_reset(state, scoring)
            else:
                defending = scoring.opponent
                state.events.append(RuleEvent("disallowed_direct_goal", scoring, set_play=state.restart_origin))
                x = (-self.config.field_length / 2.0 + self.config.goal_area_length) if defending is Team.BLUE else (self.config.field_length / 2.0 - self.config.goal_area_length)
                self._local_restart(state, SetPlay.GOAL_KICK, defending, x, 0.0, "direct_goal_not_allowed")
            return True
        if boundary in ("top", "bottom"):
            awarded = ball.last_touch_team.opponent if ball.last_touch_team else Team.BLUE
            x = float(np.clip(crossing_x, -half_l + 0.5, half_l - 0.5))
            y = half_w - self.config.restart_inset if boundary == "top" else -half_w + self.config.restart_inset
            self._local_restart(state, SetPlay.THROW_IN, awarded, x, y, "touchline_out")
            return True
        if boundary in ("right", "left"):
            defending = Team.RED if boundary == "right" else Team.BLUE
            attacking = defending.opponent
            if ball.last_touch_team is defending:
                set_play = SetPlay.CORNER_KICK
                awarded = attacking
                x = half_l - self.config.restart_inset if boundary == "right" else -half_l + self.config.restart_inset
                y = math.copysign(half_w - self.config.restart_inset, crossing_y if crossing_y else 1.0)
            else:
                set_play = SetPlay.GOAL_KICK
                awarded = defending
                x = (half_l - self.config.goal_area_length) * defending.attack_sign * -1.0
                y = math.copysign(2.0, crossing_y if crossing_y else 1.0)
            self._local_restart(state, set_play, awarded, x, y, "goal_line_out")
            return True
        return False

    def _goal_reset(self, state, scoring):
        conceding = scoring.opponent
        state.score[scoring] += 1
        state.events.append(RuleEvent("goal", scoring, detail="score=%d-%d" % (state.score[Team.BLUE], state.score[Team.RED])))
        self._reset_positions_preserving_match(state)
        # Start a fresh stalemate window. Without this reset, elapsed time is
        # still far beyond the previous touch and PLAYING immediately triggers
        # another stalemate on every subsequent tick.
        state.last_touch_at = state.elapsed
        state.game_state = GameState.READY
        state.state_elapsed = 0.0
        state.set_play = SetPlay.NONE
        state.kicking_team = conceding
        state.restart_origin = SetPlay.NONE
        state.direct_goal_allowed = True
        state.restart_pending = True
        state.restart_awarded_team = conceding
        state.restart_started_at = state.elapsed
        state.restart_expires_at = state.elapsed + self.config.kickoff_expiry_sec
        state.direct_goal_allowed = False
        state.stopped = False

    def _local_restart(self, state, set_play, awarded, x, y, detail):
        state.ball.x, state.ball.y = x, y
        state.ball.vx = state.ball.vy = 0.0
        state.ball.last_touch_team = None
        state.ball.last_touch_player = None
        state.set_play = set_play
        state.kicking_team = awarded
        state.restart_touches = []
        state.restart_origin = set_play
        state.direct_goal_allowed = set_play not in (SetPlay.INDIRECT_FREE_KICK, SetPlay.THROW_IN)
        state.restart_pending = True
        state.restart_awarded_team = awarded
        state.restart_ball_x = x
        state.restart_ball_y = y
        state.restart_started_at = state.elapsed
        state.restart_expires_at = state.elapsed + (
            self.config.kickoff_expiry_sec if set_play is SetPlay.NONE else self.config.set_play_expiry_sec
        )
        state.stopped = True
        state.state_elapsed = 0.0
        state.events.append(RuleEvent("restart", awarded, set_play=set_play, detail=detail))

    def _reset_positions_preserving_match(self, state):
        fresh = standard_robots(self.config, np.random.RandomState(0), False)
        state.robots = fresh
        state.ball = BallSimState(radius=self.config.ball_radius)
        state.restart_touches = []

    def _restart_distance_violator(self, state):
        if state.restart_awarded_team is None:
            return None
        for robot in state.robots.values():
            if robot.team is state.restart_awarded_team or not robot.active:
                continue
            dx, dy = robot.pose.x - state.ball.x, robot.pose.y - state.ball.y
            distance = math.hypot(dx, dy)
            if distance >= self.config.restart_avoid_distance:
                continue
            radial_velocity = (dx * robot.vx + dy * robot.vy) / max(distance, 1e-8)
            moving_away = radial_velocity > 0.05
            own_goal_x = -robot.team.attack_sign * self.config.field_length / 2.0
            goal_line_guard = abs(robot.pose.x - own_goal_x) <= robot.radius and abs(robot.pose.y) <= self.config.goal_width / 2.0
            if not moving_away and not goal_line_guard:
                return robot
        return None

    def _expire_restart(self, state):
        if not state.restart_pending or state.elapsed < state.restart_expires_at:
            return
        state.restart_pending = False
        state.kicking_team = None
        state.set_play = SetPlay.NONE
        state.direct_goal_allowed = True
        state.events.append(RuleEvent("restart_expired", state.restart_awarded_team, set_play=state.restart_origin))

    def _retake_restart(self, state):
        self._local_restart(
            state,
            state.restart_origin,
            state.restart_awarded_team,
            state.restart_ball_x,
            state.restart_ball_y,
            "retake",
        )

    def _check_stalemate(self, state):
        if (
            state.game_state is not GameState.PLAYING
            or state.stopped
            or state.restart_pending
            or state.elapsed - state.last_touch_at < self.config.stalemate_duration_sec
        ):
            return
        state.events.append(RuleEvent("stalemate_restart"))
        self._reset_positions_preserving_match(state)
        state.last_touch_at = state.elapsed
        state.game_state = GameState.READY
        state.state_elapsed = 0.0
        state.set_play = SetPlay.NONE
        state.kicking_team = None
        state.stopped = False
        state.restart_pending = False
        state.direct_goal_allowed = True

    def _evaluate_robot_rules(self, state):
        half_l = self.config.field_length / 2.0
        half_w = self.config.field_width / 2.0
        for robot in state.robots.values():
            outside = abs(robot.pose.x) > half_l + robot.radius or abs(robot.pose.y) > half_w + robot.radius
            if outside and robot.active:
                self._penalize(state, robot, Penalty.LEAVING_THE_FIELD)
                robot.pose.x = float(np.clip(robot.pose.x, -half_l + 0.4, half_l - 0.4))
                robot.pose.y = float(np.clip(robot.pose.y, -half_w + 0.4, half_w - 0.4))
            close = math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y) <= self.config.kick_distance
            if close and state.ball.last_touch_team is robot.team and state.ball.last_touch_player == robot.player_id:
                robot.possession_time += self.config.dt
                if robot.possession_time > self.config.ball_holding_limit_sec and robot.active:
                    self._penalize(state, robot, Penalty.BALL_HOLDING)
            else:
                robot.possession_time = 0.0

    def _penalize(self, state, robot, penalty):
        robot.penalty = penalty
        robot.penalty_remaining = self.config.penalty_duration_sec
        robot.vx = robot.vy = robot.vyaw = 0.0
        state.events.append(RuleEvent("penalty", robot.team, robot.player_id, detail=penalty.value))

    def _tick_penalties(self, state):
        for robot in state.robots.values():
            if robot.penalty_remaining <= 0.0:
                continue
            robot.penalty_remaining = max(0.0, robot.penalty_remaining - self.config.dt)
            if robot.penalty_remaining == 0.0:
                robot.penalty = Penalty.NONE
                state.events.append(RuleEvent("unpenalized", robot.team, robot.player_id))

    @staticmethod
    def _check_finite(state):
        values = [state.ball.x, state.ball.y, state.ball.vx, state.ball.vy]
        for robot in state.robots.values():
            values.extend((robot.pose.x, robot.pose.y, robot.pose.theta, robot.vx, robot.vy, robot.vyaw))
        if not all(math.isfinite(value) for value in values):
            state.truncated = True
            state.termination_reason = "non_finite_state"
            state.events.append(RuleEvent("simulation_error", detail="non_finite_state"))
