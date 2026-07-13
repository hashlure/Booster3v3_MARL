"""Dense shared team reward inspired by GRF checkpoint shaping."""

from __future__ import annotations

import math

from .config import EnvConfig, RewardConfig
from .physics import PhysicsResult
from .types import MatchState, Team


COMPONENTS = (
    "time", "goal", "concede", "ball_progress", "checkpoint", "possession",
    "approach_ball", "successful_pass", "pressured_pass", "forward_pass",
    "turnover", "shot_on_target", "bad_shot", "out_of_bounds",
    "illegal_action", "clustering", "penalty",
    "receiver_facing", "keeper_positioning", "keeper_save",
)


class RewardEngine:
    def __init__(self, env_config: EnvConfig, reward_config: RewardConfig):
        self.env_config = env_config
        self.config = reward_config
        self.previous_progress = {Team.BLUE: 0.0, Team.RED: 0.0}
        self.previous_ball_distance = {Team.BLUE: 0.0, Team.RED: 0.0}
        self.previous_touch = None
        self.previous_touch_x = 0.0
        self.previous_was_pass = False
        self.previous_pass_pressured = False
        self.previous_was_shot = False
        self.checkpoints = (-4.5, -2.5, -0.5, 1.5, 3.5, 5.5)
        self.collected = {Team.BLUE: set(), Team.RED: set()}
        self.last_components = {team: {} for team in (Team.BLUE, Team.RED)}

    def reset(self, state: MatchState):
        self.previous_progress = {Team.BLUE: state.ball.x, Team.RED: -state.ball.x}
        self.previous_ball_distance = {
            team: self._nearest_field_player_distance(state, team)
            for team in (Team.BLUE, Team.RED)
        }
        self.previous_touch = None
        self.previous_touch_x = 0.0
        self.previous_was_pass = False
        self.previous_pass_pressured = False
        self.previous_was_shot = False
        self.collected = {Team.BLUE: set(), Team.RED: set()}
        self.last_components = {team: {name: 0.0 for name in COMPONENTS} for team in (Team.BLUE, Team.RED)}

    def compute(self, state: MatchState, physics: PhysicsResult):
        c = {team: {name: 0.0 for name in COMPONENTS} for team in (Team.BLUE, Team.RED)}
        for team in (Team.BLUE, Team.RED):
            c[team]["time"] += self.config.time

        for event in state.events:
            if event.kind == "goal" and event.team is not None:
                c[event.team]["goal"] += self.config.goal
                c[event.team.opponent]["concede"] += self.config.concede
            elif event.kind == "penalty" and event.team is not None:
                c[event.team]["penalty"] += self.config.penalty
            elif event.kind == "restart" and event.team is not None and event.detail in ("touchline_out", "goal_line_out"):
                c[event.team.opponent]["out_of_bounds"] += self.config.out_of_bounds
            elif event.kind == "touch" and event.team is not None:
                self._touch_reward(state, event, c)

        progress = {Team.BLUE: state.ball.x, Team.RED: -state.ball.x}
        for team in (Team.BLUE, Team.RED):
            delta = progress[team] - self.previous_progress[team]
            c[team]["ball_progress"] += self.config.ball_progress * delta / self.env_config.field_length
            self.previous_progress[team] = progress[team]
            if state.ball.last_touch_team is team:
                c[team]["possession"] += self.config.possession
                for index, threshold in enumerate(self.checkpoints):
                    if index not in self.collected[team] and progress[team] >= threshold:
                        self.collected[team].add(index)
                        c[team]["checkpoint"] += self.config.checkpoint
            else:
                distance = self._nearest_field_player_distance(state, team)
                improvement = self.previous_ball_distance[team] - distance
                c[team]["approach_ball"] += self.config.approach_ball * max(-0.1, min(0.1, improvement))
                self.previous_ball_distance[team] = distance

            active = [r for r in state.team_robots(team) if r.active and r.player_id != 3]
            if len(active) >= 2:
                distance = math.hypot(active[0].pose.x - active[1].pose.x, active[0].pose.y - active[1].pose.y)
                if distance < 1.2:
                    c[team]["clustering"] += self.config.clustering * (1.2 - distance)
            c[team]["receiver_facing"] += self._receiver_facing_reward(state, team)
            c[team]["keeper_positioning"] += self._keeper_positioning_reward(state, team)

        for name in physics.illegal_actions:
            c[state.robots[name].team]["illegal_action"] += self.config.illegal_action

        self.last_components = c
        totals = {team: sum(c[team].values()) for team in (Team.BLUE, Team.RED)}
        return {robot.name: float(totals[robot.team]) for robot in state.robots.values()}

    def _touch_reward(self, state, event, components):
        team = event.team
        robot = state.robots.get("%s_%d" % (team.value, event.player_id))
        touch_x = team.attack_sign * (robot.pose.x if robot is not None else state.ball.x)
        if self.previous_touch is not None:
            previous_team, previous_player = self.previous_touch
            if previous_team is team and previous_player != event.player_id and self.previous_was_pass:
                components[team]["successful_pass"] += self.config.successful_pass
                if self.previous_pass_pressured:
                    components[team]["pressured_pass"] += self.config.pressured_pass
                if touch_x > self.previous_touch_x + 0.25:
                    components[team]["forward_pass"] += self.config.forward_pass
            elif previous_team is not team:
                components[previous_team]["turnover"] += self.config.turnover

        detail = event.detail
        if detail == "shoot":
            projected = self._project_ball_to_goal_y(state, team)
            safe_half = self.env_config.goal_width / 2.0 - self.env_config.ball_radius
            if projected is not None and abs(projected) <= safe_half:
                components[team]["shot_on_target"] += self.config.shot_on_target
            else:
                components[team]["bad_shot"] += self.config.bad_shot

        if (robot is not None and robot.player_id == 3 and self.previous_touch is not None
                and self.previous_touch[0] is team.opponent and self.previous_was_shot):
            components[team]["keeper_save"] += self.config.keeper_save

        pressured = False
        if robot is not None:
            pressured = any(
                opponent.active and math.hypot(opponent.pose.x - robot.pose.x, opponent.pose.y - robot.pose.y) < 1.5
                for opponent in state.team_robots(team.opponent)
            )
        self.previous_touch = (team, event.player_id)
        self.previous_touch_x = touch_x
        self.previous_was_pass = detail == "pass"
        self.previous_was_shot = detail == "shoot"
        self.previous_pass_pressured = pressured and self.previous_was_pass

    def _project_ball_to_goal_y(self, state, team):
        vx = state.ball.vx * team.attack_sign
        vy = state.ball.vy * team.attack_sign
        bx = state.ball.x * team.attack_sign
        by = state.ball.y * team.attack_sign
        goal_x = self.env_config.field_length / 2.0
        if vx <= 1e-6:
            return None
        return by + (goal_x - bx) * vy / vx

    @staticmethod
    def _nearest_field_player_distance(state, team):
        distances = [
            math.hypot(robot.pose.x - state.ball.x, robot.pose.y - state.ball.y)
            for robot in state.team_robots(team) if robot.active and robot.player_id != 3
        ]
        return min(distances, default=10.0)

    def _receiver_facing_reward(self, state, team):
        """Reward off-ball teammates for presenting their front to the carrier."""
        if state.ball.last_touch_team is not team or state.ball.last_touch_player is None:
            return 0.0
        carrier = state.robots.get("%s_%d" % (team.value, state.ball.last_touch_player))
        if carrier is None or not carrier.active:
            return 0.0
        # A stale last-touch marker is not possession.
        if math.hypot(carrier.pose.x - state.ball.x, carrier.pose.y - state.ball.y) > self.env_config.kick_distance * 1.35:
            return 0.0
        total = 0.0
        receivers = 0
        for robot in state.team_robots(team):
            if not robot.active or robot.player_id == carrier.player_id:
                continue
            dx, dy = carrier.pose.x - robot.pose.x, carrier.pose.y - robot.pose.y
            if math.hypot(dx, dy) < 0.40:
                continue
            desired = math.atan2(dy, dx)
            error = (desired - robot.pose.theta + math.pi) % (2.0 * math.pi) - math.pi
            # [0, 1], maximal when directly facing the carrier.
            total += 0.5 * (math.cos(error) + 1.0)
            receivers += 1
        return self.config.receiver_facing * total / max(receivers, 1)

    def _keeper_positioning_reward(self, state, team):
        keeper = state.robots.get("%s_3" % team.value)
        if keeper is None or not keeper.active:
            return 0.0
        half_l = self.env_config.field_length / 2.0
        target_x = -team.attack_sign * (half_l - 0.65)
        target_y = max(-1.05, min(1.05, state.ball.y * 0.60))
        distance = math.hypot(keeper.pose.x - target_x, keeper.pose.y - target_y)
        bearing = math.atan2(state.ball.y - keeper.pose.y, state.ball.x - keeper.pose.x)
        error = (bearing - keeper.pose.theta + math.pi) % (2.0 * math.pi) - math.pi
        facing = 0.5 * (math.cos(error) + 1.0)
        position_quality = math.exp(-distance / 0.75)
        return self.config.keeper_positioning * position_quality * facing
