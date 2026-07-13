"""Six-agent parallel core environment with Gymnasium/PettingZoo semantics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .actions import N_ACTIONS, available_actions, decode_action
from .config import EnvConfig, RewardConfig
from .observations import global_state, hybrid_global_state, local_observation
from .physics import PhysicsEngine
from .rewards import RewardEngine
from .rendering import RGBRenderer
from .rules import RulesEngine, create_match_state
from .types import PlannerAction, Team


class Robocup3v3Env:
    """Dependency-light core.

    ``step`` accepts either portable discrete action IDs or fully decoded
    ``PlannerAction`` objects for all six logical agents.
    """

    metadata = {"name": "robocup3v3_v0", "render_modes": ["ansi", "rgb_array"]}
    possible_agents = tuple("%s_%d" % (team, player) for team in ("blue", "red") for player in (1, 2, 3))

    def __init__(self, config=None, reward_config=None, render_mode=None):
        self.config = config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.physics = PhysicsEngine(self.config)
        self.rules = RulesEngine(self.config)
        self.reward_engine = RewardEngine(self.config, self.reward_config)
        self._rgb_renderer = RGBRenderer()
        self.np_random = np.random.RandomState(0)
        self.seed_value = 0
        self.state = None
        self.agents = list(self.possible_agents)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        options = options or {}
        kickoff = Team(options.get("kickoff_team", Team.BLUE.value))
        randomize = bool(options.get("randomize", self.config.randomize_reset))
        self.state = create_match_state(self.config, self.np_random, kickoff, randomize)
        self.reward_engine.reset(self.state)
        self.agents = list(self.possible_agents)
        return self._observations(), self._infos()

    def seed(self, seed=None):
        self.seed_value = 0 if seed is None else int(seed)
        self.np_random = np.random.RandomState(self.seed_value)
        return [self.seed_value]

    def step(self, actions):
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        if self.state.terminated or self.state.truncated:
            raise RuntimeError("step() called after episode end; call reset()")
        self.rules.begin_step(self.state)
        decoded = {}
        for name in self.possible_agents:
            robot = self.state.robots[name]
            raw = actions.get(name, 0)
            decoded[name] = raw if isinstance(raw, PlannerAction) else decode_action(raw, robot, self.state, self.config)
        physics = self.physics.step(
            self.state,
            decoded,
            self.np_random,
            lambda robot: self.rules.can_touch(self.state, robot),
        )
        self.rules.accept_physics_events(self.state, physics.events)
        self.rules.end_step(self.state)
        rewards = self.reward_engine.compute(self.state, physics)
        terminations = {name: self.state.terminated for name in self.possible_agents}
        truncations = {name: self.state.truncated for name in self.possible_agents}
        infos = self._infos()
        for name in physics.illegal_actions:
            infos[name]["illegal_action"] = True
        if self.state.terminated or self.state.truncated:
            self.agents = []
        return self._observations(), rewards, terminations, truncations, infos

    def _observations(self):
        return {
            name: local_observation(self.state, self.state.robots[name], self.config)
            for name in self.possible_agents
        }

    def _infos(self):
        events = [
            {"kind": e.kind, "team": e.team.value if e.team else None, "player_id": e.player_id, "set_play": e.set_play.value, "detail": e.detail}
            for e in (self.state.events if self.state else [])
        ]
        return {
            name: {
                "events": events,
                "score": {"blue": self.state.score[Team.BLUE], "red": self.state.score[Team.RED]} if self.state else {"blue": 0, "red": 0},
                "game_state": self.state.game_state.value if self.state else "INITIAL",
                "set_play": self.state.set_play.value if self.state else "NONE",
                "bad_transition": bool(self.state.truncated) if self.state else False,
                "termination_reason": self.state.termination_reason if self.state else "",
                "reward_components": dict(self.reward_engine.last_components.get(self.state.robots[name].team, {})) if self.state else {},
            }
            for name in self.possible_agents
        }

    def state_for_team(self, team, birdview=False, height=32, width=48):
        perspective = Team(team)
        if birdview:
            return hybrid_global_state(self.state, perspective, self.config, height, width)
        return global_state(self.state, perspective, self.config)

    def action_mask(self, agent):
        return available_actions(self.state.robots[agent], self.state, self.config)

    def render(self):
        if self.state is None:
            return "<uninitialized Robocup3v3Env>"
        if self.render_mode == "rgb_array":
            return self._rgb_renderer.render(self.state, self.config)
        lines = [
            "state=%s set_play=%s score=%d-%d t=%.1f" % (
                self.state.game_state.value, self.state.set_play.value,
                self.state.score[Team.BLUE], self.state.score[Team.RED], self.state.elapsed,
            ),
            "ball=(%.2f, %.2f) v=(%.2f, %.2f)" % (self.state.ball.x, self.state.ball.y, self.state.ball.vx, self.state.ball.vy),
        ]
        for name in self.possible_agents:
            robot = self.state.robots[name]
            lines.append("%s=(%.2f, %.2f, %.2f) active=%s" % (name, robot.pose.x, robot.pose.y, robot.pose.theta, robot.active))
        text = "\n".join(lines)
        if self.render_mode == "ansi":
            return text
        print(text)
        return None

    def close(self):
        return None
