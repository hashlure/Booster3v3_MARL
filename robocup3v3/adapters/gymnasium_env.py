"""Modern single-team Gymnasium-style wrapper for debugging."""

from __future__ import annotations

import numpy as np

from ..actions import N_ACTIONS
from ..config import EnvConfig, RewardConfig
from ..observations import global_state_size, observation_size
from ..opponents import HeuristicOpponent
from ..spaces import Box, MultiDiscrete
from ..types import Team
from ..env import Robocup3v3Env

try:
    import gymnasium as gym
    BaseEnv = gym.Env
except ImportError:
    try:
        import gym
        BaseEnv = gym.Env
    except ImportError:
        BaseEnv = object


class Robocup3v3TeamGymEnv(BaseEnv):
    """Joint three-Actor view while retaining per-player observation rows."""

    metadata = Robocup3v3Env.metadata

    def __init__(self, config=None, reward_config=None, controlled_team="blue", render_mode=None):
        self.config = config or EnvConfig()
        self.team = Team(controlled_team)
        self.core = Robocup3v3Env(self.config, reward_config or RewardConfig(), render_mode)
        self.opponent = HeuristicOpponent(self.team.opponent, self.config)
        self.agent_names = tuple("%s_%d" % (self.team.value, i) for i in (1, 2, 3))
        self.observation_space = Box(-10.0, 10.0, shape=(3, observation_size()), dtype=np.float32)
        self.action_space = MultiDiscrete([N_ACTIONS, N_ACTIONS, N_ACTIONS])

    def reset(self, seed=None, options=None):
        observations, infos = self.core.reset(seed=seed, options=options)
        return self._obs(observations), self._info(infos)

    def step(self, actions):
        action_map = {name: int(np.asarray(actions)[idx]) for idx, name in enumerate(self.agent_names)}
        action_map.update(self.opponent.actions(self.core.state))
        observations, rewards, terminations, truncations, infos = self.core.step(action_map)
        reward = float(np.mean([rewards[name] for name in self.agent_names]))
        return (
            self._obs(observations),
            reward,
            any(terminations.values()),
            any(truncations.values()),
            self._info(infos),
        )

    def _obs(self, observations):
        return np.stack([observations[name] for name in self.agent_names]).astype(np.float32)

    def _info(self, infos):
        return {
            "agents": {name: infos[name] for name in self.agent_names},
            "global_state": self.core.state_for_team(self.team.value),
            "available_actions": np.stack([self.core.action_mask(name) for name in self.agent_names]),
        }

    def render(self):
        return self.core.render()

    def close(self):
        return self.core.close()

