"""Optional PettingZoo ParallelEnv facade around the dependency-light core."""

from __future__ import annotations

import numpy as np

from ..actions import N_ACTIONS
from ..env import Robocup3v3Env
from ..observations import observation_size
from ..spaces import Box, Discrete

try:
    from pettingzoo import ParallelEnv
except ImportError:  # Keep core/on-policy operation available without PettingZoo.
    class ParallelEnv(object):
        pass


class Robocup3v3ParallelEnv(ParallelEnv):
    metadata = Robocup3v3Env.metadata

    def __init__(self, config=None, reward_config=None, render_mode=None):
        self.core = Robocup3v3Env(config, reward_config, render_mode)
        self.possible_agents = list(self.core.possible_agents)
        self.agents = list(self.possible_agents)
        self._observation_space = Box(-10.0, 10.0, shape=(observation_size(),), dtype=np.float32)
        self._action_space = Discrete(N_ACTIONS)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        obs, infos = self.core.reset(seed=seed, options=options)
        self.agents = list(self.core.agents)
        return obs, infos

    def step(self, actions):
        result = self.core.step(actions)
        self.agents = list(self.core.agents)
        return result

    def render(self):
        return self.core.render()

    def close(self):
        return self.core.close()

