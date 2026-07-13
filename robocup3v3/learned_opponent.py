"""Frozen MAPPO Actor used as a league opponent during centralized training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from .actions import N_ACTIONS, available_actions, decode_action
from .observations import local_observation, observation_size
from .spaces import Box, Discrete


class FrozenActorOpponent:
    def __init__(self, team, config, model_path, hidden_size=512, layer_n=3):
        self.team, self.config = team, config
        self.model_path = str(model_path)
        path = Path(model_path)
        config_path = path.parent.parent / "config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            hidden_size = int(saved.get("hidden_size", hidden_size))
            layer_n = int(saved.get("layer_N", layer_n))
        # Construct only the fields consumed by R_Actor/MLPBase.
        from training.train_mappo import parse_args as parse_mappo_args
        args = parse_mappo_args([
            "--algorithm_name", "mappo", "--experiment_name", "league_opponent",
            "--hidden_size", str(hidden_size), "--layer_N", str(layer_n),
        ])
        self.actor = R_Actor(args, Box(-10, 10, shape=(observation_size(),)),
                             Discrete(N_ACTIONS), device=torch.device("cpu"))
        self.actor.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        self.actor.eval()

    def actions(self, state):
        result = {}
        with torch.no_grad():
            for robot in state.team_robots(self.team):
                obs = torch.from_numpy(local_observation(state, robot, self.config)[None])
                mask = available_actions(robot, state, self.config)
                logits = self.actor.act.action_out.linear(self.actor.base(obs))
                logits.masked_fill_(torch.from_numpy(mask[None]) <= 0, -1e9)
                action_id = int(logits.argmax(dim=-1).item())
                result[robot.name] = decode_action(action_id, robot, state, self.config)
        return result
