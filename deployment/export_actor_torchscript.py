#!/usr/bin/env python3
"""Export a trained non-recurrent MAPPO actor as TorchScript."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "on-policy"))

import torch
from torch import nn

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from robocup3v3.actions import ACTION_NAMES, N_ACTIONS
from robocup3v3.observations import observation_size
from robocup3v3.spaces import Box, Discrete


class DeployableActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.base = actor.base
        self.output = actor.act.action_out.linear

    def forward(self, observation, available_actions):
        features = self.base(observation)
        logits = self.output(features)
        logits = logits.masked_fill(available_actions <= 0.0, -1.0e9)
        return torch.argmax(logits, dim=-1), logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="run directory containing config.json and models/actor.pt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    train_args = Namespace(**config)
    if train_args.use_recurrent_policy or train_args.use_naive_recurrent_policy:
        raise ValueError("TorchScript exporter supports non-recurrent MAPPO; keep actor.pt for RMAPPO")
    actor = R_Actor(
        train_args,
        Box(-10.0, 10.0, shape=(observation_size(),)),
        Discrete(N_ACTIONS),
        device=torch.device("cpu"),
    )
    actor.load_state_dict(
        torch.load(
            str(args.model_dir / "models" / "actor.pt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    actor.eval()
    deployable = DeployableActor(actor).eval()
    example_obs = torch.zeros(3, observation_size(), dtype=torch.float32)
    example_mask = torch.ones(3, N_ACTIONS, dtype=torch.float32)
    traced = torch.jit.trace(deployable, (example_obs, example_mask))
    output = args.output or (args.model_dir / "models" / "actor_deploy.ts")
    traced.save(str(output))
    spec = {
        "format": "torchscript",
        "observation_dim": observation_size(),
        "action_count": N_ACTIONS,
        "action_names": list(ACTION_NAMES),
        "team_view": {"own_goal_x": -7.0, "opponent_goal_x": 7.0, "attack_direction": "+x"},
    }
    (output.parent / "actor_deploy_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    loaded = torch.jit.load(str(output))
    action, logits = loaded(example_obs, example_mask)
    assert action.shape == (3,) and logits.shape == (3, N_ACTIONS)
    print("exported", output)


if __name__ == "__main__":
    main()
