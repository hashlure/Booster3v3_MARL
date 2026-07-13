#!/usr/bin/env python3
"""Record behavior-tree or learned-policy matches as an animated GIF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "on-policy"))

import numpy as np
import torch
from PIL import Image, ImageDraw

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from robocup3v3.actions import N_ACTIONS, available_actions, decode_action
from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.observations import local_observation, observation_size
from robocup3v3.opponents import RuleTreeOpponent
from robocup3v3.spaces import Box, Discrete
from robocup3v3.types import Team
from training.train_mappo import parse_args as parse_mappo_args


TREE_CHOICES = RuleTreeOpponent.DIFFICULTIES


class LearnedTeam:
    def __init__(self, team, config, model_path, device, hidden_size, layer_n):
        self.team, self.config, self.device = team, config, device
        config_path = model_path.parent.parent / "config.json"
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            hidden_size = int(saved.get("hidden_size", hidden_size))
            layer_n = int(saved.get("layer_N", layer_n))
        args = parse_mappo_args([
            "--algorithm_name", "mappo", "--experiment_name", "record",
            "--hidden_size", str(hidden_size), "--layer_N", str(layer_n),
        ])
        self.actor = R_Actor(args, Box(-10, 10, shape=(observation_size(),)),
                             Discrete(N_ACTIONS), device=device)
        self.actor.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
        self.actor.eval()

    def actions(self, state):
        result = {}
        for robot in state.team_robots(self.team):
            observation = torch.as_tensor(
                local_observation(state, robot, self.config)[None], device=self.device)
            mask = available_actions(robot, state, self.config)
            with torch.no_grad():
                logits = self.actor.act.action_out.linear(self.actor.base(observation))
                logits.masked_fill_(torch.as_tensor(mask[None], device=self.device) <= 0, -1e9)
                action_id = int(logits.argmax(dim=-1).item())
            result[robot.name] = decode_action(action_id, robot, state, self.config)
        return result


def controller(spec, team, config, actor_path, device, hidden_size, layer_n):
    if spec == "actor":
        if actor_path is None:
            raise ValueError("%s actor requires its --*_actor path" % team.value)
        return LearnedTeam(team, config, actor_path, device, hidden_size, layer_n)
    return RuleTreeOpponent(team, config, difficulty=spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blue", choices=TREE_CHOICES + ("actor",), default="standard")
    parser.add_argument("--red", choices=TREE_CHOICES + ("actor",), default="standard")
    parser.add_argument("--blue_actor", type=Path)
    parser.add_argument("--red_actor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".gif":
        raise ValueError("server-safe recorder currently writes .gif; use an output ending in .gif")
    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                       match_duration_sec=args.duration,
                       max_episode_steps=int(args.duration / .1) + 300,
                       randomize_reset=True, action_noise=.0)
    env = Robocup3v3Env(config, render_mode="rgb_array")
    env.reset(seed=args.seed)
    blue = controller(args.blue, Team.BLUE, config, args.blue_actor, device,
                      args.hidden_size, args.layer_N)
    red = controller(args.red, Team.RED, config, args.red_actor, device,
                     args.hidden_size, args.layer_N)
    frames, events = [], {}
    while True:
        actions = blue.actions(env.state)
        actions.update(red.actions(env.state))
        _, _, terminated, truncated, infos = env.step(actions)
        for event in infos["blue_1"]["events"]:
            events[event["kind"]] = events.get(event["kind"], 0) + 1
        if env.state.step_count % args.frame_stride == 0:
            frame = Image.fromarray(env.render())
            draw = ImageDraw.Draw(frame)
            text = "%s vs %s   BLUE %d : %d RED   t=%.1f   %s/%s" % (
                args.blue, args.red, env.state.score[Team.BLUE], env.state.score[Team.RED],
                env.state.elapsed, env.state.game_state.value, env.state.set_play.value)
            draw.rectangle((0, 0, frame.width, 28), fill=(15, 15, 15))
            draw.text((10, 7), text, fill=(255, 255, 255))
            frames.append(frame)
        if any(terminated.values()) or any(truncated.values()):
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        frames.append(Image.fromarray(env.render()))
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=max(20, int(1000 / args.fps)), loop=0, optimize=False)
    summary = {
        "blue": args.blue, "red": args.red, "seed": args.seed,
        "score": {"blue": env.state.score[Team.BLUE], "red": env.state.score[Team.RED]},
        "steps": env.state.step_count, "events": events, "frames": len(frames),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RECORDED %s score=%d:%d frames=%d" %
          (args.output, env.state.score[Team.BLUE], env.state.score[Team.RED], len(frames)))


if __name__ == "__main__":
    main()
