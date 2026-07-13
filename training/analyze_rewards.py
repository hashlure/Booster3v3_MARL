#!/usr/bin/env python3
"""Compare reward components after normalizing by environment steps/matches."""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.opponents import RuleTreeOpponent
from robocup3v3.types import Team


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--blue", default="standard")
    parser.add_argument("--red", default="standard")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    totals = collections.defaultdict(float)
    total_steps = goals_for = goals_against = 0
    for match in range(args.matches):
        config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                           match_duration_sec=args.duration,
                           max_episode_steps=int(args.duration / .1) + 300,
                           randomize_reset=True, action_noise=0.0)
        env = Robocup3v3Env(config)
        env.reset(seed=args.seed + match)
        blue = RuleTreeOpponent(Team.BLUE, config, args.blue)
        red = RuleTreeOpponent(Team.RED, config, args.red)
        while True:
            actions = blue.actions(env.state)
            actions.update(red.actions(env.state))
            _, _, terminated, truncated, infos = env.step(actions)
            for key, value in infos["blue_1"]["reward_components"].items():
                totals[key] += float(value)
            if any(terminated.values()) or any(truncated.values()):
                break
        total_steps += env.state.step_count
        goals_for += env.state.score[Team.BLUE]
        goals_against += env.state.score[Team.RED]
    print("matches=%d steps=%d score=%d:%d" %
          (args.matches, total_steps, goals_for, goals_against))
    print("%-24s %14s %18s %14s" %
          ("component", "raw_sum", "per_1000_steps", "per_match"))
    for key in sorted(totals, key=lambda name: abs(totals[name]), reverse=True):
        print("%-24s %14.6f %18.6f %14.6f" %
              (key, totals[key], totals[key] * 1000.0 / max(total_steps, 1),
               totals[key] / args.matches))


if __name__ == "__main__":
    main()
