#!/usr/bin/env python3
"""Swiss-style adjacent-rating tournament for behavior trees and snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "on-policy"))

from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.league import LeaguePool
from robocup3v3.learned_opponent import FrozenActorOpponent
from robocup3v3.opponents import RuleTreeOpponent
from robocup3v3.types import Team


def controller(entry, team, config, hidden_size, layer_n):
    if entry["kind"] == "actor":
        return FrozenActorOpponent(team, config, entry["path"], hidden_size, layer_n)
    return RuleTreeOpponent(team, config, entry["id"].replace("bt_", ""))


def play(first, second, seed, duration, hidden_size, layer_n):
    config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                       match_duration_sec=duration,
                       max_episode_steps=int(duration / .1) + 300,
                       randomize_reset=True, action_noise=0.0)
    env = Robocup3v3Env(config)
    env.reset(seed=seed)
    blue = controller(first, Team.BLUE, config, hidden_size, layer_n)
    red = controller(second, Team.RED, config, hidden_size, layer_n)
    while True:
        actions = blue.actions(env.state)
        actions.update(red.actions(env.state))
        _, _, terminated, truncated, _ = env.step(actions)
        if any(terminated.values()) or any(truncated.values()):
            break
    blue_score, red_score = env.state.score[Team.BLUE], env.state.score[Team.RED]
    return 1.0 if blue_score > red_score else .5 if blue_score == red_score else 0.0, blue_score, red_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league_dir", type=Path, default=ROOT / "results" / "league" / "main")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layer_N", type=int, default=3)
    args = parser.parse_args()
    pool = LeaguePool(args.league_dir, args.seed)
    match_index = 0
    for round_index in range(args.rounds):
        data = pool.load()
        entries = sorted((e for e in data["entries"] if e.get("active", True)),
                         key=lambda e: e["rating"], reverse=True)
        # Alternate the order within rating bands to reduce repeated pairings.
        if round_index % 2 and len(entries) > 2:
            entries[1:-1] = entries[2:-1] + entries[1:2]
        for index in range(0, len(entries) - 1, 2):
            first, second = entries[index], entries[index + 1]
            score_a, goals_a, goals_b = play(first, second, args.seed + match_index,
                                              args.duration, args.hidden_size, args.layer_N)
            match_index += 1
            score_b, goals_b2, goals_a2 = play(second, first, args.seed + match_index,
                                               args.duration, args.hidden_size, args.layer_N)
            match_index += 1
            first_score = .5 * (score_a + (1.0 - score_b))
            pool.update_pair_result(first["id"], second["id"], first_score)
            print("SWISS round=%d %s vs %s score=%.2f goals=%d:%d" %
                  (round_index + 1, first["id"], second["id"], first_score,
                   goals_a + goals_a2, goals_b + goals_b2), flush=True)
    final = pool.load()
    print("SWISS FINAL")
    for entry in sorted(final["entries"], key=lambda e: e["rating"], reverse=True):
        print("  %7.1f %-24s %s active=%s" %
              (entry["rating"], entry["id"], entry["kind"], entry.get("active", True)))


if __name__ == "__main__":
    main()
