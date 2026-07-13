#!/usr/bin/env python3
"""Evaluate one frozen actor against qualified behavior-tree opponents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "on-policy"))

import torch

from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.learned_opponent import FrozenActorOpponent
from robocup3v3.opponents import RuleTreeOpponent
from robocup3v3.types import Penalty, Team


def evaluate(model, difficulty, matches, duration, seed, hidden_size, layer_n,
             opponent_count=3):
    wins = draws = losses = goals_for = goals_against = scoreless = 0
    touches = shots = passes = saves = 0
    results = []
    config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                       match_duration_sec=duration,
                       max_episode_steps=int(duration / .1) + 300,
                       randomize_reset=True, action_noise=0.0)
    # Actor weights are immutable during evaluation. Loading the network once is
    # substantially faster and avoids repeatedly constructing PyTorch resources.
    learner = FrozenActorOpponent(Team.BLUE, config, model, hidden_size, layer_n)
    opponent = RuleTreeOpponent(Team.RED, config, difficulty)
    for match in range(matches):
        env = Robocup3v3Env(config)
        env.reset(seed=seed + match)
        active_ids = {0: (), 1: (1,), 2: (1, 3), 3: (1, 2, 3)}[opponent_count]
        for robot in env.state.team_robots(Team.RED):
            if robot.player_id not in active_ids:
                robot.penalty = Penalty.SENT_OFF
                robot.penalty_remaining = 1.0e9
                robot.pose.y = config.field_width / 2.0 + 2.0 + robot.player_id
        event_counts = {}
        while True:
            actions = learner.actions(env.state)
            actions.update(opponent.actions(env.state))
            _, _, terminated, truncated, infos = env.step(actions)
            for event in infos["blue_1"]["events"]:
                event_counts[event["kind"]] = event_counts.get(event["kind"], 0) + 1
                detail = event.get("detail", "")
                if event["kind"] == "touch" and event.get("team") == Team.BLUE.value and detail == "shoot":
                    shots += 1
                if event["kind"] == "touch" and event.get("team") == Team.BLUE.value and detail == "pass":
                    passes += 1
            components = infos["blue_1"].get("reward_components", {})
            if components.get("keeper_save", 0.0) > 0:
                saves += 1
            if any(terminated.values()) or any(truncated.values()):
                break
        own, other = env.state.score[Team.BLUE], env.state.score[Team.RED]
        wins += int(own > other)
        draws += int(own == other)
        losses += int(own < other)
        goals_for += own
        goals_against += other
        scoreless += int(own == 0)
        touches += event_counts.get("touch", 0)
        results.append({"seed": seed + match, "score": [own, other]})
    return {
        "opponent": difficulty, "matches": matches, "wins": wins,
        "draws": draws, "losses": losses, "win_rate": wins / matches,
        "non_loss_rate": (wins + draws) / matches,
        "goals_for": goals_for, "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "goals_for_per_match": goals_for / matches,
        "goals_against_per_match": goals_against / matches,
        "scoreless_rate": scoreless / matches, "touches": touches,
        "shots": shots, "passes": passes, "keeper_saves": saves,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--matches", type=int, default=10)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--opponent_count", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--opponents", type=str, default="novice,standard,expert")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"actor": str(args.actor), "evaluations": {}}
    difficulties = tuple(value.strip() for value in args.opponents.split(",") if value.strip())
    invalid = set(difficulties) - set(RuleTreeOpponent.ALL_DIFFICULTIES)
    if not difficulties or invalid:
        raise ValueError("invalid opponents: %s" % sorted(invalid))
    for index, difficulty in enumerate(difficulties):
        result = evaluate(args.actor, difficulty, args.matches, args.duration,
                          args.seed + index * 1000, args.hidden_size, args.layer_N,
                          args.opponent_count)
        report["evaluations"][difficulty] = result
        print("EVAL %-8s W/D/L=%d/%d/%d score=%d:%d win=%.3f nonloss=%.3f scoreless=%.3f shots=%d passes=%d saves=%d" %
              (difficulty, result["wins"], result["draws"], result["losses"],
               result["goals_for"], result["goals_against"], result["win_rate"],
               result["non_loss_rate"], result["scoreless_rate"], result["shots"],
               result["passes"], result["keeper_saves"]), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
