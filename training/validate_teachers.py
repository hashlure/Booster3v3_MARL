#!/usr/bin/env python3
"""Qualification suite for behavior-tree teachers under the 3v3 rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.opponents import RuleTreeOpponent
from robocup3v3.types import Team


MATCHUPS = (
    ("novice", "stationary"),
    ("standard", "stationary"),
    ("expert", "stationary"),
    ("standard", "novice"),
    ("expert", "standard"),
    ("standard", "standard"),
)


def run_match(left, right, seed, duration):
    config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                       match_duration_sec=duration,
                       max_episode_steps=int(duration / .1) + 300,
                       randomize_reset=True, action_noise=.0)
    env = Robocup3v3Env(config)
    env.reset(seed=seed)
    blue = RuleTreeOpponent(Team.BLUE, config, difficulty=left)
    red = RuleTreeOpponent(Team.RED, config, difficulty=right)
    events, event_details = {}, {}
    penalties_by_team = {"blue": 0, "red": 0}
    while True:
        actions = blue.actions(env.state)
        actions.update(red.actions(env.state))
        _, _, terminated, truncated, infos = env.step(actions)
        for event in infos["blue_1"]["events"]:
            events[event["kind"]] = events.get(event["kind"], 0) + 1
            detail_key = "%s:%s" % (event["kind"], event.get("detail", ""))
            event_details[detail_key] = event_details.get(detail_key, 0) + 1
            if event["kind"] == "penalty" and event.get("team") in penalties_by_team:
                penalties_by_team[event["team"]] += 1
        if any(terminated.values()) or any(truncated.values()):
            break
    return {
        "seed": seed,
        "score_blue": env.state.score[Team.BLUE],
        "score_red": env.state.score[Team.RED],
        "steps": env.state.step_count,
        "touches": events.get("touch", 0),
        "penalties": events.get("penalty", 0),
        "restart_retakes": events.get("restart_retake", 0),
        "events": events, "event_details": event_details,
        "penalties_by_team": penalties_by_team,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--max_scoreless_rate", type=float, default=.30)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "teacher_validation.json")
    args = parser.parse_args()
    report = {"criteria": vars(args), "matchups": {}}
    qualified = True
    for matchup_index, (left, right) in enumerate(MATCHUPS):
        matches = [run_match(left, right, 1000 + matchup_index * 100 + seed, args.duration)
                   for seed in range(args.seeds)]
        goals = sum(item["score_blue"] + item["score_red"] for item in matches)
        scoreless = sum(item["score_blue"] + item["score_red"] == 0 for item in matches)
        penalties = sum(item["penalties"] for item in matches)
        teacher_penalties = sum(
            (item["penalties_by_team"]["blue"] if left != "stationary" else 0)
            + (item["penalties_by_team"]["red"] if right != "stationary" else 0)
            for item in matches
        )
        retakes = sum(item["restart_retakes"] for item in matches)
        touches = sum(item["touches"] for item in matches)
        passed = (goals > 0 and scoreless / args.seeds <= args.max_scoreless_rate
                  and teacher_penalties == 0 and retakes == 0 and touches > 0)
        qualified &= passed
        name = "%s_vs_%s" % (left, right)
        report["matchups"][name] = {
            "passed": passed, "goals": goals, "scoreless": scoreless,
            "scoreless_rate": scoreless / args.seeds, "touches": touches,
            "penalties": penalties, "teacher_penalties": teacher_penalties,
            "restart_retakes": retakes, "matches": matches,
        }
        print("TEACHER %-24s pass=%s goals=%d scoreless=%d/%d touches=%d teacher_penalties=%d retakes=%d" %
              (name, passed, goals, scoreless, args.seeds, touches, teacher_penalties, retakes), flush=True)
    report["qualified"] = qualified
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, default=str)
    print("TEACHER QUALIFIED=%s report=%s" % (qualified, args.output), flush=True)
    raise SystemExit(0 if qualified else 2)


if __name__ == "__main__":
    main()
