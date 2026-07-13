#!/usr/bin/env python3
"""Run a full deterministic heuristic-vs-heuristic match."""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robocup3v3.env import Robocup3v3Env
from robocup3v3.opponents import HeuristicOpponent
from robocup3v3.types import Team


def main():
    env = Robocup3v3Env()
    env.reset(seed=4)
    blue = HeuristicOpponent(Team.BLUE, env.config)
    red = HeuristicOpponent(Team.RED, env.config)
    counts = collections.Counter()
    while True:
        actions = blue.actions(env.state)
        actions.update(red.actions(env.state))
        _, _, terminated, truncated, infos = env.step(actions)
        counts.update(event["kind"] for event in infos["blue_1"]["events"])
        if any(terminated.values()) or any(truncated.values()):
            break
    print(
        "finished score=%d-%d steps=%d reason=%s events=%s"
        % (
            env.state.score[Team.BLUE],
            env.state.score[Team.RED],
            env.state.step_count,
            env.state.termination_reason,
            dict(sorted(counts.items())),
        )
    )


if __name__ == "__main__":
    main()

