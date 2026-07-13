#!/usr/bin/env python3
"""Fast environment and on-policy contract smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robocup3v3.adapters.onpolicy import OnPolicyTeamEnv


def main():
    env = OnPolicyTeamEnv()
    env.seed(7)
    obs, state, available = env.reset()
    total = 0.0
    for _ in range(2000):
        actions = np.asarray([
            [np.random.choice(np.flatnonzero(available[i]))]
            for i in range(3)
        ])
        obs, state, rewards, dones, infos, available = env.step(actions)
        total += float(rewards.mean())
        assert np.isfinite(obs).all() and np.isfinite(state).all()
        if dones.all():
            obs, state, available = env.reset()
    print("smoke OK obs=%s state=%s actions=%s reward=%.3f" % (
        obs.shape, state.shape, available.shape, total,
    ))


if __name__ == "__main__":
    main()

