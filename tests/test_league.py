import json
from pathlib import Path

import numpy as np

from robocup3v3.league import LeaguePool


def test_league_filters_mastered_weak_non_anchor(tmpdir):
    tmp_path = Path(str(tmpdir))
    pool = LeaguePool(tmp_path / "league", seed=1)
    data = pool.load()
    novice = next(entry for entry in data["entries"] if entry["id"] == "bt_novice")
    novice.update(games=20, losses_vs_learner=18, draws_vs_learner=0,
                  wins_vs_learner=2, rating=800.0)
    data["learner_rating"] = 1300.0
    pool.save(data)
    eligible = {entry["id"] for entry in LeaguePool.eligible_entries(pool.load())}
    assert "bt_novice" not in eligible
    assert "bt_standard" in eligible
    assert "bt_expert" in eligible


def test_league_registers_and_samples_actor_snapshot(tmpdir):
    tmp_path = Path(str(tmpdir))
    directory = tmp_path / "league"
    pool = LeaguePool(directory, seed=2)
    snapshot = directory / "snapshots" / "actor.pt"
    snapshot.write_bytes(b"test")
    entry_id = pool.register_snapshot(snapshot, 500)
    assert entry_id == "actor_000000000500"
    data = pool.load()
    for entry in data["entries"]:
        entry["active"] = entry["id"] == entry_id
    pool.save(data)
    sampled = pool.sample(np.random.RandomState(3))
    assert sampled["id"] == entry_id
    assert sampled["kind"] == "actor"
