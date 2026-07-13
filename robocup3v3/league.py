"""File-backed opponent league with Elo and competence-aware sampling."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


BT_RATINGS = {"novice": 900.0, "standard": 1200.0, "expert": 1500.0}


def _entry(entry_id, kind, rating, path=None, anchor=False):
    return {
        "id": entry_id, "kind": kind, "rating": float(rating), "path": path,
        "anchor": bool(anchor), "active": True, "games": 0,
        "wins_vs_learner": 0, "draws_vs_learner": 0, "losses_vs_learner": 0,
        "goals_for": 0, "goals_against": 0, "created_at": time.time(),
    }


class LeaguePool:
    def __init__(self, directory, seed=1):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "snapshots").mkdir(exist_ok=True)
        self.manifest_path = self.directory / "league.json"
        self.rng = np.random.RandomState(seed)
        if not self.manifest_path.exists():
            data = {
                "version": 1, "learner_rating": 1200.0, "snapshot_count": 0,
                "entries": [
                    _entry("bt_novice", "behavior_tree", BT_RATINGS["novice"]),
                    _entry("bt_standard", "behavior_tree", BT_RATINGS["standard"], anchor=True),
                    _entry("bt_expert", "behavior_tree", BT_RATINGS["expert"], anchor=True),
                ],
            }
            self.save(data)

    def load(self):
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def save(self, data):
        temporary = self.manifest_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
        os.replace(str(temporary), str(self.manifest_path))

    def register_snapshot(self, path, steps):
        data = self.load()
        entry_id = "actor_%012d" % int(steps)
        if any(item["id"] == entry_id for item in data["entries"]):
            return entry_id
        data["entries"].append(_entry(
            entry_id, "actor", data["learner_rating"], path=str(Path(path).resolve())))
        data["snapshot_count"] = int(data.get("snapshot_count", 0)) + 1
        self.save(data)
        return entry_id

    def prune_actor_entries(self, maximum):
        data = self.load()
        actors = sorted((e for e in data["entries"] if e["kind"] == "actor"),
                        key=lambda e: e.get("created_at", 0))
        if len(actors) <= maximum:
            return
        # Preserve the strongest historical actor plus the newest generation.
        strongest = max(actors, key=lambda e: e["rating"])["id"]
        removable = [e for e in actors[:-1] if e["id"] != strongest]
        for entry in removable[:max(0, len(actors) - maximum)]:
            entry["active"] = False
        self.save(data)

    @staticmethod
    def _learner_win_rate(entry):
        games = max(int(entry.get("games", 0)), 1)
        # Entry losses are learner wins; draws count half.
        return (entry.get("losses_vs_learner", 0) + .5 * entry.get("draws_vs_learner", 0)) / games

    @classmethod
    def eligible_entries(cls, data):
        learner = float(data.get("learner_rating", 1200.0))
        result = []
        for entry in data["entries"]:
            if not entry.get("active", True):
                continue
            weak = (entry.get("games", 0) >= 20
                    and cls._learner_win_rate(entry) >= .85
                    and float(entry["rating"]) < learner - 250.0)
            if weak and not entry.get("anchor", False):
                continue
            result.append(entry)
        return result

    @classmethod
    def sample_from_data(cls, data, rng):
        entries = cls.eligible_entries(data)
        if not entries:
            entries = [item for item in data["entries"] if item.get("anchor", False)]
        learner = float(data.get("learner_rating", 1200.0))
        actors = [entry for entry in entries if entry["kind"] == "actor"]
        if actors:
            novice = [entry for entry in entries if entry["id"] == "bt_novice"]
            anchors = [entry for entry in entries if entry.get("anchor", False)]
            draw = rng.rand()
            # Controlled self-play: preserve the currently hardest tactical
            # exploit (novice direct press), stable BT anchors, and historical
            # policies. Pure self-play is intentionally avoided.
            if draw < .30 and novice:
                return dict(novice[int(rng.randint(len(novice)))])
            if draw < .60:
                scores = np.asarray([
                    np.exp(-abs(float(e["rating"]) - learner) / 120.0) for e in actors])
                scores /= scores.sum()
                return dict(actors[int(rng.choice(len(actors), p=scores))])
            if draw < .90 and anchors:
                return dict(anchors[int(rng.randint(len(anchors)))])
            return dict(entries[int(rng.randint(len(entries)))])
        draw = rng.rand()
        if draw < .55:       # closest competitive neighbours
            scores = np.asarray([np.exp(-abs(float(e["rating"]) - learner) / 120.0) for e in entries])
        elif draw < .75:     # slightly stronger opponents
            scores = np.asarray([
                np.exp(-abs(float(e["rating"]) - (learner + 150.0)) / 140.0)
                if float(e["rating"]) >= learner - 30.0 else .01 for e in entries])
        elif draw < .90:     # permanent behavior-tree anchors
            scores = np.asarray([1.0 if e.get("anchor", False) else .01 for e in entries])
        else:                # diversity/history protection
            scores = np.ones(len(entries), dtype=np.float64)
        scores = np.maximum(scores, 1e-9)
        scores /= scores.sum()
        return dict(entries[int(rng.choice(len(entries), p=scores))])

    def sample(self, rng=None):
        return self.sample_from_data(self.load(), rng or self.rng)

    def update_results(self, results, k_factor=24.0):
        if not results:
            return self.load()
        data = self.load()
        by_id = {entry["id"]: entry for entry in data["entries"]}
        learner_rating = float(data.get("learner_rating", 1200.0))
        for result in results:
            entry = by_id.get(result.get("opponent_id"))
            if entry is None:
                continue
            opponent_rating = float(entry["rating"])
            learner_score = 1.0 if result["result"] == "win" else .5 if result["result"] == "draw" else 0.0
            expected = 1.0 / (1.0 + 10.0 ** ((opponent_rating - learner_rating) / 400.0))
            delta = k_factor * (learner_score - expected)
            learner_rating += delta
            entry["rating"] = opponent_rating - delta
            entry["games"] = int(entry.get("games", 0)) + 1
            if learner_score == 1.0:
                entry["losses_vs_learner"] = int(entry.get("losses_vs_learner", 0)) + 1
            elif learner_score == .5:
                entry["draws_vs_learner"] = int(entry.get("draws_vs_learner", 0)) + 1
            else:
                entry["wins_vs_learner"] = int(entry.get("wins_vs_learner", 0)) + 1
            entry["goals_for"] = int(entry.get("goals_for", 0)) + int(result.get("goals_against", 0))
            entry["goals_against"] = int(entry.get("goals_against", 0)) + int(result.get("goals_for", 0))
        data["learner_rating"] = learner_rating
        self.save(data)
        return data

    def update_pair_result(self, first_id, second_id, first_score, k_factor=24.0):
        data = self.load()
        by_id = {entry["id"]: entry for entry in data["entries"]}
        first, second = by_id[first_id], by_id[second_id]
        expected = 1.0 / (1.0 + 10.0 ** ((second["rating"] - first["rating"]) / 400.0))
        delta = k_factor * (float(first_score) - expected)
        first["rating"] += delta
        second["rating"] -= delta
        first["swiss_games"] = int(first.get("swiss_games", 0)) + 1
        second["swiss_games"] = int(second.get("swiss_games", 0)) + 1
        self.save(data)
