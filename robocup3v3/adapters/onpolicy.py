"""SMAC-style adapter for the supplied marlbenchmark/on-policy runner."""

from __future__ import annotations

from typing import List

import numpy as np

from ..actions import N_ACTIONS
from ..config import EnvConfig, RewardConfig
from ..env import Robocup3v3Env
from ..observations import global_state_size, hybrid_global_state_size, observation_size
from ..opponents import RuleTreeOpponent
from ..league import LeaguePool
from ..spaces import Box, Discrete
from ..types import Penalty, Team


class OnPolicyTeamEnv:
    """Expose three learning agents against a scripted opponent.

    API matches ``ShareSubprocVecEnv`` + ``SMACRunner``:
    reset -> obs, share_obs, available_actions
    step  -> obs, share_obs, rewards, dones, infos, available_actions
    """

    def __init__(self, config=None, reward_config=None, controlled_team="blue", curriculum=True,
                 birdview=False, birdview_height=32, birdview_width=48,
                 league_dir=None, league_hidden_size=512, league_layer_n=3,
                 reward_heads=4, curriculum_initial_stage=0,
                 opponent_count_mode="curriculum",
                 opponent_count_probs="0.0,0.20,0.30,0.50",
                 fixed_opponent_count=3, opponent_difficulty_probs=""):
        self.config = config or EnvConfig()
        self.team = Team(controlled_team)
        self.core = Robocup3v3Env(self.config, reward_config or RewardConfig())
        self.opponent = RuleTreeOpponent(self.team.opponent, self.config)
        self.league_dir = league_dir
        self.league_hidden_size = int(league_hidden_size)
        self.league_layer_n = int(league_layer_n)
        self.reward_heads = int(reward_heads)
        if self.reward_heads not in (1, 4):
            raise ValueError("reward_heads must be 1 or 4")
        self._actor_opponent_cache = {}
        self.opponent_entry = None
        self.agent_names = tuple("%s_%d" % (self.team.value, i) for i in (1, 2, 3))
        self.num_agents = 3
        self.birdview = bool(birdview)
        self.birdview_height = int(birdview_height)
        self.birdview_width = int(birdview_width)
        self.observation_space = [
            Box(low=-10.0, high=10.0, shape=(observation_size(self.config),), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            Box(low=-10.0, high=10.0,
                shape=((hybrid_global_state_size(self.birdview_height, self.birdview_width)
                        if self.birdview else global_state_size(self.config)),), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.action_space = [Discrete(N_ACTIONS) for _ in range(self.num_agents)]
        self._next_seed = 0
        self.opponent_count_mode = str(opponent_count_mode)
        if self.opponent_count_mode not in ("curriculum", "mixed", "fixed"):
            raise ValueError("opponent_count_mode must be curriculum, mixed, or fixed")
        if isinstance(opponent_count_probs, str):
            opponent_count_probs = [float(value) for value in opponent_count_probs.split(",")]
        self.opponent_count_probs = np.asarray(opponent_count_probs, dtype=np.float64)
        if self.opponent_count_probs.shape != (4,) or np.any(self.opponent_count_probs < 0):
            raise ValueError("opponent_count_probs must contain four non-negative values")
        probability_sum = self.opponent_count_probs.sum()
        if probability_sum <= 0:
            raise ValueError("opponent_count_probs must have positive sum")
        self.opponent_count_probs /= probability_sum
        self.fixed_opponent_count = int(fixed_opponent_count)
        if self.fixed_opponent_count not in (0, 1, 2, 3):
            raise ValueError("fixed_opponent_count must be 0..3")
        self.opponent_difficulty_probs = None
        if opponent_difficulty_probs:
            if isinstance(opponent_difficulty_probs, str):
                opponent_difficulty_probs = [float(value) for value in opponent_difficulty_probs.split(",")]
            values = np.asarray(opponent_difficulty_probs, dtype=np.float64)
            if values.shape != (4,) or np.any(values < 0) or values.sum() <= 0:
                raise ValueError("opponent_difficulty_probs must contain four non-negative values")
            self.opponent_difficulty_probs = values / values.sum()
        self.curriculum = bool(curriculum and self.opponent_count_mode == "curriculum")
        self.curriculum_stage = int(curriculum_initial_stage) if self.curriculum else 3
        self.current_opponent_count = (self.curriculum_stage if self.curriculum
                                       else self.fixed_opponent_count)
        self.curriculum_goals = 0
        self.stage_goal_thresholds = (20, 50, 90)
        self.stage_results = []
        self.stage_gate_window = 20
        self.reward_totals = {}
        self.reward_abs_totals = {}
        self.matches = self.wins = self.draws = self.losses = 0
        self.goals_for = self.goals_against = 0
        self.opponent_difficulty = "standard"
        self.difficulty_counts = {name: 0 for name in RuleTreeOpponent.DIFFICULTIES}
        self.difficulty_counts["actor"] = 0
        self.opponent_id_counts = {}

    def seed(self, seed=None):
        self._next_seed = 0 if seed is None else int(seed)
        self.core.seed(self._next_seed)
        return [self._next_seed]

    def reset(self):
        observations, _ = self.core.reset(seed=self._next_seed)
        # Advance between episodes; otherwise every auto-reset reproduces the
        # same initial positions and the same sampled opponent difficulty.
        self._next_seed += 1
        self._sample_opponent()
        self._sample_opponent_count()
        self._apply_curriculum_stage()
        observations = self.core._observations()
        return self._arrays(observations)

    def step(self, actions):
        action_map = {}
        flat = np.asarray(actions).reshape(self.num_agents, -1)
        for idx, name in enumerate(self.agent_names):
            action_map[name] = int(flat[idx, 0])
        action_map.update(self.opponent.actions(self.core.state))
        observations, rewards, terminations, truncations, infos = self.core.step(action_map)
        components = infos[self.agent_names[0]].get("reward_components", {})
        for key, value in components.items():
            self.reward_totals[key] = self.reward_totals.get(key, 0.0) + float(value)
            self.reward_abs_totals[key] = self.reward_abs_totals.get(key, 0.0) + abs(float(value))
        obs, share_obs, available = self._arrays(observations)
        groups = self._reward_groups(components)
        if self.reward_heads == 1:
            groups = (sum(groups),)
        reward_array = np.repeat(np.asarray(groups, dtype=np.float32)[None, :], self.num_agents, axis=0)
        done = bool(any(terminations.values()) or any(truncations.values()))
        if done:
            own = self.core.state.score[self.team]
            other = self.core.state.score[self.team.opponent]
            self.matches += 1
            self.goals_for += own
            self.goals_against += other
            self.curriculum_goals += own
            self.stage_results.append((int(own), int(other)))
            self.stage_results = self.stage_results[-self.stage_gate_window:]
            if own > other:
                self.wins += 1
            elif own == other:
                self.draws += 1
            else:
                self.losses += 1
            self._promote_curriculum()
            print(
                "MATCH result=%s score=%d-%d W/D/L=%d/%d/%d win_rate=%.3f goals=%d:%d"
                % ("win" if own > other else "draw" if own == other else "loss", own, other,
                   self.wins, self.draws, self.losses, self.wins / self.matches,
                   self.goals_for, self.goals_against),
                flush=True,
            )
        dones = np.full(self.num_agents, done, dtype=np.bool_)
        info_list = []
        for name in self.agent_names:
            item = dict(infos[name])
            item["bad_transition"] = bool(truncations[name])
            item["won"] = self.core.state.score[self.team] > self.core.state.score[self.team.opponent]
            item.update({
                "battles_game": self.matches,
                "battles_won": self.wins,
                "matches": self.matches,
                "draws": self.draws,
                "losses": self.losses,
                "goals_for": self.goals_for,
                "goals_against": self.goals_against,
                "goal_difference": self.goals_for - self.goals_against,
                "win_rate": self.wins / self.matches if self.matches else 0.0,
                "curriculum_stage": self.curriculum_stage,
                "curriculum_opponents": self._opponent_count(),
                **{"curriculum_gate_" + key: value for key, value in self._curriculum_gate_stats().items()},
                "opponent_difficulty": (RuleTreeOpponent.DIFFICULTIES.index(self.opponent_difficulty)
                                        if self.opponent_difficulty in RuleTreeOpponent.DIFFICULTIES else -1),
                "league_opponent_id": self.opponent_entry["id"] if self.opponent_entry else "",
                "league_opponent_kind": self.opponent_entry["kind"] if self.opponent_entry else "behavior_tree",
                "league_opponent_rating": float(self.opponent_entry["rating"]) if self.opponent_entry else 0.0,
            })
            if done and self.opponent_entry:
                item["league_match_result"] = {
                    "opponent_id": self.opponent_entry["id"],
                    "result": "win" if own > other else "draw" if own == other else "loss",
                    "goals_for": int(own), "goals_against": int(other),
                }
            item.update({"opponent_%s_matches" % key: value
                         for key, value in self.difficulty_counts.items()})
            item["opponent_id_sample_counts"] = dict(self.opponent_id_counts)
            item.update({"reward_total_" + key: value for key, value in self.reward_totals.items()})
            item.update({"reward_abs_total_" + key: value for key, value in self.reward_abs_totals.items()})
            info_list.append(item)
        return obs, share_obs, reward_array, dones, info_list, available

    @staticmethod
    def _reward_groups(components):
        outcome = sum(components.get(key, 0.0) for key in ("goal", "concede"))
        attack = sum(components.get(key, 0.0) for key in (
            "ball_progress", "checkpoint", "approach_ball", "shot_on_target", "bad_shot",
        ))
        teamwork = sum(components.get(key, 0.0) for key in (
            "possession", "successful_pass", "pressured_pass", "forward_pass", "turnover", "clustering",
            "receiver_facing",
            "keeper_positioning", "keeper_save",
        ))
        safety = sum(components.get(key, 0.0) for key in (
            "penalty", "illegal_action", "out_of_bounds", "time",
        ))
        return outcome, attack, teamwork, safety

    def _opponent_count(self):
        return self.current_opponent_count

    def _sample_opponent_count(self):
        if self.opponent_count_mode == "curriculum":
            self.current_opponent_count = self.curriculum_stage
        elif self.opponent_count_mode == "mixed":
            self.current_opponent_count = int(
                self.core.np_random.choice((0, 1, 2, 3), p=self.opponent_count_probs))
        else:
            self.current_opponent_count = self.fixed_opponent_count

    def _sample_opponent(self):
        if self.league_dir:
            pool = LeaguePool(self.league_dir, seed=self._next_seed)
            entry = pool.sample(self.core.np_random)
            self.opponent_entry = entry
            if entry["kind"] == "behavior_tree":
                difficulty = entry["id"].replace("bt_", "")
                if not isinstance(self.opponent, RuleTreeOpponent):
                    self.opponent = RuleTreeOpponent(self.team.opponent, self.config)
                self.opponent.set_difficulty(difficulty)
                self.opponent_difficulty = difficulty
            else:
                path = entry["path"]
                if path not in self._actor_opponent_cache:
                    from ..learned_opponent import FrozenActorOpponent
                    self._actor_opponent_cache[path] = FrozenActorOpponent(
                        self.team.opponent, self.config, path,
                        self.league_hidden_size, self.league_layer_n)
                self.opponent = self._actor_opponent_cache[path]
                self.opponent_difficulty = "actor"
            self.difficulty_counts[self.opponent_difficulty] += 1
            self.opponent_id_counts[entry["id"]] = self.opponent_id_counts.get(entry["id"], 0) + 1
            return
        self.opponent_entry = None
        self._sample_opponent_difficulty()

    def _sample_opponent_difficulty(self):
        # Adaptive mixture inspired by TiZero: the number of active opponents
        # and their tactical strength both increase with curriculum stage.
        mixtures = {
            0: (("stationary", .70), ("novice", .30)),
            1: (("stationary", .20), ("novice", .55), ("standard", .25)),
            2: (("novice", .20), ("standard", .55), ("expert", .25)),
            3: (("novice", .05), ("standard", .40), ("expert", .55)),
        }
        if self.opponent_difficulty_probs is not None:
            names = RuleTreeOpponent.DIFFICULTIES
            probabilities = self.opponent_difficulty_probs
        else:
            names, probabilities = zip(*mixtures[self.curriculum_stage])
        self.opponent_difficulty = str(self.core.np_random.choice(names, p=probabilities))
        if not isinstance(self.opponent, RuleTreeOpponent):
            self.opponent = RuleTreeOpponent(self.team.opponent, self.config)
        self.opponent.set_difficulty(self.opponent_difficulty)
        self.difficulty_counts[self.opponent_difficulty] += 1
        entry_id = "bt_" + self.opponent_difficulty
        self.opponent_id_counts[entry_id] = self.opponent_id_counts.get(entry_id, 0) + 1

    def _apply_curriculum_stage(self):
        active_ids = {
            0: (),
            # 3v1 must contain a field chaser. The former (3,) mapping trained
            # only against a stationary keeper and did not transfer to 3v3.
            1: (1,),
            2: (1, 3),
            3: (1, 2, 3),
        }[self._opponent_count()]
        for robot in self.core.state.team_robots(self.team.opponent):
            if robot.player_id in active_ids:
                robot.penalty = Penalty.NONE
                robot.penalty_remaining = 0.0
            else:
                robot.penalty = Penalty.SENT_OFF
                robot.penalty_remaining = 1.0e9
                robot.pose.y = self.config.field_width / 2.0 + 2.0 + robot.player_id

    def _promote_curriculum(self):
        if not self.curriculum or self.curriculum_stage >= 3:
            return
        threshold = self.stage_goal_thresholds[self.curriculum_stage]
        stats = self._curriculum_gate_stats()
        gates = (
            dict(min_matches=5, min_goals_per_match=1.0, min_nonloss_rate=0.80, max_scoreless_rate=0.20),
            dict(min_matches=10, min_goals_per_match=0.50, min_nonloss_rate=0.40, max_scoreless_rate=0.50),
            dict(min_matches=15, min_goals_per_match=0.35, min_nonloss_rate=0.30, max_scoreless_rate=0.65),
        )[self.curriculum_stage]
        performance_pass = (
            stats["matches"] >= gates["min_matches"] and
            stats["goals_per_match"] >= gates["min_goals_per_match"] and
            stats["nonloss_rate"] >= gates["min_nonloss_rate"] and
            stats["scoreless_rate"] <= gates["max_scoreless_rate"]
        )
        if self.curriculum_goals >= threshold and performance_pass:
            old = self.curriculum_stage
            self.curriculum_stage += 1
            print(
                "CURRICULUM promote stage=%d->%d scenario=3v%d goals=%d threshold=%d"
                % (old, self.curriculum_stage, self._opponent_count(), self.curriculum_goals, threshold),
                flush=True,
            )
            self.stage_results = []

    def _curriculum_gate_stats(self):
        matches = len(self.stage_results)
        if not matches:
            return {"matches": 0, "goals_per_match": 0.0,
                    "nonloss_rate": 0.0, "scoreless_rate": 1.0}
        goals = sum(own for own, _ in self.stage_results)
        nonlosses = sum(own >= other for own, other in self.stage_results)
        scoreless = sum(own == 0 for own, _ in self.stage_results)
        return {"matches": matches, "goals_per_match": goals / matches,
                "nonloss_rate": nonlosses / matches, "scoreless_rate": scoreless / matches}

    def _arrays(self, observations):
        obs = np.stack([observations[name] for name in self.agent_names]).astype(np.float32)
        state = self.core.state_for_team(self.team.value, self.birdview,
                                         self.birdview_height, self.birdview_width).astype(np.float32)
        share_obs = np.repeat(state[None, :], self.num_agents, axis=0)
        available = np.stack([self.core.action_mask(name) for name in self.agent_names]).astype(np.float32)
        return obs, share_obs, available

    def close(self):
        self.core.close()

    def render(self, mode="human"):
        return self.core.render()
