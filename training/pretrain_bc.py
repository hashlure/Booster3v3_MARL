#!/usr/bin/env python3
"""Behavior-clone RuleTreeOpponent into the shared MAPPO Actor."""

from __future__ import annotations

import argparse
import collections
import time
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "on-policy"))

import numpy as np
import torch
import torch.nn.functional as F

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from robocup3v3.actions import N_ACTIONS, available_actions, decode_action
from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.observations import local_observation, observation_size
from robocup3v3.opponents import RuleTreeOpponent
from robocup3v3.spaces import Box, Discrete
from robocup3v3.types import Team
from training.train_mappo import parse_args as parse_mappo_args


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layer_N", type=int, default=3)
    parser.add_argument("--dagger_from", type=Path, default=None,
                        help="rejected/student actor checkpoint used to collect corrective states")
    parser.add_argument("--dagger_samples", type=int, default=250000)
    parser.add_argument("--dagger_beta", type=float, default=0.20,
                        help="probability of executing teacher action during DAgger rollout")
    parser.add_argument("--dagger_opponents", type=str, default="novice,novice,standard",
                        help="comma-separated opponent curriculum used during DAgger collection")
    parser.add_argument("--eval_matches", type=int, default=10)
    parser.add_argument("--min_stationary_win_rate", type=float, default=0.80)
    parser.add_argument("--min_stationary_goals_per_match", type=float, default=1.0)
    parser.add_argument("--min_novice_goals_per_match", type=float, default=0.50)
    parser.add_argument("--max_novice_scoreless_rate", type=float, default=0.70)
    parser.add_argument("--min_standard_goals_per_match", type=float, default=0.30)
    parser.add_argument("--max_standard_scoreless_rate", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "bc" / "rule_tree_31" / "run1")
    return parser.parse_args()


def collect(sample_count, seed):
    config = EnvConfig(
        ready_duration_sec=0.2,
        set_duration_sec=0.1,
        match_duration_sec=60.0,
        max_episode_steps=900,
        randomize_reset=True,
        action_noise=0.02,
    )
    env = Robocup3v3Env(config)
    teachers = {team: RuleTreeOpponent(team, config) for team in (Team.BLUE, Team.RED)}
    rng = np.random.RandomState(seed)
    # Deliberately include easy scoring scenarios as well as competitive ones.
    # Competitive scoring trajectories dominate; stationary is retained only
    # to prevent forgetting the basic approach/shot chain.
    matchups = (
        ("expert", "stationary"), ("standard", "stationary"),
        ("expert", "novice"), ("expert", "novice"), ("expert", "novice"),
        ("standard", "novice"), ("standard", "novice"),
        ("expert", "standard"), ("expert", "standard"), ("expert", "standard"),
        ("standard", "expert"), ("standard", "expert"),
    )
    observations, masks, labels, priorities = [], [], [], []
    episode = successful_episodes = total_goals = 0
    collection_started = time.time()
    target_candidates = sample_count * 2
    goal_episode_target = max(10, sample_count // 30000)
    episode_records = []
    blue_level, red_level = matchups[rng.randint(len(matchups))]
    teachers[Team.BLUE].set_difficulty(blue_level)
    teachers[Team.RED].set_difficulty(red_level)
    env.reset(seed=seed + episode)
    while (len(labels) < target_candidates or successful_episodes < goal_episode_target) and episode < 1000:
        action_map = {}
        for team, teacher in teachers.items():
            teacher_labels = teacher.teacher_action_ids(env.state)
            action_map.update(teacher.actions(env.state))
            if teacher.difficulty == "stationary" or env.state.game_state.value != "PLAYING" or env.state.stopped:
                continue
            for robot in env.state.team_robots(team):
                label = teacher_labels[robot.name]
                mask = available_actions(robot, env.state, config)
                # Invalid teacher labels indicate a teacher/environment contract
                # bug. Dropping them is safer than silently converting them to
                # hold and poisoning the dataset.
                if mask[label] <= 0.0:
                    continue
                episode_records.append((
                    local_observation(env.state, robot, config), mask, label, team,
                ))
        _, _, terminated, truncated, _ = env.step(action_map)
        if any(terminated.values()) or any(truncated.values()):
            blue_goals, red_goals = env.state.score[Team.BLUE], env.state.score[Team.RED]
            goals = blue_goals + red_goals
            total_goals += goals
            successful_episodes += int(goals > 0)
            for observation, mask, label, team in episode_records:
                team_scored = blue_goals > 0 if team is Team.BLUE else red_goals > 0
                observations.append(observation)
                masks.append(mask)
                labels.append(label)
                priorities.append(5.0 if team_scored else 1.0)
            episode += 1
            elapsed = max(time.time() - collection_started, 1e-6)
            if episode == 1 or episode % 5 == 0 or goals > 0:
                print(
                    "BC COLLECT episode=%d candidates=%d/%d scoring_episodes=%d/%d "
                    "goals=%d last_match=%s_vs_%s score=%d:%d rate=%.0f samples/s elapsed=%.1fs"
                    % (episode, len(labels), target_candidates, successful_episodes,
                       goal_episode_target, total_goals, blue_level, red_level,
                       blue_goals, red_goals, len(labels) / elapsed, elapsed),
                    flush=True,
                )
            episode_records = []
            blue_level, red_level = matchups[rng.randint(len(matchups))]
            teachers[Team.BLUE].set_difficulty(blue_level)
            teachers[Team.RED].set_difficulty(red_level)
            env.reset(seed=seed + episode)
    if not labels:
        raise RuntimeError("behavior-tree collection produced no valid PLAYING samples")

    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=N_ACTIONS).astype(np.float64)
    weights = np.asarray(priorities, dtype=np.float64)
    # Counter class imbalance without making rare labels overwhelmingly large.
    weights *= 1.0 / np.sqrt(np.maximum(counts[labels_array], 1.0))
    weights *= np.where(np.isin(labels_array, (13, 14, 15)), 2.5, 1.0)
    # A shot is a one-tick event while positioning lasts hundreds of ticks;
    # strong oversampling is required for shooting to survive classification.
    weights *= np.where(np.isin(labels_array, (16, 17, 18, 22)), 40.0, 1.0)
    weights *= np.where(labels_array == 0, 0.15, 1.0)
    weights /= weights.sum()
    chosen = rng.choice(len(labels_array), size=sample_count, replace=len(labels_array) < sample_count, p=weights)
    sampled_labels = labels_array[chosen]
    distribution = collections.Counter(sampled_labels.tolist())
    print("BC collection episodes=%d scoring_episodes=%d goals=%d candidates=%d" %
          (episode, successful_episodes, total_goals, len(labels_array)), flush=True)
    print("BC sampled_action_distribution=%s" % dict(sorted(distribution.items())), flush=True)
    return (
        np.asarray(observations, dtype=np.float32)[chosen],
        np.asarray(masks, dtype=np.float32)[chosen],
        sampled_labels,
    )


def collect_dagger(actor, device, sample_count, seed, beta, opponent_names):
    """Label states visited by the student with an expert behavior tree."""
    if sample_count <= 0:
        empty_obs = np.empty((0, observation_size()), dtype=np.float32)
        return empty_obs, np.empty((0, N_ACTIONS), dtype=np.float32), np.empty(0, dtype=np.int64)
    config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                       match_duration_sec=60.0, max_episode_steps=900,
                       randomize_reset=True, action_noise=.02)
    env = Robocup3v3Env(config)
    teacher = RuleTreeOpponent(Team.BLUE, config, difficulty="expert")
    opponents = tuple(name.strip() for name in opponent_names.split(",") if name.strip())
    invalid = set(opponents) - set(RuleTreeOpponent.DIFFICULTIES)
    if not opponents or invalid:
        raise ValueError("invalid --dagger_opponents: %s" % opponent_names)
    rng = np.random.RandomState(seed + 31000)
    observations, masks, labels, priorities = [], [], [], []
    episode = goals = 0
    actor.eval()
    while len(labels) < sample_count * 2 and episode < 1000:
        difficulty = opponents[episode % len(opponents)]
        opponent = RuleTreeOpponent(Team.RED, config, difficulty=difficulty)
        env.reset(seed=seed + 32000 + episode)
        episode_records = []
        while True:
            action_map = opponent.actions(env.state)
            teacher_ids = teacher.teacher_action_ids(env.state)
            teacher_actions = teacher.actions(env.state)
            # CTDE team behavior must stay coherent: switch the entire blue
            # team per timestep, never independently mix teacher/student robots.
            teacher_controls_team = rng.rand() < beta
            playing = env.state.game_state.value == "PLAYING" and not env.state.stopped
            for robot in env.state.team_robots(Team.BLUE):
                mask = available_actions(robot, env.state, config)
                label = teacher_ids[robot.name]
                if playing and mask[label] > 0.0:
                    episode_records.append((local_observation(env.state, robot, config), mask, label))
                if teacher_controls_team:
                    action_map[robot.name] = teacher_actions[robot.name]
                else:
                    obs = torch.as_tensor(local_observation(env.state, robot, config)[None], device=device)
                    with torch.no_grad():
                        logits = actor.act.action_out.linear(actor.base(obs))
                        logits = logits.masked_fill(torch.as_tensor(mask[None], device=device) <= 0, -1.0e9)
                        action_id = int(logits.argmax(dim=-1).item())
                    action_map[robot.name] = decode_action(action_id, robot, env.state, config)
            _, _, terminated, truncated, _ = env.step(action_map)
            if any(terminated.values()) or any(truncated.values()):
                break
        own = env.state.score[Team.BLUE]
        goals += own
        # Successful competitive episodes receive extra priority, but failed
        # student states remain present because their corrective labels are the
        # main reason for DAgger.
        for observation, mask, label in episode_records:
            observations.append(observation)
            masks.append(mask)
            labels.append(label)
            priority = 3.0 if own > 0 else 1.0
            if label in (13, 14, 15):
                priority *= 3.0
            if label in (16, 17, 18, 22):
                priority *= 30.0
            priorities.append(priority)
        episode += 1
        if episode == 1 or episode % 5 == 0:
            print("DAGGER COLLECT episode=%d candidates=%d/%d goals=%d opponent=%s beta=%.2f" %
                  (episode, len(labels), sample_count * 2, goals, difficulty, beta), flush=True)
    actor.train()
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=N_ACTIONS).astype(np.float64)
    weights = np.asarray(priorities, dtype=np.float64)
    weights *= 1.0 / np.sqrt(np.maximum(counts[labels_array], 1.0))
    weights *= np.where(labels_array == 0, .10, 1.0)
    weights /= weights.sum()
    chosen = rng.choice(len(labels_array), size=sample_count,
                        replace=len(labels_array) < sample_count, p=weights)
    print("DAGGER collection episodes=%d goals=%d sampled=%d distribution=%s" %
          (episode, goals, sample_count,
           dict(sorted(collections.Counter(labels_array[chosen].tolist()).items()))), flush=True)
    return (np.asarray(observations, dtype=np.float32)[chosen],
            np.asarray(masks, dtype=np.float32)[chosen], labels_array[chosen])


def evaluate_actor(actor, device, seed, matches=3):
    """Report goals after BC; accuracy alone is not a football metric."""
    results = {}
    actor.eval()
    for difficulty in ("stationary", "novice", "standard"):
        goals_for = goals_against = wins = draws = scoreless = 0
        for match in range(matches):
            config = EnvConfig(ready_duration_sec=.2, set_duration_sec=.1,
                               match_duration_sec=60.0, max_episode_steps=900,
                               randomize_reset=True, action_noise=0.0)
            env = Robocup3v3Env(config)
            env.reset(seed=seed + 10000 + 1000 * len(results) + match)
            opponent = RuleTreeOpponent(Team.RED, config, difficulty=difficulty)
            while True:
                action_map = opponent.actions(env.state)
                for robot in env.state.team_robots(Team.BLUE):
                    obs = torch.as_tensor(local_observation(env.state, robot, config)[None], device=device)
                    mask = available_actions(robot, env.state, config)
                    with torch.no_grad():
                        logits = actor.act.action_out.linear(actor.base(obs))
                        logits = logits.masked_fill(torch.as_tensor(mask[None], device=device) <= 0, -1.0e9)
                        action_id = int(logits.argmax(dim=-1).item())
                    action_map[robot.name] = decode_action(action_id, robot, env.state, config)
                _, _, terminated, truncated, _ = env.step(action_map)
                if any(terminated.values()) or any(truncated.values()):
                    break
            own, other = env.state.score[Team.BLUE], env.state.score[Team.RED]
            goals_for += own
            goals_against += other
            wins += int(own > other)
            draws += int(own == other)
            scoreless += int(own == 0)
        results[difficulty] = dict(
            matches=matches, wins=wins, draws=draws, losses=matches - wins - draws,
            goals_for=goals_for, goals_against=goals_against,
            win_rate=wins / matches, goals_per_match=goals_for / matches,
            scoreless_rate=scoreless / matches,
        )
        print("BC EVAL opponent=%s W/D/L=%d/%d/%d goals=%d:%d "
              "win_rate=%.3f goals_per_match=%.3f scoreless_rate=%.3f" %
              (difficulty, wins, draws, matches - wins - draws, goals_for, goals_against,
               wins / matches, goals_for / matches, scoreless / matches), flush=True)
    actor.train()
    return results


def qualify_evaluation(evaluation, criteria):
    """Apply football-performance gates; classification accuracy is insufficient."""
    checks = (
        ("stationary.win_rate", evaluation["stationary"]["win_rate"],
         criteria["min_stationary_win_rate"], ">="),
        ("stationary.goals_per_match", evaluation["stationary"]["goals_per_match"],
         criteria["min_stationary_goals_per_match"], ">="),
        ("novice.goals_per_match", evaluation["novice"]["goals_per_match"],
         criteria["min_novice_goals_per_match"], ">="),
        ("novice.scoreless_rate", evaluation["novice"]["scoreless_rate"],
         criteria["max_novice_scoreless_rate"], "<="),
        ("standard.goals_per_match", evaluation["standard"]["goals_per_match"],
         criteria["min_standard_goals_per_match"], ">="),
        ("standard.scoreless_rate", evaluation["standard"]["scoreless_rate"],
         criteria["max_standard_scoreless_rate"], "<="),
    )
    failures = []
    for name, actual, required, operator in checks:
        passed = actual >= required if operator == ">=" else actual <= required
        if not passed:
            failures.append("%s=%.4f required %s %.4f" % (name, actual, operator, required))
    return {"qualified": not failures, "criteria": dict(criteria), "failures": failures}


def main():
    args = arguments()
    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    train_args = parse_mappo_args([
        "--algorithm_name", "mappo", "--experiment_name", "bc_rule_tree_31",
        "--seed", str(args.seed),
        "--hidden_size", str(args.hidden_size),
        "--layer_N", str(args.layer_N),
    ])
    actor = R_Actor(
        train_args,
        Box(-10.0, 10.0, shape=(observation_size(),)),
        Discrete(N_ACTIONS),
        device=device,
    ).to(device)
    if args.dagger_from:
        actor.load_state_dict(torch.load(str(args.dagger_from), map_location=device, weights_only=True))
        print("DAGGER loaded student", args.dagger_from, flush=True)
    obs, masks, labels = collect(args.samples, args.seed)
    if args.dagger_from and args.dagger_samples > 0:
        dagger_obs, dagger_masks, dagger_labels = collect_dagger(
            actor, device, args.dagger_samples, args.seed, args.dagger_beta,
            args.dagger_opponents)
        obs = np.concatenate((obs, dagger_obs), axis=0)
        masks = np.concatenate((masks, dagger_masks), axis=0)
        labels = np.concatenate((labels, dagger_labels), axis=0)
        print("BC merged teacher=%d dagger=%d total=%d" %
              (args.samples, len(dagger_labels), len(labels)), flush=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    indices = np.arange(len(labels))
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(indices)
        total_loss = total_correct = total = 0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            batch_obs = torch.as_tensor(obs[batch], device=device)
            batch_mask = torch.as_tensor(masks[batch], device=device)
            batch_labels = torch.as_tensor(labels[batch], device=device)
            features = actor.base(batch_obs)
            logits = actor.act.action_out.linear(features).masked_fill(batch_mask <= 0.0, -1.0e9)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss) * len(batch)
            total_correct += int((logits.argmax(dim=-1) == batch_labels).sum())
            total += len(batch)
        print("BC epoch=%d loss=%.6f accuracy=%.4f" % (epoch, total_loss / total, total_correct / total), flush=True)
    evaluation = evaluate_actor(actor, device, args.seed, matches=args.eval_matches)
    criteria = {key: getattr(args, key) for key in (
        "min_stationary_win_rate", "min_stationary_goals_per_match",
        "min_novice_goals_per_match", "max_novice_scoreless_rate",
        "min_standard_goals_per_match", "max_standard_scoreless_rate",
    )}
    gate = qualify_evaluation(evaluation, criteria)
    args.output.mkdir(parents=True, exist_ok=True)
    models = args.output / "models"
    models.mkdir(exist_ok=True)
    model_name = "actor.pt" if gate["qualified"] else "actor_rejected.pt"
    torch.save(actor.state_dict(), models / model_name)
    with (args.output / "config.json").open("w", encoding="utf-8") as stream:
        payload = dict(vars(train_args))
        payload["bc_evaluation"] = evaluation
        payload["bc_gate"] = gate
        json.dump(payload, stream, indent=2, default=str)
    print("BC GATE qualified=%s failures=%s" % (gate["qualified"], gate["failures"]), flush=True)
    print("saved", models / model_name)
    if not gate["qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
