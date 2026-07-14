#!/usr/bin/env python3
"""Compare MAPPO decision traces from training simulation and Booster runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


FEATURE_NAMES = (
    "self_x", "self_y", "self_cos", "self_sin", "self_active", "self_penalty",
    "ball_rel_x", "ball_rel_y", "ball_x", "ball_y", "ball_visible",
    *(f"teammate_{slot}_{field}" for slot in range(2)
      for field in ("rel_x", "rel_y", "cos", "sin", "active")),
    *(f"opponent_{slot}_{field}" for slot in range(3)
      for field in ("rel_x", "rel_y", "cos", "sin", "active")),
    *(f"game_{name}" for name in ("initial", "ready", "set", "playing", "finished")),
    *(f"setplay_{name}" for name in (
        "none", "direct_free_kick", "indirect_free_kick", "penalty_kick",
        "throw_in", "goal_kick", "corner_kick")),
    "kick_own", "kick_opponent", "kick_none", "time_remaining", "score_difference",
    *(f"previous_intent_{name}" for name in (
        "hold", "move", "dribble", "pass", "shoot", "guard")),
)


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def stats(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "p01": quantile(values, .01),
        "p50": quantile(values, .50), "p95": quantile(values, .95),
        "p99": quantile(values, .99), "max": max(values),
    }


def read_decisions(path):
    decisions = []
    errors = Counter()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors["invalid_json"] += 1
                continue
            event = record.get("event")
            if event == "rl_decision" or record.get("record_type") == "rl_decision":
                observation = record.get("observation")
                mask = record.get("action_mask", record.get("mask"))
                logits = record.get("logits")
                if not (isinstance(observation, list) and len(observation) == 59):
                    errors["bad_observation"] += 1
                    continue
                if not (isinstance(mask, list) and len(mask) == 31):
                    errors["bad_mask"] += 1
                    continue
                if not (isinstance(logits, list) and len(logits) == 31):
                    errors["bad_logits"] += 1
                    continue
                record["_line"] = line_number
                decisions.append(record)
            elif event == "rl_inference_failed":
                errors["inference_failed"] += 1
    return decisions, dict(errors)


def summarize(decisions, errors):
    observations = [[] for _ in range(59)]
    legal = [0] * 31
    action_counts = Counter()
    states = Counter()
    players = Counter()
    intents = Counter()
    inference = []
    ball_ages = []
    pose_ages = []
    margins = []
    for record in decisions:
        for index, value in enumerate(record["observation"]):
            observations[index].append(float(value))
        for index, value in enumerate(record.get("action_mask", record.get("mask"))):
            legal[index] += int(float(value) > 0.0)
        action_counts[int(record["action_id"])] += 1
        states[str(record.get("game_state", "unknown"))] += 1
        players[str(record.get("player_id", "unknown"))] += 1
        command = record.get("command") or {}
        intents[str(command.get("intent", "unknown"))] += 1
        if record.get("inference_ms") is not None:
            inference.append(float(record["inference_ms"]))
        ball = record.get("ball") or {}
        if ball.get("age_sec") is not None:
            ball_ages.append(float(ball["age_sec"]))
        robot = record.get("self_robot") or {}
        if robot.get("age_sec") is not None:
            pose_ages.append(float(robot["age_sec"]))
        mask = record.get("action_mask", record.get("mask"))
        allowed = sorted(
            (float(record["logits"][i]) for i in range(31) if float(mask[i]) > 0.0),
            reverse=True,
        )
        if len(allowed) > 1:
            margins.append(allowed[0] - allowed[1])
    count = max(len(decisions), 1)
    return {
        "decision_count": len(decisions), "errors": errors,
        "observation": {
            FEATURE_NAMES[index]: stats(values) for index, values in enumerate(observations)
        },
        "legal_fraction": {str(index): legal[index] / count for index in range(31)},
        "action_fraction": {
            str(index): action_counts[index] / count for index in range(31)
        },
        "action_count": dict(action_counts), "game_states": dict(states),
        "players": dict(players), "command_intents": dict(intents),
        "inference_ms": stats(inference), "ball_age_sec": stats(ball_ages),
        "pose_age_sec": stats(pose_ages), "top2_logit_margin": stats(margins),
    }


def compare(sim, real):
    feature_shift = []
    for name in FEATURE_NAMES:
        left, right = sim["observation"][name], real["observation"][name]
        if not left.get("count") or not right.get("count"):
            continue
        scale = math.sqrt(left["std"] ** 2 + right["std"] ** 2 + 1.0e-8)
        feature_shift.append({
            "feature": name,
            "sim_mean": left["mean"], "deployment_mean": right["mean"],
            "standardized_shift": abs(left["mean"] - right["mean"]) / scale,
            "sim_min": left["min"], "sim_max": left["max"],
            "deployment_min": right["min"], "deployment_max": right["max"],
        })
    feature_shift.sort(key=lambda item: item["standardized_shift"], reverse=True)
    action_l1 = sum(abs(
        sim["action_fraction"][str(index)] - real["action_fraction"][str(index)]
    ) for index in range(31)) / 2.0
    mask_l1 = sum(abs(
        sim["legal_fraction"][str(index)] - real["legal_fraction"][str(index)]
    ) for index in range(31)) / 31.0
    action_shift = sorted(({
        "action_id": index,
        "sim_fraction": sim["action_fraction"][str(index)],
        "deployment_fraction": real["action_fraction"][str(index)],
        "absolute_shift": abs(sim["action_fraction"][str(index)] -
                              real["action_fraction"][str(index)]),
    } for index in range(31)), key=lambda item: item["absolute_shift"], reverse=True)
    return {
        "observation_shift": feature_shift,
        "action_total_variation": action_l1,
        "mask_mean_absolute_shift": mask_l1,
        "action_shift": action_shift,
    }


def markdown(report):
    sim, real, gap = report["simulation"], report["deployment"], report["gap"]
    lines = [
        "# MAPPO Sim-to-Deployment Gap Report", "",
        f"- Simulation decisions: {sim['decision_count']}",
        f"- Deployment decisions: {real['decision_count']}",
        f"- Action distribution total variation: {gap['action_total_variation']:.3f}",
        f"- Mean action-mask shift: {gap['mask_mean_absolute_shift']:.3f}",
        f"- Deployment inference p50/p95/max: "
        f"{real['inference_ms'].get('p50', 0):.2f}/"
        f"{real['inference_ms'].get('p95', 0):.2f}/"
        f"{real['inference_ms'].get('max', 0):.2f} ms", "",
        "## Largest observation shifts", "",
        "| Feature | Standardized shift | Sim mean | Deployment mean |", "|---|---:|---:|---:|",
    ]
    for item in gap["observation_shift"][:15]:
        lines.append(
            f"| {item['feature']} | {item['standardized_shift']:.3f} | "
            f"{item['sim_mean']:.4f} | {item['deployment_mean']:.4f} |"
        )
    lines.extend(["", "## Largest action shifts", "",
                  "| Action | Sim | Deployment | Absolute shift |",
                  "|---:|---:|---:|---:|"])
    for item in gap["action_shift"][:12]:
        lines.append(
            f"| {item['action_id']} | {item['sim_fraction']:.3f} | "
            f"{item['deployment_fraction']:.3f} | {item['absolute_shift']:.3f} |"
        )
    lines.extend(["", "## Diagnostic priorities", ""])
    priorities = []
    if real["ball_age_sec"].get("p95", 0) > .2:
        priorities.append("Ball observations are stale at p95; add latency/noise randomization and state prediction.")
    if real["pose_age_sec"].get("p95", 0) > .2:
        priorities.append("Robot poses are stale at p95; align timestamping and train with delayed poses.")
    if real["inference_ms"].get("p95", 0) > 80:
        priorities.append("Inference consumes most of a 10 Hz control period; cache team features or use TorchScript.")
    if gap["mask_mean_absolute_shift"] > .10:
        priorities.append("Deployment action masks differ materially; unify kick-facing, nearest-controller and restart legality.")
    if gap["action_total_variation"] > .25:
        priorities.append("Policy actions differ materially; fix observation/mask shift before further RL training.")
    if gap["observation_shift"] and gap["observation_shift"][0]["standardized_shift"] > .5:
        priorities.append("Large observation covariate shift exists; use recorded deployment states for replay and domain randomization.")
    if not priorities:
        priorities.append("No single dominant shift crossed the default threshold; inspect execution response and per-opponent slices.")
    lines.extend(f"1. {item}" for item in priorities)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sim_records, sim_errors = read_decisions(args.simulation)
    real_records, real_errors = read_decisions(args.deployment)
    if not sim_records or not real_records:
        raise RuntimeError("both traces need valid rl_decision records")
    sim = summarize(sim_records, sim_errors)
    real = summarize(real_records, real_errors)
    report = {"simulation": sim, "deployment": real, "gap": compare(sim, real)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(markdown(report), encoding="utf-8")
    print("wrote", args.output)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
