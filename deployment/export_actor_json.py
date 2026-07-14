#!/usr/bin/env python3
"""Export a MAPPO actor state dict for dependency-free Python inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--training_step", type=int, default=None)
    args = parser.parse_args()

    state = torch.load(str(args.actor), map_location="cpu", weights_only=True)
    required = {
        "base.feature_norm.weight",
        "base.feature_norm.bias",
        "base.mlp.fc1.0.weight",
        "base.mlp.fc1.0.bias",
        "act.action_out.linear.weight",
        "act.action_out.linear.bias",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError("actor is missing deployment tensors: %s" % missing)

    output_dim = int(state["act.action_out.linear.bias"].numel())
    input_dim = int(state["base.feature_norm.bias"].numel())
    if input_dim != 59 or output_dim != 31:
        raise RuntimeError("expected actor contract 59 -> 31, got %d -> %d" %
                           (input_dim, output_dim))

    payload = {
        name: tensor.detach().cpu().float().tolist()
        for name, tensor in state.items()
        if name.startswith("base.") or name.startswith("act.action_out.linear.")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    metadata_path = args.metadata or args.output.with_name("actor_deploy_spec.json")
    layer_indices = sorted({
        int(name.split(".")[3])
        for name in payload
        if name.startswith("base.mlp.fc2.") and name.endswith(".0.weight")
    })
    metadata = {
        "format": "mappo-pure-python-json-v2",
        "source_actor": str(args.actor),
        "training_step": args.training_step,
        "observation_dim": input_dim,
        "action_count": output_dim,
        "hidden_size": len(payload["base.mlp.fc1.0.bias"]),
        "layer_indices": layer_indices,
        "dtype": "float32",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("exported", args.output, "bytes", args.output.stat().st_size)
    print("metadata", metadata_path)


if __name__ == "__main__":
    main()
