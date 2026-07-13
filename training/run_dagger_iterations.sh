#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/data1/yangqinze/anaconda3/envs/mappo/bin/python}"
INITIAL="${1:-$ROOT/results/bc/rule_tree_goal_prioritized_31/run1/models/actor_rejected.pt}"
BASE="${2:-$ROOT/results/bc/rule_tree_dagger_auto}"
STUDENT="$INITIAL"
BETAS=(0.20 0.10 0.05 0.02)

cd "$ROOT"
for index in "${!BETAS[@]}"; do
  round=$((index + 1))
  output="$BASE/run$round"
  echo "AUTO_DAGGER round=$round beta=${BETAS[$index]} student=$STUDENT output=$output"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" training/pretrain_bc.py \
    --samples 500000 \
    --dagger_from "$STUDENT" \
    --dagger_samples 500000 \
    --dagger_beta "${BETAS[$index]}" \
    --epochs 20 \
    --batch_size 2048 \
    --lr 3e-4 \
    --hidden_size 512 \
    --layer_N 3 \
    --seed "$round" \
    --gpu_id 0 \
    --eval_matches 10 \
    --output "$output"
  status=$?
  if [[ -f "$output/models/actor.pt" ]]; then
    echo "AUTO_DAGGER qualified round=$round actor=$output/models/actor.pt"
    exit 0
  fi
  STUDENT="$output/models/actor_rejected.pt"
  if [[ ! -f "$STUDENT" ]]; then
    echo "AUTO_DAGGER aborted round=$round status=$status missing=$STUDENT"
    exit 3
  fi
  echo "AUTO_DAGGER retry round=$round status=$status next_student=$STUDENT"
done

echo "AUTO_DAGGER exhausted rounds=${#BETAS[@]} last_student=$STUDENT"
exit 2
