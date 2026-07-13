#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/data1/yangqinze/anaconda3/envs/mappo/bin/python}"
STUDENT="${1:-$ROOT/results/bc/rule_tree_dagger_31/run2/models/actor_rejected.pt}"
BASE="${2:-$ROOT/results/bc/rule_tree_dagger_31}"

# Attack curriculum: first learn to break the aggressive single chaser, then
# progressively restore coordinated standard/expert opponents.
ROUNDS=(3 4 5 6)
BETAS=(0.70 0.50 0.30 0.15)
OPPONENTS=(
  "novice,novice,novice"
  "novice,novice,standard"
  "novice,standard,standard"
  "standard,standard,expert"
)

cd "$ROOT"
for index in "${!ROUNDS[@]}"; do
  round="${ROUNDS[$index]}"
  output="$BASE/run$round"
  echo "ADAPTIVE_DAGGER start round=$round beta=${BETAS[$index]} opponents=${OPPONENTS[$index]} student=$STUDENT"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" training/pretrain_bc.py \
    --samples 300000 \
    --dagger_from "$STUDENT" \
    --dagger_samples 700000 \
    --dagger_beta "${BETAS[$index]}" \
    --dagger_opponents "${OPPONENTS[$index]}" \
    --epochs 20 \
    --batch_size 2048 \
    --lr 3e-4 \
    --hidden_size 512 \
    --layer_N 3 \
    --seed "$round" \
    --gpu_id 0 \
    --eval_matches 20 \
    --output "$output"
  status=$?
  if [[ -f "$output/models/actor.pt" ]]; then
    echo "ADAPTIVE_DAGGER QUALIFIED round=$round actor=$output/models/actor.pt"
    exit 0
  fi
  STUDENT="$output/models/actor_rejected.pt"
  if [[ ! -f "$STUDENT" ]]; then
    echo "ADAPTIVE_DAGGER ABORT round=$round status=$status missing=$STUDENT"
    exit 3
  fi
  echo "ADAPTIVE_DAGGER rejected round=$round status=$status next_student=$STUDENT"
done

echo "ADAPTIVE_DAGGER EXHAUSTED last_student=$STUDENT"
exit 2
