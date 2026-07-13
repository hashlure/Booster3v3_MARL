#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/data1/yangqinze/anaconda3/envs/mappo/bin/python"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" training/train_mappo.py \
  --algorithm_name mappo \
  --experiment_name learnability_stage1_1m \
  --seed 22 --cuda --gpu_id 0 \
  --n_rollout_threads 32 --n_training_threads 8 \
  --episode_length 128 --num_env_steps 1000000 \
  --match_duration_sec 60 --env_max_episode_steps 650 \
  --train_ready_duration_sec 0.2 --train_set_duration_sec 0.1 \
  --curriculum_initial_stage 1 \
  --hidden_size 512 --layer_N 3 \
  --multi_critic_heads 1 --critic_weights 1.0 \
  --critic_warmup_updates 10 \
  --actor_model_dir results/mappo/learnability_scalar_warmup_300k/run1/models \
  --allow_unqualified_bc \
  --gamma 0.997 --gae_lambda 0.995 \
  --lr 5e-5 --critic_lr 3e-4 --entropy_coef 0.003 \
  --ppo_epoch 5 --num_mini_batch 8 \
  --log_interval 5 --save_interval 20

"$PYTHON" training/evaluate_actor_vs_trees.py \
  --actor results/mappo/learnability_stage1_1m/run1/models/actor.pt \
  --matches 20 --duration 60 --seed 2200 \
  --hidden_size 512 --layer_N 3 --opponent_count 3 \
  --output results/diagnostics/learnability_stage1_1m_full3.json
