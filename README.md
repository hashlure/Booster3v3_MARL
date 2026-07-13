# Booster3v3_MARL

Lightweight 2D 3v3 soccer environment for upper-level MAPPO planning.  The core
uses NumPy only, exposes six simultaneous logical players, and aligns field,
GameController, team-view coordinates, freshness, and action semantics with the
Booster `MyAgent` project.

## What is implemented

- Six-agent parallel core and optional PettingZoo facade.
- SMAC-style three-agent adapter for the supplied `on-policy` MAPPO repository.
- READY -> SET -> PLAYING -> FINISHED state machine.
- Goals, complete-ball crossing, touchline out, corners, goal kicks, kickoff
  after goals, score/time limits and deterministic reset.
- Robot/ball kinematics, collision separation, friction, dribble/pass/shot.
- Restart ownership/distance, last-touch tracking, leaving-field and holding
  penalties.
- Local Actor observation restricted to deployable PlayContext information.
- Centralized Critic state with global simulator information.
- Stable 31-action Discrete planning catalogue with action masks; legacy IDs 0-21 are preserved.
- PlayContext-compatible dictionary adapter with nonzero freshness timestamps.

## Server environment

```bash
cd /data1/yangqinze/Robocup3v3
PY=/data1/yangqinze/anaconda3/envs/mappo/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q
$PY training/smoke_env.py
$PY training/evaluate_scripted.py
```

Short MAPPO smoke run:

```bash
$PY training/train_mappo.py \
  --algorithm_name mappo \
  --experiment_name smoke \
  --num_env_steps 64 \
  --episode_length 16 \
  --n_rollout_threads 1 \
  --n_training_threads 1 \
  --num_mini_batch 1 \
  --ppo_epoch 1 \
  --log_interval 1 \
  --save_interval 1
```

Long run example:

```bash
CUDA_VISIBLE_DEVICES=0 $PY training/train_mappo.py \
  --algorithm_name mappo \
  --experiment_name baseline \
  --num_env_steps 10000000 \
  --episode_length 256 \
  --n_rollout_threads 32 \
  --n_training_threads 8
```

The repository vendors the small set of MAPPO changes required for multi-head
critics, behavior-cloning initialization, curriculum/league training, and stable
football metrics under `on-policy/`. Models are written under `results/` and are
intentionally excluded from Git.

Export a non-recurrent MAPPO Actor as a self-contained TorchScript model:

```bash
$PY deployment/export_actor_torchscript.py results/mappo/baseline/run1
```

## Migration boundary

The trained Actor outputs one of 31 portable planning action IDs.  The catalogue
maps IDs to team-view field targets and tactical intent.  Deployment only needs
the Actor, observation normalization/ordering, the action catalogue, and a
behavior-tree adapter; physics and centralized-critic features are never shipped.

See `docs/RULES_ASSUMPTIONS.md` before interpreting the environment as an
official referee implementation.  Existing documents define the protocol, not
every normative 3v3 ruling.
