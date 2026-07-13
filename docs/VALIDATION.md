# Validation record

Validated on `10.103.11.97` with:

```text
Python 3.8.20
NumPy 1.19.5
Gym 0.17.2
PyTorch 2.4.1+cu121
8 x NVIDIA RTX 4090
```

Checks completed:

- 21 deterministic tests: rules, goals, continuous high-speed boundary crossing,
  touchline/corner/goal-kick, reset levels, truncation, numerical stability,
  Gym/SMAC interfaces, RGB rendering, and train/deploy observation-action parity.
- 2,000-step random environment smoke test with finite observations/states.
- Full scripted 3v3 match with goals and internal kickoff resets.
- MAPPO 128-step training update and TorchScript Actor export.
- Four-process `ShareSubprocVecEnv` MAPPO smoke test (256 steps).
- Supplied `on-policy` repository remains clean (`git status --porcelain` empty).

Commands:

```bash
cd /data1/yangqinze/Robocup3v3
PY=/data1/yangqinze/anaconda3/envs/mappo/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q
$PY training/smoke_env.py
$PY training/evaluate_scripted.py
```

Smoke artifacts:

```text
results/mappo/verified_smoke/run1/models/actor.pt
results/mappo/verified_smoke/run1/models/critic.pt
results/mappo/verified_smoke/run1/models/actor_deploy.ts
results/mappo/verified_smoke/run1/models/actor_deploy_spec.json
```

These models only prove the training/export path; 128 steps are not a trained
competitive policy.

