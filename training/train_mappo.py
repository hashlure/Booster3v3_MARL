#!/usr/bin/env python3
"""Train the blue three-agent CTDE policy with the supplied on-policy code."""

from __future__ import annotations

import os
import socket
import sys
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ON_POLICY = ROOT / "on-policy"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ON_POLICY))

import numpy as np
import setproctitle
import torch

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv

from robocup3v3.adapters.onpolicy import OnPolicyTeamEnv
from robocup3v3.config import EnvConfig


def parse_args(argv):
    parser = get_config()
    parser.add_argument("--map_name", type=str, default="adult_3v3")
    parser.add_argument("--controlled_team", type=str, default="blue", choices=("blue", "red"))
    parser.add_argument("--match_duration_sec", type=float, default=600.0)
    parser.add_argument("--env_max_episode_steps", type=int, default=7200)
    parser.add_argument("--train_ready_duration_sec", type=float, default=0.2)
    parser.add_argument("--train_set_duration_sec", type=float, default=0.1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--wandb_enabled", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="robocup3v3-mappo")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--actor_model_dir", type=str, default=None,
                        help="directory containing a BC-pretrained actor.pt; critic remains fresh")
    parser.add_argument("--allow_unqualified_bc", action="store_true", default=False,
                        help="unsafe escape hatch: load BC without a passing bc_gate")
    parser.add_argument("--critic_warmup_updates", type=int, default=0,
                        help="train only the fresh critic for N PPO updates before changing a BC actor")
    parser.add_argument("--disable_curriculum", action="store_true", default=False)
    parser.add_argument("--curriculum_initial_stage", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--opponent_count_mode", type=str, default="curriculum",
                        choices=("curriculum", "mixed", "fixed"))
    parser.add_argument("--opponent_count_probs", type=str, default="0.0,0.20,0.30,0.50",
                        help="probabilities for 0,1,2,3 active opponents in mixed mode")
    parser.add_argument("--fixed_opponent_count", type=int, default=3, choices=(0, 1, 2, 3))
    parser.add_argument("--opponent_difficulty_probs", type=str, default="",
                        help="optional stationary,novice,standard,expert sampling probabilities")
    parser.add_argument("--multi_critic_heads", type=int, default=4)
    parser.add_argument("--critic_weights", type=str, default="1.0,0.50,0.35,0.25")
    parser.add_argument("--birdview_critic", action="store_true", default=False)
    parser.add_argument("--birdview_channels", type=int, default=10)
    parser.add_argument("--birdview_height", type=int, default=32)
    parser.add_argument("--birdview_width", type=int, default=48)
    parser.add_argument("--birdview_vector_size", type=int, default=64)
    parser.add_argument("--league_enabled", action="store_true", default=False)
    parser.add_argument("--league_dir", type=str, default="results/league/main")
    parser.add_argument("--league_snapshot_interval_steps", type=int, default=5000000)
    parser.add_argument("--league_max_actor_opponents", type=int, default=8)
    parser.add_argument("--league_auto_swiss", action="store_true", default=False)
    parser.add_argument("--league_swiss_rounds", type=int, default=3)
    parser.add_argument("--league_swiss_duration", type=float, default=120.0)
    parser.add_argument("--league_seed_actor", type=str, default=None,
                        help="actor checkpoint copied into a new league as generation zero")
    args = parser.parse_args(argv)
    args.env_name = "Robocup3v3"
    # The upstream parser defaults to enabling wandb via a store_false flag.
    # Local JSON/TensorBoard logging is deterministic and needs no credentials.
    args.use_wandb = args.wandb_enabled
    if args.algorithm_name == "rmappo":
        args.use_recurrent_policy = True
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "mappo":
        args.use_recurrent_policy = False
        args.use_naive_recurrent_policy = False
    elif args.algorithm_name == "ippo":
        args.use_centralized_V = False
    else:
        raise ValueError("supported algorithms: mappo, rmappo, ippo")
    return args


def make_envs(args, evaluation=False):
    count = args.n_eval_rollout_threads if evaluation else args.n_rollout_threads

    def factory(rank):
        def init_env():
            config = EnvConfig(
                match_duration_sec=args.match_duration_sec,
                max_episode_steps=args.env_max_episode_steps,
                ready_duration_sec=args.train_ready_duration_sec,
                set_duration_sec=args.train_set_duration_sec,
            )
            env = OnPolicyTeamEnv(
                config=config,
                controlled_team=args.controlled_team,
                curriculum=(not evaluation and not args.disable_curriculum),
                birdview=args.birdview_critic,
                birdview_height=args.birdview_height,
                birdview_width=args.birdview_width,
                league_dir=(str((ROOT / args.league_dir).resolve()) if args.league_enabled else None),
                league_hidden_size=args.hidden_size,
                league_layer_n=args.layer_N,
                reward_heads=args.multi_critic_heads,
                curriculum_initial_stage=args.curriculum_initial_stage,
                opponent_count_mode=("fixed" if evaluation else args.opponent_count_mode),
                opponent_count_probs=args.opponent_count_probs,
                fixed_opponent_count=(3 if evaluation else args.fixed_opponent_count),
                opponent_difficulty_probs=args.opponent_difficulty_probs,
            )
            base = args.seed * 50000 if evaluation else args.seed
            env.seed(base + rank * 1000)
            return env
        return init_env

    constructors = [factory(rank) for rank in range(count)]
    return ShareDummyVecEnv(constructors) if count == 1 else ShareSubprocVecEnv(constructors)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(args.n_training_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    run_dir = ROOT / "results" / args.algorithm_name / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name[3:]) for p in run_dir.glob("run*") if p.name[3:].isdigit()]
    run_dir = run_dir / ("run%d" % ((max(existing) + 1) if existing else 1))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2, sort_keys=True, default=str)
    setproctitle.setproctitle("%s-Robocup3v3@%s" % (args.algorithm_name, socket.gethostname()))

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="%s-%s-seed%d" % (args.algorithm_name, args.experiment_name, args.seed),
            group=args.experiment_name,
            config=vars(args),
            dir=str(run_dir),
            mode=args.wandb_mode,
        )

    if args.league_enabled:
        from robocup3v3.league import LeaguePool
        league_pool = LeaguePool((ROOT / args.league_dir).resolve(), seed=args.seed)
        if args.league_seed_actor:
            source = Path(args.league_seed_actor).resolve()
            if not source.is_file():
                raise RuntimeError("league seed actor missing: %s" % source)
            destination = league_pool.directory / "snapshots" / "actor_seed.pt"
            if not destination.exists():
                shutil.copy2(str(source), str(destination))
            league_pool.register_snapshot(destination, 0)
    envs = make_envs(args)
    eval_envs = make_envs(args, evaluation=True) if args.use_eval else None
    from onpolicy.runner.shared.smac_runner import SMACRunner

    runner = SMACRunner({
        "all_args": args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": 3,
        "device": device,
        "run_dir": run_dir,
    })
    if args.actor_model_dir:
        model_dir = Path(args.actor_model_dir)
        actor_path = model_dir / "actor.pt"
        if args.allow_unqualified_bc and not actor_path.is_file():
            rejected_path = model_dir / "actor_rejected.pt"
            if rejected_path.is_file():
                actor_path = rejected_path
                print("WARNING loading unqualified BC actor", actor_path, flush=True)
        config_path = model_dir.parent / "config.json"
        if not args.allow_unqualified_bc:
            if not config_path.is_file():
                raise RuntimeError("BC gate failed: missing qualification report %s" % config_path)
            with config_path.open(encoding="utf-8") as stream:
                bc_config = json.load(stream)
            gate = bc_config.get("bc_gate", {})
            if gate.get("qualified") is not True:
                raise RuntimeError("BC gate failed: %s; retrain/validate BC or explicitly pass "
                                   "--allow_unqualified_bc" % gate.get("failures", "no bc_gate"))
        if not actor_path.is_file():
            raise RuntimeError("qualified BC actor missing: %s" % actor_path)
        runner.trainer.policy.actor.load_state_dict(
            torch.load(str(actor_path), map_location=device, weights_only=True)
        )
        print("loaded BC actor", actor_path, flush=True)
    try:
        runner.run()
    finally:
        envs.close()
        if eval_envs is not None:
            eval_envs.close()
        if args.use_wandb:
            if wandb_run is not None:
                wandb_run.finish()
        else:
            runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
            runner.writter.close()


if __name__ == "__main__":
    main()
