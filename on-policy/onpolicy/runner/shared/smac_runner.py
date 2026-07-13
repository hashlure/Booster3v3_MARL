import time
import wandb
import numpy as np
from functools import reduce
import torch
import subprocess
import sys
from pathlib import Path
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self.league_pool = None
        self.next_league_snapshot = None
        if self.env_name == "Robocup3v3" and getattr(self.all_args, "league_enabled", False):
            from robocup3v3.league import LeaguePool
            league_dir = Path(__file__).resolve().parents[4] / self.all_args.league_dir
            self.league_pool = LeaguePool(league_dir.resolve(), seed=self.all_args.seed)
            self.next_league_snapshot = int(self.all_args.league_snapshot_interval_steps)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_goals_for = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_goals_against = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_draws = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_reward_totals = {}
        last_reward_abs_totals = {}
        last_reward_log_steps = 0
        last_opponent_type_counts = {}
        last_opponent_id_counts = {}
        metric_ema = {}

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                if self.league_pool is not None:
                    league_results = []
                    for env_info in infos:
                        result = env_info[0].get("league_match_result")
                        if result:
                            league_results.append(result)
                    if league_results:
                        self.league_pool.update_results(league_results)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            warmup_updates = int(getattr(self.all_args, "critic_warmup_updates", 0))
            if episode < warmup_updates:
                self.trainer.prep_training()
                train_infos = self.trainer.train(self.buffer, update_actor=False)
                self.buffer.after_update()
                train_infos["critic_warmup/actor_frozen"] = 1.0
            else:
                train_infos = self.train()
                train_infos["critic_warmup/actor_frozen"] = 0.0
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
            if (self.league_pool is not None and total_num_steps >= self.next_league_snapshot):
                snapshot_dir = self.league_pool.directory / "snapshots"
                snapshot_path = snapshot_dir / ("actor_%012d.pt" % total_num_steps)
                torch.save(self.trainer.policy.actor.state_dict(), snapshot_path)
                entry_id = self.league_pool.register_snapshot(snapshot_path, total_num_steps)
                self.league_pool.prune_actor_entries(int(self.all_args.league_max_actor_opponents))
                print("LEAGUE snapshot=%s steps=%d" % (entry_id, total_num_steps), flush=True)
                if getattr(self.all_args, "league_auto_swiss", False):
                    swiss_script = Path(__file__).resolve().parents[4] / "training" / "run_league_swiss.py"
                    command = [
                        sys.executable, str(swiss_script),
                        "--league_dir", str(self.league_pool.directory),
                        "--rounds", str(self.all_args.league_swiss_rounds),
                        "--duration", str(self.all_args.league_swiss_duration),
                        "--hidden_size", str(self.hidden_size),
                        "--layer_N", str(self.all_args.layer_N),
                        "--seed", str(self.all_args.seed + total_num_steps),
                    ]
                    completed = subprocess.run(command, check=False)
                    if completed.returncode != 0:
                        print("LEAGUE swiss failed returncode=%d" % completed.returncode, flush=True)
                while self.next_league_snapshot <= total_num_steps:
                    self.next_league_snapshot += int(self.all_args.league_snapshot_interval_steps)

            # log information
            if episode % self.log_interval == 0:
                wandb_step_payload = {}
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name in ("StarCraft2", "SMACv2", "SMAC", "StarCraft2v2", "Robocup3v3"):
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    
                    goals_for = []
                    goals_against = []
                    draws = []
                    incre_goals_for = []
                    incre_goals_against = []
                    incre_draws = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])
                        if 'goals_for' in info[0].keys():
                            goals_for.append(info[0]['goals_for'])
                            incre_goals_for.append(info[0]['goals_for']-last_goals_for[i])
                        if 'goals_against' in info[0].keys():
                            goals_against.append(info[0]['goals_against'])
                            incre_goals_against.append(info[0]['goals_against']-last_goals_against[i])
                        if 'draws' in info[0].keys():
                            draws.append(info[0]['draws'])
                            incre_draws.append(info[0]['draws']-last_draws[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    completed = np.sum(incre_battles_game)
                    cumulative_matches = float(np.sum(battles_game))
                    cumulative_wins = float(np.sum(battles_won))
                    football_metrics = {
                        "train/cumulative_matches": cumulative_matches,
                        "train/cumulative_win_rate": (cumulative_wins / cumulative_matches
                                                      if cumulative_matches > 0 else 0.0),
                        "train/cumulative_goals_for": float(np.sum(goals_for)),
                        "train/cumulative_goals_against": float(np.sum(goals_against)),
                    }
                    if completed > 0:
                        football_metrics.update({
                            "train/win_rate": float(incre_win_rate),
                            "train/matches": float(completed),
                            "train/draws": float(np.sum(incre_draws)),
                            "train/goals_for": float(np.sum(incre_goals_for)),
                            "train/goals_against": float(np.sum(incre_goals_against)),
                            "train/goal_difference": float(np.sum(incre_goals_for) - np.sum(incre_goals_against)),
                            "train/goals_for_per_match": float(np.sum(incre_goals_for) / completed),
                            "train/goals_against_per_match": float(np.sum(incre_goals_against) / completed),
                        })
                        previous_ema = metric_ema.get("train/win_rate", float(incre_win_rate))
                        metric_ema["train/win_rate"] = .9 * previous_ema + .1 * float(incre_win_rate)
                        football_metrics["train_ema/win_rate"] = metric_ema["train/win_rate"]
                    if self.env_name == "Robocup3v3":
                        reward_keys = sorted(
                            key for key in infos[0][0].keys()
                            if key.startswith("reward_total_")
                        )
                        for reward_key in reward_keys:
                            current = np.asarray(
                                [float(info[0].get(reward_key, 0.0)) for info in infos],
                                dtype=np.float64,
                            )
                            previous = last_reward_totals.get(reward_key, np.zeros_like(current))
                            component_name = reward_key[len("reward_total_"):]
                            raw_delta = float(np.sum(current - previous))
                            football_metrics["reward/" + component_name] = raw_delta
                            interval_steps = max(total_num_steps - last_reward_log_steps, 1)
                            football_metrics["reward_per_1000_steps/" + component_name] = (
                                raw_delta * 1000.0 / interval_steps)
                            if completed > 0:
                                football_metrics["reward_per_match/" + component_name] = (
                                    raw_delta / completed)
                            last_reward_totals[reward_key] = current
                        signed_rewards = [
                            value for key, value in football_metrics.items()
                            if key.startswith("reward/")
                        ]
                        football_metrics["reward/positive_total"] = float(sum(value for value in signed_rewards if value > 0.0))
                        football_metrics["reward/negative_total"] = float(sum(value for value in signed_rewards if value < 0.0))
                        football_metrics["reward/net_total"] = float(sum(signed_rewards))
                        absolute_keys = sorted(
                            key for key in infos[0][0].keys()
                            if key.startswith("reward_abs_total_")
                        )
                        for absolute_key in absolute_keys:
                            current = np.asarray(
                                [float(info[0].get(absolute_key, 0.0)) for info in infos],
                                dtype=np.float64,
                            )
                            previous = last_reward_abs_totals.get(
                                absolute_key, np.zeros_like(current))
                            absolute_delta = float(np.sum(current - previous))
                            component_name = absolute_key[len("reward_abs_total_"):]
                            football_metrics["reward_abs/" + component_name] = absolute_delta
                            football_metrics["reward_abs_per_1000_steps/" + component_name] = (
                                absolute_delta * 1000.0 / interval_steps)
                            last_reward_abs_totals[absolute_key] = current
                        interval_steps = max(total_num_steps - last_reward_log_steps, 1)
                        football_metrics["reward_per_1000_steps/positive_total"] = (
                            football_metrics["reward/positive_total"] * 1000.0 / interval_steps)
                        football_metrics["reward_per_1000_steps/negative_total"] = (
                            football_metrics["reward/negative_total"] * 1000.0 / interval_steps)
                        football_metrics["reward_per_1000_steps/net_total"] = (
                            football_metrics["reward/net_total"] * 1000.0 / interval_steps)
                        football_metrics["curriculum/stage"] = float(np.mean([
                            info[0].get("curriculum_stage", 3) for info in infos
                        ]))
                        football_metrics["curriculum/opponents"] = float(np.mean([
                            info[0].get("curriculum_opponents", 3) for info in infos
                        ]))
                        football_metrics["opponent/difficulty_index"] = float(np.mean([
                            info[0].get("opponent_difficulty", 2) for info in infos
                        ]))
                        if self.league_pool is not None:
                            league_data = self.league_pool.load()
                            active_entries = self.league_pool.eligible_entries(league_data)
                            football_metrics["league/current_elo"] = float(league_data["learner_rating"])
                            football_metrics["league/pool_size"] = float(len(league_data["entries"]))
                            football_metrics["league/active_pool_size"] = float(len(active_entries))
                            football_metrics["league/snapshot_count"] = float(league_data.get("snapshot_count", 0))
                            football_metrics["league/opponent_elo"] = float(np.mean([
                                info[0].get("league_opponent_rating", 0.0) for info in infos
                            ]))
                            football_metrics["league/rating_gap"] = (
                                football_metrics["league/opponent_elo"]
                                - football_metrics["league/current_elo"])
                            football_metrics["league/opponent_actor_fraction"] = float(np.mean([
                                1.0 if info[0].get("league_opponent_kind") == "actor" else 0.0
                                for info in infos
                            ]))
                            active_raw = sum(1 for entry in league_data["entries"]
                                             if entry.get("active", True))
                            football_metrics["league/weak_opponents_rejected"] = float(
                                max(0, active_raw - len(active_entries)))
                            football_metrics["opponent_sampling_target/nearby"] = 0.55
                            football_metrics["opponent_sampling_target/stronger"] = 0.20
                            football_metrics["opponent_sampling_target/anchors"] = 0.15
                            football_metrics["opponent_sampling_target/diversity"] = 0.10
                        type_deltas = {}
                        for difficulty in ("stationary", "novice", "standard", "expert", "actor"):
                            key = "opponent_%s_matches" % difficulty
                            current = np.asarray([
                                float(info[0].get(key, 0)) for info in infos
                            ], dtype=np.float64)
                            previous = last_opponent_type_counts.get(
                                difficulty, np.zeros_like(current))
                            type_deltas[difficulty] = max(0.0, float(np.sum(current - previous)))
                            last_opponent_type_counts[difficulty] = current
                        interval_sample_count = float(sum(type_deltas.values()))
                        total_samples = max(interval_sample_count, 1.0)
                        football_metrics["opponent/sample_count_total"] = interval_sample_count
                        for difficulty, delta in type_deltas.items():
                            football_metrics["opponent_sample_count/" + difficulty] = delta
                            if interval_sample_count > 0:
                                football_metrics["opponent_sample_fraction/" + difficulty] = delta / total_samples
                            cumulative = float(np.sum(last_opponent_type_counts[difficulty]))
                            cumulative_total = float(sum(
                                np.sum(value) for value in last_opponent_type_counts.values()))
                            football_metrics["opponent_sample_fraction_total/" + difficulty] = (
                                cumulative / cumulative_total if cumulative_total > 0 else 0.0)

                        id_maps = [info[0].get("opponent_id_sample_counts", {}) for info in infos]
                        opponent_ids = sorted(set().union(*(mapping.keys() for mapping in id_maps)))
                        id_deltas = {}
                        for opponent_id in opponent_ids:
                            current = np.asarray([
                                float(mapping.get(opponent_id, 0)) for mapping in id_maps
                            ], dtype=np.float64)
                            previous = last_opponent_id_counts.get(
                                opponent_id, np.zeros_like(current))
                            id_deltas[opponent_id] = max(0.0, float(np.sum(current - previous)))
                            last_opponent_id_counts[opponent_id] = current
                        cumulative_all_ids = float(sum(
                            np.sum(value) for value in last_opponent_id_counts.values()))
                        for opponent_id in opponent_ids:
                            delta = id_deltas[opponent_id]
                            football_metrics["opponent_id_sample_count/" + opponent_id] = delta
                            if interval_sample_count > 0:
                                football_metrics["opponent_id_sample_fraction/" + opponent_id] = delta / total_samples
                            cumulative_id = float(np.sum(last_opponent_id_counts[opponent_id]))
                            football_metrics["opponent_id_sample_fraction_total/" + opponent_id] = (
                                cumulative_id / cumulative_all_ids if cumulative_all_ids > 0 else 0.0)

                        for metric_name, metric_value in list(football_metrics.items()):
                            if metric_name.startswith("reward_per_1000_steps/"):
                                ema_key = metric_name.replace(
                                    "reward_per_1000_steps/", "reward_ema_per_1000_steps/")
                                previous = metric_ema.get(ema_key, float(metric_value))
                                metric_ema[ema_key] = .9 * previous + .1 * float(metric_value)
                                football_metrics[ema_key] = metric_ema[ema_key]
                    if self.env_name == "Robocup3v3":
                        print(
                            "football matches={:.0f} win_rate={:.3f} draws={:.0f} goals={:.0f}:{:.0f} diff={:+.0f}".format(
                                football_metrics.get("train/matches", 0.0), football_metrics.get("train/win_rate", 0.0),
                                football_metrics.get("train/draws", 0.0), football_metrics.get("train/goals_for", 0.0),
                                football_metrics.get("train/goals_against", 0.0), football_metrics.get("train/goal_difference", 0.0),
                            )
                        )
                        print(
                            "opponent sample mix stationary={:.3f} novice={:.3f} standard={:.3f} expert={:.3f} actor={:.3f} n={:.0f}".format(
                                football_metrics.get("opponent_sample_fraction/stationary", 0.0),
                                football_metrics.get("opponent_sample_fraction/novice", 0.0),
                                football_metrics.get("opponent_sample_fraction/standard", 0.0),
                                football_metrics.get("opponent_sample_fraction/expert", 0.0),
                                football_metrics.get("opponent_sample_fraction/actor", 0.0),
                                football_metrics.get("opponent/sample_count_total", 0.0),
                            )
                        )
                    if self.use_wandb:
                        payload = {"incre_win_rate": incre_win_rate} if completed > 0 else {}
                        if self.env_name == "Robocup3v3":
                            payload.update(football_metrics)
                        if payload:
                            wandb_step_payload.update(payload)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                        if self.env_name == "Robocup3v3":
                            for metric_name, metric_value in football_metrics.items():
                                self.writter.add_scalar(metric_name, metric_value, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won
                    if self.env_name == "Robocup3v3":
                        last_goals_for = goals_for
                        last_goals_against = goals_against
                        last_draws = draws
                        last_reward_log_steps = total_num_steps

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                if self.use_wandb:
                    train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                    wandb_step_payload.update(train_infos)
                    wandb.log(wandb_step_payload, step=total_num_steps)
                else:
                    self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        if self.use_wandb:
            wandb.log(train_infos, step=total_num_steps)
        else:
            for k, v in train_infos.items():
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
