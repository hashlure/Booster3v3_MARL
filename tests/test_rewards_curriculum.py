from robocup3v3.adapters.onpolicy import OnPolicyTeamEnv
from robocup3v3.config import EnvConfig, RewardConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.physics import PhysicsResult
from robocup3v3.types import GameState, PlannerAction, PlannerIntent, Team
import math
import numpy as np


def test_checkpoint_reward_is_collected_only_once():
    env = Robocup3v3Env(EnvConfig(randomize_reset=False), RewardConfig())
    env.reset(options={"randomize": False})
    env.state.game_state = GameState.PLAYING
    env.state.stopped = False
    env.state.restart_pending = False
    env.state.ball.last_touch_team = Team.BLUE
    env.state.ball.x = 1.6
    env.reward_engine.compute(env.state, PhysicsResult())
    first = env.reward_engine.last_components[Team.BLUE]["checkpoint"]
    env.reward_engine.compute(env.state, PhysicsResult())
    second = env.reward_engine.last_components[Team.BLUE]["checkpoint"]
    assert first > 0.0
    assert second == 0.0


def test_curriculum_starts_at_3v0_and_promotes_by_goals():
    env = OnPolicyTeamEnv(
        config=EnvConfig(randomize_reset=False),
        controlled_team="blue",
        curriculum=True,
    )
    env.reset()
    assert env.curriculum_stage == 0
    assert not any(robot.active for robot in env.core.state.team_robots(Team.RED))
    env.curriculum_goals = env.stage_goal_thresholds[0]
    env.stage_results = [(2, 0)] * 5
    env._promote_curriculum()
    assert env.curriculum_stage == 1
    env.reset()
    assert sum(robot.active for robot in env.core.state.team_robots(Team.RED)) == 1
    active_ids = [robot.player_id for robot in env.core.state.team_robots(Team.RED) if robot.active]
    assert active_ids == [1]


def test_mixed_opponent_counts_include_full_3v3():
    env = OnPolicyTeamEnv(
        config=EnvConfig(randomize_reset=False), curriculum=True,
        opponent_count_mode="mixed", opponent_count_probs="0,0,0,1",
    )
    env.reset()
    assert env._opponent_count() == 3
    assert sum(robot.active for robot in env.core.state.team_robots(Team.RED)) == 3


def test_curriculum_does_not_promote_on_cumulative_goals_with_bad_recent_form():
    env = OnPolicyTeamEnv(config=EnvConfig(randomize_reset=False), curriculum=True)
    env.curriculum_goals = env.stage_goal_thresholds[0]
    env.stage_results = [(0, 0)] * 20
    env._promote_curriculum()
    assert env.curriculum_stage == 0


def test_dense_reward_components_are_exposed_in_info():
    env = Robocup3v3Env(EnvConfig(randomize_reset=False))
    env.reset(options={"randomize": False})
    _, _, _, _, infos = env.step({name: 0 for name in env.possible_agents})
    components = infos["blue_1"]["reward_components"]
    assert "checkpoint" in components
    assert "successful_pass" in components
    assert "shot_on_target" in components
    assert "turnover" in components
    assert "receiver_facing" in components
    assert "keeper_positioning" in components
    assert "keeper_save" in components


def test_receiver_facing_reward_prefers_looking_at_carrier():
    env = Robocup3v3Env(EnvConfig(randomize_reset=False))
    env.reset(options={"randomize": False})
    carrier = env.state.robots["blue_1"]
    receiver = env.state.robots["blue_2"]
    env.state.ball.x, env.state.ball.y = carrier.pose.x, carrier.pose.y
    env.state.ball.last_touch_team = Team.BLUE
    env.state.ball.last_touch_player = carrier.player_id
    desired = math.atan2(carrier.pose.y - receiver.pose.y, carrier.pose.x - receiver.pose.x)
    receiver.pose.theta = desired
    facing = env.reward_engine._receiver_facing_reward(env.state, Team.BLUE)
    receiver.pose.theta = desired + math.pi
    away = env.reward_engine._receiver_facing_reward(env.state, Team.BLUE)
    assert facing > away


def test_yaw_accelerates_instead_of_turning_instantly():
    config = EnvConfig(randomize_reset=False, action_noise=0.0)
    env = Robocup3v3Env(config)
    env.reset(options={"randomize": False})
    robot = env.state.robots["blue_1"]
    robot.pose.theta = 0.0
    action = PlannerAction(PlannerIntent.MOVE, robot.pose.x, robot.pose.y + 2.0,
                           robot.pose.x, robot.pose.y + 2.0)
    env.physics._move_robot(robot, action, np.random.RandomState(0))
    assert 0.0 < robot.vyaw < config.max_angular_speed
    assert robot.vyaw <= config.angular_acceleration * config.dt + 1e-8


def test_keeper_positioning_reward_prefers_goal_line_tracking_pose():
    env = Robocup3v3Env(EnvConfig(randomize_reset=False))
    env.reset(options={"randomize": False})
    keeper = env.state.robots["blue_3"]
    env.state.ball.x, env.state.ball.y = 0.0, 1.0
    keeper.pose.x = -env.config.field_length / 2.0 + 0.65
    keeper.pose.y = 0.60
    keeper.pose.theta = math.atan2(env.state.ball.y - keeper.pose.y,
                                   env.state.ball.x - keeper.pose.x)
    good = env.reward_engine._keeper_positioning_reward(env.state, Team.BLUE)
    keeper.pose.y = -3.0
    bad = env.reward_engine._keeper_positioning_reward(env.state, Team.BLUE)
    assert good > bad
