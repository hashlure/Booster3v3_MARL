import math

import numpy as np

from robocup3v3.adapters.onpolicy import OnPolicyTeamEnv
from robocup3v3.adapters.gymnasium_env import Robocup3v3TeamGymEnv
from robocup3v3.adapters.play_context import to_play_context_dict
from robocup3v3.adapters.booster import (
    action_from_play_context,
    action_mask_from_play_context,
    observation_from_play_context,
)
from robocup3v3.actions import ACTION_NAMES, N_ACTIONS, best_shot_target_y, decode_action, world_to_team
from robocup3v3.config import EnvConfig
from robocup3v3.env import Robocup3v3Env
from robocup3v3.types import GameState, Team
from robocup3v3.observations import local_observation


def test_core_parallel_shapes_and_simultaneous_actions():
    env = Robocup3v3Env()
    observations, infos = env.reset(seed=2)
    assert set(observations) == set(env.possible_agents)
    assert all(value.shape == (59,) for value in observations.values())
    actions = {name: 1 for name in env.possible_agents}
    result = env.step(actions)
    assert len(result) == 5
    assert all(np.isfinite(value).all() for value in result[0].values())


def test_onpolicy_smac_contract():
    env = OnPolicyTeamEnv()
    env.seed(11)
    obs, state, available = env.reset()
    assert obs.shape == (3, 59)
    assert state.shape == (3, 64)
    assert available.shape == (3, N_ACTIONS)
    result = env.step(np.zeros((3, 1), dtype=np.int64))
    assert len(result) == 6
    assert result[2].shape == (3, 4)
    assert result[3].shape == (3,)
    assert len(result[4]) == 3


def test_ready_action_mask_disables_kicks():
    env = Robocup3v3Env()
    env.reset(seed=3, options={"randomize": False})
    env.state.robots["blue_1"].pose.x = env.state.ball.x
    env.state.robots["blue_1"].pose.y = env.state.ball.y
    mask = env.action_mask("blue_1")
    assert env.state.game_state is GameState.READY
    assert not mask[9:19].any()
    assert not mask[22]


def test_playing_action_mask_requires_facing_ball_to_kick():
    env = Robocup3v3Env()
    env.reset(seed=3, options={"randomize": False})
    robot = env.state.robots["blue_1"]
    env.state.game_state = GameState.PLAYING
    env.state.stopped = False
    env.state.restart_pending = False
    env.state.ball.x = robot.pose.x + 0.2
    env.state.ball.y = robot.pose.y
    robot.pose.theta = math.pi
    assert not env.action_mask(robot.name)[22]
    robot.pose.theta = 0.0
    assert env.action_mask(robot.name)[22]


def test_gymnasium_team_wrapper_contract():
    env = Robocup3v3TeamGymEnv()
    obs, info = env.reset(seed=5)
    assert obs.shape == (3, 59)
    assert info["global_state"].shape == (64,)
    result = env.step(np.zeros(3, dtype=np.int64))
    assert len(result) == 5
    assert result[0].shape == (3, 59)


def test_play_context_team_view_and_freshness():
    env = Robocup3v3Env()
    env.reset(seed=1, options={"randomize": False})
    blue = to_play_context_dict(env.state, "blue")
    red = to_play_context_dict(env.state, "red")
    assert blue["ball"]["last_seen_at"] > 0.0
    assert red["ball"]["last_seen_at"] > 0.0
    assert blue["teammates"][1]["pose"]["x"] < 0.0
    assert red["teammates"][1]["pose"]["x"] < 0.0
    assert blue["game_state"]["players_per_team"] == 3
    assert red["game_state"]["kicking_team"] == 1


def test_training_and_booster_observations_are_identical():
    env = Robocup3v3Env()
    env.reset(seed=1, options={"randomize": False})
    for team_name, team_id in (("blue", 1), ("red", 2)):
        context = to_play_context_dict(
            env.state, team_name, match_duration_sec=env.config.match_duration_sec,
        )
        for player_id in (1, 2, 3):
            robot = env.state.robots["%s_%d" % (team_name, player_id)]
            expected = local_observation(env.state, robot, env.config)
            actual = observation_from_play_context(context, player_id, team_id, env.config)
            np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_training_and_booster_action_adapters_are_identical():
    env = Robocup3v3Env()
    env.reset(seed=2, options={"randomize": False})
    for team_name, team_id in (("blue", 1), ("red", 2)):
        context = to_play_context_dict(env.state, team_name)
        for player_id in (1, 2, 3):
            robot = env.state.robots["%s_%d" % (team_name, player_id)]
            for action_id in range(N_ACTIONS):
                expected = decode_action(action_id, robot, env.state, env.config)
                actual = action_from_play_context(action_id, context, player_id, env.config)
                assert actual == expected
            np.testing.assert_array_equal(
                action_mask_from_play_context(context, player_id, team_id, env.config),
                env.action_mask(robot.name),
            )


def test_random_rollout_remains_finite():
    env = Robocup3v3Env()
    observations, _ = env.reset(seed=99)
    rng = np.random.RandomState(99)
    for _ in range(1500):
        actions = {name: int(rng.randint(N_ACTIONS)) for name in env.possible_agents}
        observations, _, terminated, truncated, _ = env.step(actions)
        assert all(np.isfinite(value).all() for value in observations.values())
        if any(terminated.values()) or any(truncated.values()):
            observations, _ = env.reset(seed=99)


def test_headless_rgb_render():
    env = Robocup3v3Env(render_mode="rgb_array")
    env.reset(seed=8)
    frame = env.render()
    assert frame.shape == (640, 960, 3)
    assert frame.dtype == np.uint8
    assert frame.max() > frame.min()


def test_team_view_coordinates_are_exact_mirrors():
    env = Robocup3v3Env()
    env.reset(options={"randomize": False})
    for blue, red in zip(env.state.team_robots(Team.BLUE), env.state.team_robots(Team.RED)):
        bx, by = world_to_team(Team.BLUE, blue.pose.x, blue.pose.y)
        rx, ry = world_to_team(Team.RED, red.pose.x, red.pose.y)
        assert np.isclose(bx, rx)
        assert np.isclose(abs(by), abs(ry))


def test_best_gap_shot_is_continuous_and_inside_effective_goal():
    env = Robocup3v3Env()
    env.reset(options={"randomize": False})
    robot = env.state.robots["blue_1"]
    safe_half_width = env.config.goal_width / 2.0 - env.config.ball_radius
    action = decode_action(22, robot, env.state, env.config)
    assert action.kick_target_x == env.config.field_length / 2.0
    assert abs(action.kick_target_y) < safe_half_width
    assert ACTION_NAMES[22] == "shoot_best_gap"


def test_best_gap_moves_away_from_central_blocker():
    config = EnvConfig()
    target = best_shot_target_y(0.0, 0.0, [(4.5, 0.0, config.robot_radius)], config)
    assert abs(target) > 0.25
