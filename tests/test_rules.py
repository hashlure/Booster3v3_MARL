import math

import numpy as np

from robocup3v3 import EnvConfig, GameState, Robocup3v3Env, SetPlay, Team
from robocup3v3.types import Penalty, RuleEvent


def playing_env():
    env = Robocup3v3Env(EnvConfig(randomize_reset=False, action_noise=0.0))
    env.reset(seed=1, options={"randomize": False})
    env.state.game_state = GameState.PLAYING
    env.state.stopped = False
    env.state.kicking_team = None
    env.state.restart_pending = False
    env.state.direct_goal_allowed = True
    env.state.state_elapsed = 0.0
    return env


def idle(env):
    return {name: 0 for name in env.possible_agents}


def events(infos, name="blue_1"):
    return infos[name]["events"]


def test_ball_must_fully_cross_goal_line():
    env = playing_env()
    env.state.ball.x = env.config.field_length / 2.0 + env.state.ball.radius * 0.5
    _, _, _, _, infos = env.step(idle(env))
    assert env.state.score[Team.BLUE] == 0
    assert not any(event["kind"] == "goal" for event in events(infos))


def test_high_speed_crossing_detects_goal_at_intersection():
    env = playing_env()
    env.state.ball.x = env.config.field_length / 2.0 - 0.2
    env.state.ball.y = 0.0
    env.state.ball.vx = 10.0
    env.step(idle(env))
    assert env.state.score[Team.BLUE] == 1


def test_diagonal_out_uses_first_crossed_boundary():
    env = playing_env()
    env.state.ball.x = env.config.field_length / 2.0 - 0.2
    env.state.ball.y = env.config.field_width / 2.0 - 0.5
    env.state.ball.vx = 10.0
    env.state.ball.vy = 20.0
    env.state.ball.last_touch_team = Team.BLUE
    env.step(idle(env))
    assert env.state.set_play is SetPlay.THROW_IN
    assert env.state.kicking_team is Team.RED


def test_goal_scores_and_performs_internal_kickoff_reset():
    env = playing_env()
    env.state.elapsed = 12.0
    env.state.ball.x = env.config.field_length / 2.0 + env.state.ball.radius + 0.02
    _, _, terminated, truncated, infos = env.step(idle(env))
    assert env.state.score[Team.BLUE] == 1
    assert env.state.score[Team.RED] == 0
    assert env.state.elapsed == 12.0
    assert env.state.game_state is GameState.READY
    assert env.state.kicking_team is Team.RED
    assert env.state.ball.x == 0.0 and env.state.ball.y == 0.0
    assert not any(terminated.values()) and not any(truncated.values())
    assert sum(event["kind"] == "goal" for event in events(infos)) == 1


def test_touchline_out_is_throw_in_without_team_reset():
    env = playing_env()
    before = {name: (r.pose.x, r.pose.y) for name, r in env.state.robots.items()}
    env.state.ball.last_touch_team = Team.BLUE
    env.state.ball.y = env.config.field_width / 2.0 + env.state.ball.radius + 0.02
    env.step(idle(env))
    assert env.state.set_play is SetPlay.THROW_IN
    assert env.state.kicking_team is Team.RED
    assert before == {name: (r.pose.x, r.pose.y) for name, r in env.state.robots.items()}
    assert env.state.score == {Team.BLUE: 0, Team.RED: 0}


def test_goal_line_restart_corner_or_goal_kick_from_last_touch():
    env = playing_env()
    env.state.ball.x = env.config.field_length / 2.0 + env.state.ball.radius + 0.02
    env.state.ball.y = env.config.goal_width
    env.state.ball.last_touch_team = Team.RED
    env.step(idle(env))
    assert env.state.set_play is SetPlay.CORNER_KICK
    assert env.state.kicking_team is Team.BLUE

    env = playing_env()
    env.state.ball.x = env.config.field_length / 2.0 + env.state.ball.radius + 0.02
    env.state.ball.y = env.config.goal_width
    env.state.ball.last_touch_team = Team.BLUE
    env.step(idle(env))
    assert env.state.set_play is SetPlay.GOAL_KICK
    assert env.state.kicking_team is Team.RED


def test_restart_has_placing_phase_then_becomes_active():
    config = EnvConfig(randomize_reset=False, restart_placing_duration_sec=0.1)
    env = Robocup3v3Env(config)
    env.reset(options={"randomize": False})
    env.state.game_state = GameState.PLAYING
    env.rules.start_restart(env.state, SetPlay.DIRECT_FREE_KICK, Team.BLUE, 0.0, 0.0)
    assert env.state.stopped
    env.step(idle(env))
    assert env.state.stopped
    env.step(idle(env))
    assert not env.state.stopped
    assert env.state.set_play is SetPlay.DIRECT_FREE_KICK


def test_indirect_restart_cannot_score_before_second_distinct_touch():
    env = playing_env()
    env.rules.start_restart(env.state, SetPlay.INDIRECT_FREE_KICK, Team.BLUE, 0.0, 0.0)
    env.state.stopped = False
    env.state.ball.x = env.config.field_length / 2.0 + env.state.ball.radius + 0.02
    env.state.ball.y = 0.0
    _, _, _, _, infos = env.step(idle(env))
    assert env.state.score[Team.BLUE] == 0
    assert env.state.set_play is SetPlay.GOAL_KICK
    assert env.state.kicking_team is Team.RED
    assert any(event["kind"] == "disallowed_direct_goal" for event in events(infos))


def test_ready_set_playing_state_machine():
    config = EnvConfig(randomize_reset=False, ready_duration_sec=0.2, set_duration_sec=0.1)
    env = Robocup3v3Env(config)
    env.reset(options={"randomize": False})
    env.step(idle(env))
    env.step(idle(env))
    env.step(idle(env))
    assert env.state.game_state is GameState.SET
    env.step(idle(env))
    assert env.state.game_state is GameState.PLAYING
    assert not env.state.stopped


def test_full_reset_is_seed_deterministic_and_clears_match_state():
    env = Robocup3v3Env()
    env.reset(seed=123)
    first = np.asarray([(r.pose.x, r.pose.y) for r in env.state.robots.values()])
    env.state.score[Team.BLUE] = 4
    env.state.ball.vx = 3.0
    env.state.ball.last_touch_team = Team.RED
    env.reset(seed=123)
    second = np.asarray([(r.pose.x, r.pose.y) for r in env.state.robots.values()])
    np.testing.assert_allclose(first, second)
    assert env.state.score == {Team.BLUE: 0, Team.RED: 0}
    assert env.state.ball.vx == 0.0
    assert env.state.ball.last_touch_team is None


def test_time_limit_is_truncation_with_bad_transition():
    env = Robocup3v3Env(EnvConfig(max_episode_steps=1, randomize_reset=False))
    env.reset(options={"randomize": False})
    _, _, terminations, truncations, infos = env.step(idle(env))
    assert not any(terminations.values())
    assert all(truncations.values())
    assert all(info["bad_transition"] for info in infos.values())


def test_non_finite_state_truncates():
    env = playing_env()
    env.state.ball.x = float("nan")
    _, _, _, truncations, infos = env.step(idle(env))
    assert all(truncations.values())
    assert infos["blue_1"]["termination_reason"] == "non_finite_state"


def test_official_rule_durations_are_defaults():
    config = EnvConfig()
    assert config.match_duration_sec == 600.0
    assert config.ready_duration_sec == 45.0
    assert config.ready_stable_duration_sec == 5.0
    assert config.set_duration_sec == 5.0
    assert config.kickoff_expiry_sec == 10.0
    assert config.set_play_expiry_sec == 45.0
    assert config.penalty_duration_sec == 30.0


def test_restart_expires_and_opens_direct_scoring():
    env = playing_env()
    env.rules.start_restart(env.state, SetPlay.THROW_IN, Team.BLUE, 0.0, 4.45)
    env.state.stopped = False
    env.state.elapsed = env.state.restart_expires_at
    env.rules.begin_step(env.state)
    assert not env.state.restart_pending
    assert env.state.kicking_team is None
    assert env.state.set_play is SetPlay.NONE
    assert env.state.direct_goal_allowed
    assert any(event.kind == "restart_expired" for event in env.state.events)


def test_defender_touch_before_restart_is_penalized_and_retake():
    env = playing_env()
    env.rules.start_restart(env.state, SetPlay.THROW_IN, Team.BLUE, 0.0, 4.45)
    env.state.stopped = False
    red = env.state.robots["red_1"]
    env.rules.accept_physics_events(env.state, [RuleEvent("touch", Team.RED, 1)])
    assert red.penalty is Penalty.ILLEGAL_POSITIONING
    assert red.penalty_remaining == 30.0
    assert env.state.restart_pending
    assert env.state.kicking_team is Team.BLUE
    assert any(event.kind == "restart_retake" for event in env.state.events)


def test_goal_kick_uses_official_approximately_six_two_placement():
    env = playing_env()
    env.state.ball.x = env.config.field_length / 2.0 + env.state.ball.radius + 0.02
    env.state.ball.y = 3.0
    env.state.ball.last_touch_team = Team.BLUE
    env.step(idle(env))
    assert env.state.set_play is SetPlay.GOAL_KICK
    assert env.state.kicking_team is Team.RED
    assert env.state.ball.x == 6.0
    assert env.state.ball.y == 2.0


def test_restart_own_goal_is_corner_not_score():
    env = playing_env()
    env.rules.start_restart(env.state, SetPlay.GOAL_KICK, Team.BLUE, -6.0, 2.0)
    env.state.stopped = False
    env.state.ball.x = -env.config.field_length / 2.0 - env.state.ball.radius - 0.02
    env.state.ball.y = 0.0
    env.step(idle(env))
    assert env.state.score == {Team.BLUE: 0, Team.RED: 0}
    assert env.state.set_play is SetPlay.CORNER_KICK
    assert env.state.kicking_team is Team.RED


def test_thirty_seconds_without_touch_triggers_stalemate_restart():
    env = playing_env()
    env.state.last_touch_at = 0.0
    env.state.elapsed = env.config.stalemate_duration_sec
    env.step(idle(env))
    assert env.state.game_state is GameState.READY
    assert env.state.ball.x == 0.0 and env.state.ball.y == 0.0
    assert any(event.kind == "stalemate_restart" for event in env.state.events)
    assert env.state.last_touch_at == env.state.elapsed
    env.state.game_state = GameState.PLAYING
    env.state.stopped = False
    env.state.events = []
    env.rules._check_stalemate(env.state)
    assert not any(event.kind == "stalemate_restart" for event in env.state.events)
