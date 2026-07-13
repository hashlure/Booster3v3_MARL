"""Simulator configuration aligned with the adult-size Booster field."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    # MyAgent ADULT_FIELD_DIMENSIONS
    field_length: float = 14.0
    field_width: float = 9.0
    goal_width: float = 2.6
    goal_depth: float = 0.6
    center_circle_radius: float = 1.5
    penalty_area_length: float = 3.0
    penalty_area_width: float = 6.0
    goal_area_length: float = 1.0
    goal_area_width: float = 4.0

    dt: float = 0.10
    match_duration_sec: float = 600.0
    max_episode_steps: int = 7200
    score_limit: int = 10
    ready_duration_sec: float = 45.0
    ready_stable_duration_sec: float = 5.0
    set_duration_sec: float = 5.0
    restart_placing_duration_sec: float = 0.2

    robot_radius: float = 0.28
    ball_radius: float = 0.11
    max_robot_speed: float = 0.8
    # Humanoid yaw cannot jump to the target heading in one control tick.
    max_angular_speed: float = 0.60
    angular_acceleration: float = 1.50
    robot_acceleration: float = 2.0
    ball_friction: float = 0.90
    ball_restitution: float = 0.45
    kick_distance: float = 0.50
    kick_facing_tolerance_rad: float = 0.55
    max_control_ball_speed: float = 2.5
    dribble_speed: float = 0.80
    pass_speed: float = 2.20
    shot_speed: float = 3.80

    # 1.45m is the rule distance; the deployed behavior tree keeps its own
    # conservative 1.60m tactical buffer.
    restart_avoid_distance: float = 1.45
    restart_inset: float = 0.05
    kickoff_expiry_sec: float = 10.0
    set_play_expiry_sec: float = 45.0
    stalemate_duration_sec: float = 30.0
    ball_holding_limit_sec: float = 5.0
    penalty_duration_sec: float = 30.0
    randomize_reset: bool = True
    position_noise: float = 0.08
    action_noise: float = 0.02

    target_x_bins: int = 29
    target_y_bins: int = 19


@dataclass
class RewardConfig:
    goal: float = 10.0
    concede: float = -10.0
    ball_progress: float = 0.40
    checkpoint: float = 0.15
    possession: float = 0.001
    approach_ball: float = 0.015
    successful_pass: float = 0.18
    pressured_pass: float = 0.08
    forward_pass: float = 0.05
    turnover: float = -0.05
    shot: float = 0.0
    shot_on_target: float = 0.20
    bad_shot: float = -0.05
    out_of_bounds: float = -0.10
    illegal_action: float = -0.05
    clustering: float = -0.015
    receiver_facing: float = 0.0030
    keeper_positioning: float = 0.0008
    keeper_save: float = 0.30
    penalty: float = -0.50
    time: float = -0.0005
