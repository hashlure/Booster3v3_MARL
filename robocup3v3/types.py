"""Dependency-free state and action contracts.

Names and enum values intentionally match ``MyAgent/src/soccer_framework/types.py``
where the contracts overlap.  The simulator keeps velocity and rule bookkeeping
in additional fields that are not exposed by ``PlayContext``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple


class Team(str, Enum):
    BLUE = "blue"
    RED = "red"

    @property
    def opponent(self) -> "Team":
        return Team.RED if self is Team.BLUE else Team.BLUE

    @property
    def team_id(self) -> int:
        return 1 if self is Team.BLUE else 2

    @property
    def attack_sign(self) -> float:
        return 1.0 if self is Team.BLUE else -1.0


class GameState(str, Enum):
    INITIAL = "INITIAL"
    READY = "READY"
    SET = "SET"
    PLAYING = "PLAYING"
    FINISHED = "FINISHED"


class GamePhase(str, Enum):
    NORMAL = "NORMAL"
    PENALTY_SHOOT_OUT = "PENALTY_SHOOT_OUT"
    EXTRA_TIME = "EXTRA_TIME"
    TIMEOUT = "TIMEOUT"


class SetPlay(str, Enum):
    NONE = "NONE"
    DIRECT_FREE_KICK = "DIRECT_FREE_KICK"
    INDIRECT_FREE_KICK = "INDIRECT_FREE_KICK"
    PENALTY_KICK = "PENALTY_KICK"
    THROW_IN = "THROW_IN"
    GOAL_KICK = "GOAL_KICK"
    CORNER_KICK = "CORNER_KICK"


class Penalty(str, Enum):
    NONE = "NONE"
    ILLEGAL_POSITIONING = "ILLEGAL_POSITIONING"
    MOTION_IN_SET = "MOTION_IN_SET"
    LOCAL_GAME_STUCK = "LOCAL_GAME_STUCK"
    INCAPABLE_ROBOT = "INCAPABLE_ROBOT"
    PICKED_UP = "PICKED_UP"
    BALL_HOLDING = "BALL_HOLDING"
    LEAVING_THE_FIELD = "LEAVING_THE_FIELD"
    PLAYING_WITH_ARMS_HANDS = "PLAYING_WITH_ARMS_HANDS"
    PUSHING = "PUSHING"
    SENT_OFF = "SENT_OFF"
    SUBSTITUTE = "SUBSTITUTE"


class PlannerIntent(IntEnum):
    HOLD = 0
    MOVE = 1
    DRIBBLE = 2
    PASS = 3
    SHOOT = 4
    GUARD = 5


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class PlannerAction:
    intent: PlannerIntent = PlannerIntent.HOLD
    target_x: float = 0.0
    target_y: float = 0.0
    kick_target_x: float = 0.0
    kick_target_y: float = 0.0


@dataclass
class RobotSimState:
    team: Team
    player_id: int
    pose: Pose2D = field(default_factory=Pose2D)
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    radius: float = 0.28
    penalty: Penalty = Penalty.NONE
    penalty_remaining: float = 0.0
    possession_time: float = 0.0
    last_action: PlannerAction = field(default_factory=PlannerAction)

    @property
    def name(self) -> str:
        return "%s_%d" % (self.team.value, self.player_id)

    @property
    def active(self) -> bool:
        return self.penalty is Penalty.NONE and self.penalty_remaining <= 0.0


@dataclass
class BallSimState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.11
    last_touch_team: Optional[Team] = None
    last_touch_player: Optional[int] = None


@dataclass
class RuleEvent:
    kind: str
    team: Optional[Team] = None
    player_id: Optional[int] = None
    set_play: SetPlay = SetPlay.NONE
    detail: str = ""


@dataclass
class MatchState:
    robots: Dict[str, RobotSimState]
    ball: BallSimState = field(default_factory=BallSimState)
    game_state: GameState = GameState.INITIAL
    game_phase: GamePhase = GamePhase.NORMAL
    set_play: SetPlay = SetPlay.NONE
    kicking_team: Optional[Team] = None
    stopped: bool = True
    score: Dict[Team, int] = field(
        default_factory=lambda: {Team.BLUE: 0, Team.RED: 0}
    )
    elapsed: float = 0.0
    state_elapsed: float = 0.0
    step_count: int = 0
    packet_number: int = 0
    restart_touches: List[Tuple[Team, int]] = field(default_factory=list)
    restart_origin: SetPlay = SetPlay.NONE
    direct_goal_allowed: bool = True
    restart_pending: bool = False
    restart_awarded_team: Optional[Team] = None
    restart_ball_x: float = 0.0
    restart_ball_y: float = 0.0
    restart_started_at: float = 0.0
    restart_expires_at: float = 0.0
    last_touch_at: float = 0.0
    previous_ball_x: float = 0.0
    previous_ball_y: float = 0.0
    events: List[RuleEvent] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    termination_reason: str = ""

    def team_robots(self, team: Team) -> List[RobotSimState]:
        return [self.robots["%s_%d" % (team.value, i)] for i in (1, 2, 3)]
