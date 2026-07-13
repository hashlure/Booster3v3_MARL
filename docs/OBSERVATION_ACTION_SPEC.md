# Deployment observation/action contract

## Actor observation (59 float32 values)

All coordinates are in the observing team's frame: own goal x=-7, opponent goal
x=+7, attack direction +x. No simulator-only velocity is present.

Order:

1. Self (6): normalized x/y, cos/sin heading, active mask, penalty time.
2. Ball (5): relative x/y, absolute x/y, validity mask.
3. Two teammates (2x5): relative x/y, cos/sin heading, active mask.
4. Three opponents (3x5): relative x/y, cos/sin heading, active mask.
5. GameState one-hot (5).
6. SetPlay one-hot (7).
7. Kicking team one-hot, remaining-time fraction, score difference (5).
8. Previous PlannerIntent one-hot (6).

The simulator sets validity to one. The MyAgent adapter must zero-fill and set
validity to zero when PlayContext data is missing or stale. BallState/RobotState
timestamps must be positive and in the behavior-tree tick clock domain.

## Centralized critic state (64 float32 values)

Includes six global robot states and true simulator velocities, ball position and
velocity, GameState, SetPlay, kicking team, scores and remaining time. It is
training-only and must never be fed to the deployed Actor.

## Action (Discrete 31)

```text
0 hold
1..8 relative move directions
9..12 dribble goal/left/right/center
13..15 pass to teammate 1/2/3
16..18 shoot center/left/right
19 guard goal
20..21 support left/right
22 shoot_best_gap (continuous geometry chooses the exact goal-line y)
23..30 precision move (0.45m) in eight directions
```

Legacy IDs 0-21 retain their original meaning. `shoot_best_gap` subtracts the
expanded angular shadows of all opponents from the usable goal mouth and aims
at the centre of the widest remaining interval.

The behavior tree remains responsible for target validation, obstacle avoidance,
kick range, fall recovery, referee safety and final RobotCommand dispatch.
