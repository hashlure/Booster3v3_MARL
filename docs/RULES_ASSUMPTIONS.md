# Rules profile: booster_3v3_official_2026_07_11_v1

Source: `赛场规则` Feishu export supplied on 2026-07-11.

Implemented normative rules:

- 3v3 adult field: 14 x 9m, 2.6m goal, 0.11m ball radius; the whole ball must cross a boundary.
- Regulation duration 600s using simulation time.
- READY timeout 45s, stable window parameter 5s, SET duration 5s.
- Kick-off rights expire after 10s; throw-in, goal-kick and corner rights expire after 45s. Expiry clears the set play, opens possession to both teams and permits direct goals.
- A goal preserves match time/score, resets positions and gives kick-off to the conceding team.
- Touchline out gives the last-touch opponent a throw-in. Goal-line out gives a corner after defender touch, otherwise a goal kick near x=+/-6, y=+/-2. Corners are inset 0.05m.
- Defender first touch before the awarded side, or a non-exempt defender within 1.45m when the restart is taken, causes a 30s penalty and retake.
- Restart-distance exemptions: moving radially away faster than 0.05m/s, or defending on one's own goal line within the goal width.
- Kick-off direct goals need two distinct touches or expiry. A throw-in/goal-kick/corner kicked directly into the taker's own goal is not scored and gives the opponent a corner.
- 30s with no robot touching the ball triggers a center dropped-ball READY -> SET -> PLAYING restart.
- Standard penalty duration is 30s.

Approximations/engine limits:

- The environment is 2D, so goal height/crossbar and `upDot` fall detection are outside this engine.
- Stable READY transition currently uses the configured READY deadline; automatic early transition needs stable-pose telemetry from the referee layer.
- Warning/caution/sent-off escalation fields are represented by deployment GameController data but are not accumulated by this training simulator.
- Stop/SET path-distance penalties and long-inactivity robot penalties need per-robot pose history and are deferred; policies still receive stopped/state masks and cannot legally act in those phases.

Reset levels:

- `env.reset`: clears score/time/velocities/touches/restarts/penalties/reward state and applies the seed.
- Goal reset: preserves score and match time, resets poses and awards the next kick-off.
- Boundary restart: preserves score/time/robot poses and places only the ball/restart metadata.
- Non-finite state: truncates with `termination_reason=non_finite_state`.
