import math


class ActivityStreamPolicy:
    """Choose which claimed sessions need a newly rendered phone frame."""

    UNCLAIMED = "unclaimed"
    INITIAL = "initial"
    IDLE = "idle"
    ACTIVE = "active"
    SETTLING = "settling"
    RECOVERY = "recovery"
    FINAL = "final"

    def __init__(
        self,
        num_sessions,
        *,
        motion_threshold=1e-4,
        stable_frames=5,
        full_rate_settle_s=1.0,
        recovery_fps=2.0,
        input_epsilon=1e-6,
    ):
        self.num_sessions = int(num_sessions)
        self.motion_threshold = float(motion_threshold)
        self.stable_frames = int(stable_frames)
        self.full_rate_settle_s = float(full_rate_settle_s)
        self.recovery_fps = float(recovery_fps)
        self.input_epsilon = float(input_epsilon)

        if self.num_sessions < 1:
            raise ValueError("num_sessions must be positive")
        if not math.isfinite(self.motion_threshold) or self.motion_threshold < 0:
            raise ValueError("motion_threshold must be finite and non-negative")
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        if not math.isfinite(self.full_rate_settle_s) or self.full_rate_settle_s < 0:
            raise ValueError("full_rate_settle_s must be finite and non-negative")
        if not math.isfinite(self.recovery_fps) or self.recovery_fps <= 0:
            raise ValueError("recovery_fps must be finite and positive")
        if not math.isfinite(self.input_epsilon) or self.input_epsilon < 0:
            raise ValueError("input_epsilon must be finite and non-negative")

        self._states = [self.UNCLAIMED] * self.num_sessions
        self._claim_ids = [None] * self.num_sessions
        self._stable_counts = [0] * self.num_sessions
        self._settle_started_at = [None] * self.num_sessions
        self._next_recovery_frame_at = [0.0] * self.num_sessions

    def _valid_session_id(self, session_id):
        session_id = int(session_id)
        return 0 <= session_id < self.num_sessions

    def _input_is_active(self, claim):
        for hand in ("left", "right"):
            values = claim.get(hand, (0.0, 0.0, 0.0))
            if isinstance(values, dict):
                values = (values.get("x", 0.0), values.get("y", 0.0), values.get("z", 0.0))
            if any(abs(float(value)) > self.input_epsilon for value in values):
                return True
        return False

    def observe_inputs(self, session_snapshot, now):
        """Advance claim/action states before the current simulation step."""

        now = float(now)
        snapshot = {
            int(session_id): claim
            for session_id, claim in session_snapshot.items()
            if self._valid_session_id(session_id)
        }
        for session_id in range(self.num_sessions):
            claim = snapshot.get(session_id)
            if claim is None:
                self._states[session_id] = self.UNCLAIMED
                self._claim_ids[session_id] = None
                self._stable_counts[session_id] = 0
                self._settle_started_at[session_id] = None
                self._next_recovery_frame_at[session_id] = 0.0
                continue

            claim_id = int(claim.get("claim_id", 1))
            input_active = self._input_is_active(claim)
            if self._claim_ids[session_id] != claim_id:
                self._claim_ids[session_id] = claim_id
                self._states[session_id] = self.ACTIVE if input_active else self.INITIAL
                self._stable_counts[session_id] = 0
                self._settle_started_at[session_id] = None
                self._next_recovery_frame_at[session_id] = now
                continue

            state = self._states[session_id]
            if input_active:
                self._states[session_id] = self.ACTIVE
                self._stable_counts[session_id] = 0
                self._settle_started_at[session_id] = None
                self._next_recovery_frame_at[session_id] = now
            elif state == self.ACTIVE:
                self._states[session_id] = self.SETTLING
                self._stable_counts[session_id] = 0
                self._settle_started_at[session_id] = now
                self._next_recovery_frame_at[session_id] = now

    def motion_session_ids(self):
        return [
            session_id
            for session_id, state in enumerate(self._states)
            if state in (self.SETTLING, self.RECOVERY)
        ]

    def observe_motion(self, motion_by_session, now):
        """Update settling state from RMS object displacement per simulation frame."""

        now = float(now)
        for raw_session_id, raw_motion in motion_by_session.items():
            session_id = int(raw_session_id)
            if not self._valid_session_id(session_id):
                continue
            state = self._states[session_id]
            if state not in (self.SETTLING, self.RECOVERY):
                continue

            motion = float(raw_motion)
            if math.isfinite(motion) and motion <= self.motion_threshold:
                self._stable_counts[session_id] += 1
            else:
                self._stable_counts[session_id] = 0

            if self._stable_counts[session_id] >= self.stable_frames:
                self._states[session_id] = self.FINAL
                continue

            settle_started_at = self._settle_started_at[session_id]
            if (
                state == self.SETTLING
                and settle_started_at is not None
                and now - settle_started_at >= self.full_rate_settle_s
            ):
                self._states[session_id] = self.RECOVERY
                self._next_recovery_frame_at[session_id] = now

    def eligible_sessions(self, requested_session_ids, now):
        """Filter transport demand through action and settling state."""

        now = float(now)
        eligible = []
        for raw_session_id in requested_session_ids:
            session_id = int(raw_session_id)
            if not self._valid_session_id(session_id):
                continue
            state = self._states[session_id]
            if state in (self.INITIAL, self.ACTIVE, self.SETTLING, self.FINAL):
                eligible.append(session_id)
            elif (
                state == self.RECOVERY
                and now >= self._next_recovery_frame_at[session_id]
            ):
                eligible.append(session_id)
        return eligible

    def mark_published(self, session_ids, now):
        """Record that the current rendered frame satisfied phone demand."""

        now = float(now)
        recovery_interval = 1.0 / self.recovery_fps
        for raw_session_id in session_ids:
            session_id = int(raw_session_id)
            if not self._valid_session_id(session_id):
                continue
            state = self._states[session_id]
            if state in (self.INITIAL, self.FINAL):
                self._states[session_id] = self.IDLE
            elif state == self.RECOVERY:
                self._next_recovery_frame_at[session_id] = now + recovery_interval

    def snapshot(self):
        return [
            {
                "state": self._states[session_id],
                "claim_id": self._claim_ids[session_id],
                "stable_frames": self._stable_counts[session_id],
            }
            for session_id in range(self.num_sessions)
        ]
