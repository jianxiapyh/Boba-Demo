import secrets
import threading
import time
from dataclasses import dataclass

from demos.demo2.control import clamp_vector3, legacy_joystick_to_control_vector


@dataclass
class SessionClaim:
    claim_id: int
    token: str
    claimed_at: float
    last_seen: float
    left: tuple = (0.0, 0.0, 0.0)
    right: tuple = (0.0, 0.0, 0.0)


class SessionManager:
    def __init__(self, num_sessions, heartbeat_timeout_s=10.0, input_limit=1.0):
        self.num_sessions = int(num_sessions)
        if self.num_sessions < 1:
            raise ValueError("num_sessions must be positive")
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.input_limit = abs(float(input_limit))
        self._claims = {}
        self._next_claim_id = 1
        self._lock = threading.RLock()

    def _now(self):
        return time.monotonic()

    def _clamp_input(self, value):
        return max(-self.input_limit, min(self.input_limit, float(value)))

    def _clamp_vector(self, vector):
        return tuple(self._clamp_input(value) for value in clamp_vector3(vector))

    def _validate_session_id(self, session_id):
        session_id = int(session_id)
        if session_id < 0 or session_id >= self.num_sessions:
            raise ValueError(
                f"session_id must be in [0, {self.num_sessions - 1}], got {session_id}"
            )
        return session_id

    def _drop_stale_locked(self, now):
        stale = [
            session_id
            for session_id, claim in self._claims.items()
            if now - claim.last_seen > self.heartbeat_timeout_s
        ]
        for session_id in stale:
            del self._claims[session_id]

    def drop_stale(self):
        with self._lock:
            self._drop_stale_locked(self._now())

    def list_sessions(self, token=None):
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            sessions = []
            for session_id in range(self.num_sessions):
                claim = self._claims.get(session_id)
                is_mine = bool(claim is not None and token and claim.token == token)
                sessions.append(
                    {
                        "id": session_id,
                        "available": claim is None,
                        "occupied": claim is not None,
                        "you": is_mine,
                        "claim_id": claim.claim_id if is_mine else None,
                    }
                )
            return sessions

    def claim(self, session_id):
        session_id = self._validate_session_id(session_id)
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            if session_id in self._claims:
                return None
            claim_id = self._next_claim_id
            self._next_claim_id += 1
            token = secrets.token_urlsafe(24)
            self._claims[session_id] = SessionClaim(
                claim_id=claim_id,
                token=token,
                claimed_at=now,
                last_seen=now,
            )
            return {"token": token, "claim_id": claim_id}

    def validate(self, session_id, token):
        session_id = self._validate_session_id(session_id)
        if not token:
            return False
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            claim = self._claims.get(session_id)
            return bool(claim is not None and claim.token == token)

    def heartbeat(self, session_id, token):
        session_id = self._validate_session_id(session_id)
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            claim = self._claims.get(session_id)
            if claim is None or claim.token != token:
                return False
            claim.last_seen = now
            return True

    def release(self, session_id, token):
        session_id = self._validate_session_id(session_id)
        with self._lock:
            claim = self._claims.get(session_id)
            if claim is None or claim.token != token:
                return False
            del self._claims[session_id]
            return True

    def update_input(
        self,
        session_id,
        token,
        x=0.0,
        y=0.0,
        z=0.0,
        *,
        left=None,
        right=None,
        dx=None,
        dy=None,
    ):
        session_id = self._validate_session_id(session_id)
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            claim = self._claims.get(session_id)
            if claim is None or claim.token != token:
                return False
            if left is not None or right is not None:
                claim.left = self._clamp_vector(left)
                claim.right = self._clamp_vector(right)
            elif dx is not None or dy is not None:
                claim.left = self._clamp_vector(
                    legacy_joystick_to_control_vector(
                        0.0 if dx is None else dx,
                        0.0 if dy is None else dy,
                    )
                )
                claim.right = (0.0, 0.0, 0.0)
            else:
                claim.left = self._clamp_vector((x, y, z))
                claim.right = (0.0, 0.0, 0.0)
            claim.last_seen = now
            return True

    def snapshot_sessions(self):
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            return {
                session_id: {
                    "claim_id": claim.claim_id,
                    "left": claim.left,
                    "right": claim.right,
                }
                for session_id, claim in self._claims.items()
            }

    def snapshot_inputs(self):
        return {
            session_id: claim["left"]
            for session_id, claim in self.snapshot_sessions().items()
        }

    def occupied_ids(self):
        now = self._now()
        with self._lock:
            self._drop_stale_locked(now)
            return sorted(self._claims.keys())
