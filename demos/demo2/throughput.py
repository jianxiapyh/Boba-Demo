"""Rolling throughput measurement for the Demo 2 public display."""

from __future__ import annotations

from collections import deque
import math
import time


class AggregateThroughputMeter:
    """Measure completed batched simulation instances per wall-clock second.

    Demo 2 advances every instance once per displayed runtime frame. Measuring
    the interval between loop starts therefore captures rendering, simulation,
    display synchronization, and intentional frame pacing in one number.
    """

    def __init__(self, batch_size: int, window_size: int = 30):
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        self._intervals: deque[float] = deque(maxlen=self.window_size)
        self._last_timestamp: float | None = None

    def sample(self, timestamp: float | None = None) -> float:
        """Record a loop start and return rolling aggregate instances/second."""

        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None:
            interval = now - self._last_timestamp
            if interval > 0.0:
                self._intervals.append(interval)
        self._last_timestamp = now
        if not self._intervals:
            return 0.0
        elapsed = sum(self._intervals)
        return self.batch_size * len(self._intervals) / elapsed
