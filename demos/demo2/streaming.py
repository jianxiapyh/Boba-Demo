import threading
import time
from collections import OrderedDict

import numpy as np


class MjpegFrameStore:
    """Keep only each session's newest JPEG and optionally encode off-thread."""

    def __init__(self, jpeg_quality=80, encode_workers=0):
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")

        self.encode_workers = int(encode_workers)
        if self.encode_workers < 0:
            raise ValueError("encode_workers must be non-negative")

        self._frames = {}
        self._sequences = {}
        self._frame_condition = threading.Condition()

        self._pending = OrderedDict()
        self._frame_requests = {}
        self._pending_condition = threading.Condition()
        self._closed = False
        self._encoder_error = None
        self._submitted_frames = 0
        self._replaced_frames = 0
        self._encoded_frames = 0
        self._workers = []
        for worker_idx in range(self.encode_workers):
            worker = threading.Thread(
                target=self._encoder_worker,
                name=f"demo2-jpeg-{worker_idx}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    @staticmethod
    def _prepare_bgr(rgb):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H,W,3), got {rgb.shape}")
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def _encode_bgr(self, bgr):
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG frame")
        return encoded.tobytes()

    def _store_jpeg(self, session_id, jpeg):
        session_id = int(session_id)
        with self._frame_condition:
            sequence = self._sequences.get(session_id, 0) + 1
            self._sequences[session_id] = sequence
            self._frames[session_id] = (jpeg, sequence, time.monotonic())
            self._encoded_frames += 1
            self._frame_condition.notify_all()
        return sequence

    def publish_rgb(self, session_id, rgb):
        """Synchronously encode and publish a frame for compatibility callers."""

        session_id = int(session_id)
        self._consume_frame_requests(session_id)
        bgr = self._prepare_bgr(rgb)
        jpeg = self._encode_bgr(bgr)
        return self._store_jpeg(session_id, jpeg)

    def submit_rgb(self, session_id, rgb):
        """Queue the newest raw frame, replacing an older unencoded frame."""

        if not self._workers:
            return self.publish_rgb(session_id, rgb)

        session_id = int(session_id)
        bgr = self._prepare_bgr(rgb)
        with self._pending_condition:
            if self._encoder_error is not None:
                raise RuntimeError("Demo 2 JPEG encoder worker failed") from self._encoder_error
            if self._closed:
                raise RuntimeError("Demo 2 JPEG frame store is closed")
            self._frame_requests.pop(session_id, None)
            self._submitted_frames += 1
            if session_id in self._pending:
                self._replaced_frames += 1
            self._pending[session_id] = bgr
            self._pending_condition.notify()
        return None

    def _encoder_worker(self):
        while True:
            with self._pending_condition:
                self._pending_condition.wait_for(
                    lambda: self._closed or bool(self._pending)
                )
                if self._closed and not self._pending:
                    return
                session_id, bgr = self._pending.popitem(last=False)

            try:
                jpeg = self._encode_bgr(bgr)
                self._store_jpeg(session_id, jpeg)
            except Exception as exc:
                with self._pending_condition:
                    self._encoder_error = exc
                    self._closed = True
                    self._pending.clear()
                    self._pending_condition.notify_all()
                with self._frame_condition:
                    self._frame_condition.notify_all()
                return

    def request_frame(self, session_id, consumer_id):
        """Mark a consumer ready for the session's next simulation frame."""

        session_id = int(session_id)
        with self._pending_condition:
            if self._closed:
                return False
            consumers = self._frame_requests.setdefault(session_id, set())
            consumers.add(consumer_id)
            return True

    def cancel_frame_request(self, session_id, consumer_id):
        session_id = int(session_id)
        with self._pending_condition:
            consumers = self._frame_requests.get(session_id)
            if consumers is None:
                return
            consumers.discard(consumer_id)
            if not consumers:
                self._frame_requests.pop(session_id, None)

    def requested_sessions(self, candidate_session_ids):
        """Return candidates with at least one consumer waiting for a frame."""

        with self._pending_condition:
            requested = set(self._frame_requests)
        return [
            int(session_id)
            for session_id in candidate_session_ids
            if int(session_id) in requested
        ]

    def _consume_frame_requests(self, session_id):
        with self._pending_condition:
            self._frame_requests.pop(int(session_id), None)

    def latest(self, session_id):
        packet = self.latest_packet(session_id)
        return packet[0] if packet is not None else None

    def latest_packet(self, session_id):
        session_id = int(session_id)
        with self._frame_condition:
            return self._frames.get(session_id)

    def wait_for_frame(
        self,
        session_id,
        *,
        after_sequence=None,
        timeout_s=0.5,
        stop_fn=None,
    ):
        """Return the newest (jpeg, sequence, timestamp), never an old queue."""

        session_id = int(session_id)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._frame_condition:
            while True:
                frame = self._frames.get(session_id)
                if frame is not None and frame[1] != after_sequence:
                    return frame
                if self._encoder_error is not None:
                    raise RuntimeError("Demo 2 JPEG encoder worker failed") from self._encoder_error
                if stop_fn is not None and stop_fn():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._frame_condition.wait(timeout=min(remaining, 0.25))

    def stats(self):
        with self._pending_condition:
            submitted = self._submitted_frames
            replaced = self._replaced_frames
            pending = len(self._pending)
            requested_sessions = len(self._frame_requests)
        with self._frame_condition:
            encoded = self._encoded_frames
        return {
            "submitted_frames": submitted,
            "encoded_frames": encoded,
            "replaced_frames": replaced,
            "pending_frames": pending,
            "requested_sessions": requested_sessions,
        }

    def close(self, timeout_s=2.0):
        with self._pending_condition:
            self._closed = True
            self._pending.clear()
            self._frame_requests.clear()
            self._pending_condition.notify_all()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

    def mjpeg_generator(self, session_id, stop_fn=None, idle_timeout_s=30.0):
        session_id = int(session_id)
        consumer_id = object()
        current_packet = self.latest_packet(session_id)
        last_sequence = current_packet[1] if current_packet is not None else None
        last_activity = time.monotonic()
        try:
            while True:
                if not self.request_frame(session_id, consumer_id):
                    return
                packet = self.wait_for_frame(
                    session_id,
                    after_sequence=last_sequence,
                    timeout_s=0.5,
                    stop_fn=stop_fn,
                )
                if packet is None:
                    if stop_fn is not None and stop_fn():
                        return
                    if time.monotonic() - last_activity > idle_timeout_s:
                        return
                    continue

                jpeg, sequence, updated_at = packet
                last_sequence = sequence
                last_activity = updated_at
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n".encode("ascii")
                    + b"Cache-Control: no-store\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
        finally:
            self.cancel_frame_request(session_id, consumer_id)
