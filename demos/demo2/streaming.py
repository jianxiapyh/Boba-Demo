import threading
import time

import numpy as np


class MjpegFrameStore:
    def __init__(self, jpeg_quality=80):
        self.jpeg_quality = int(jpeg_quality)
        self._frames = {}
        self._condition = threading.Condition()

    def publish_rgb(self, session_id, rgb):
        import cv2

        session_id = int(session_id)
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H,W,3), got {rgb.shape}")
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode MJPEG frame")
        jpeg = encoded.tobytes()
        with self._condition:
            self._frames[session_id] = (jpeg, time.monotonic())
            self._condition.notify_all()

    def latest(self, session_id):
        session_id = int(session_id)
        with self._condition:
            frame = self._frames.get(session_id)
            return frame[0] if frame is not None else None

    def mjpeg_generator(self, session_id, stop_fn=None, idle_timeout_s=30.0):
        session_id = int(session_id)
        last_updated_at = None
        last_activity = time.monotonic()
        while True:
            if stop_fn is not None and stop_fn():
                return
            with self._condition:
                self._condition.wait(timeout=0.5)
                frame = self._frames.get(session_id)
                if frame is None:
                    if time.monotonic() - last_activity > idle_timeout_s:
                        return
                    continue
                jpeg, updated_at = frame
                if updated_at == last_updated_at:
                    if time.monotonic() - last_activity > idle_timeout_s:
                        return
                    continue
                last_updated_at = updated_at
                last_activity = updated_at
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
