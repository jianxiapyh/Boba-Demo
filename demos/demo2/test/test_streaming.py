import time
import threading
import unittest

import cv2
import numpy as np

from demos.demo2.session_manager import SessionManager
from demos.demo2.streaming import MjpegFrameStore
from demos.demo2_server import Sock, create_app, start_flask_server

try:
    from simple_websocket import Client
except ImportError:
    Client = None


class SlowEncoderStore(MjpegFrameStore):
    def __init__(self):
        self.encode_started = threading.Event()
        self.allow_encode = threading.Event()
        super().__init__(jpeg_quality=80, encode_workers=1)

    def _encode_bgr(self, bgr):
        self.encode_started.set()
        if not self.allow_encode.wait(timeout=2.0):
            raise RuntimeError("test encoder gate timed out")
        return super()._encode_bgr(bgr)


class FrameStoreTest(unittest.TestCase):
    @staticmethod
    def frame(value):
        return np.full((48, 64, 3), value, dtype=np.uint8)

    def test_rejects_invalid_encoder_configuration(self):
        with self.assertRaisesRegex(ValueError, "jpeg_quality"):
            MjpegFrameStore(jpeg_quality=0)
        with self.assertRaisesRegex(ValueError, "encode_workers"):
            MjpegFrameStore(encode_workers=-1)

    def test_mjpeg_and_websocket_source_share_the_exact_jpeg(self):
        store = MjpegFrameStore(jpeg_quality=80)
        sequence = store.publish_rgb(3, self.frame(120))
        jpeg, packet_sequence, _ = store.latest_packet(3)

        self.assertEqual(sequence, 1)
        self.assertEqual(packet_sequence, 1)
        self.assertEqual(store.latest(3), jpeg)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (48, 64, 3))

        generator = store.mjpeg_generator(3)
        parts = []
        reader = threading.Thread(target=lambda: parts.append(next(generator)))
        try:
            reader.start()
            deadline = time.monotonic() + 1.0
            while store.requested_sessions([3]) != [3] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(store.requested_sessions([3]), [3])
            store.publish_rgb(3, self.frame(121))
            reader.join(timeout=1.0)
            self.assertFalse(reader.is_alive())
            part = parts[0]
        finally:
            generator.close()
            reader.join(timeout=1.0)

        jpeg = store.latest(3)
        self.assertIn(b"Content-Type: image/jpeg\r\n", part)
        self.assertIn(f"Content-Length: {len(jpeg)}\r\n".encode("ascii"), part)
        self.assertIn(b"Cache-Control: no-store\r\n", part)
        self.assertTrue(part.endswith(jpeg + b"\r\n"))

    def test_async_encoder_replaces_stale_pending_frames(self):
        store = SlowEncoderStore()
        try:
            store.submit_rgb(0, self.frame(0))
            self.assertTrue(store.encode_started.wait(timeout=1.0))
            for value in range(1, 6):
                store.submit_rgb(0, self.frame(value))
            store.allow_encode.set()

            deadline = time.monotonic() + 2.0
            while store.stats()["encoded_frames"] < 2 and time.monotonic() < deadline:
                time.sleep(0.01)

            stats = store.stats()
            self.assertEqual(stats["submitted_frames"], 6)
            self.assertEqual(stats["encoded_frames"], 2)
            self.assertEqual(stats["replaced_frames"], 4)
            jpeg, sequence, _ = store.latest_packet(0)
            self.assertEqual(sequence, 2)
            decoded = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertAlmostEqual(float(decoded.mean()), 5.0, delta=2.0)
        finally:
            store.allow_encode.set()
            store.close()

    def test_async_encoder_preserves_synchronous_jpeg_bytes(self):
        frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        synchronous = MjpegFrameStore(jpeg_quality=80)
        asynchronous = MjpegFrameStore(jpeg_quality=80, encode_workers=1)
        try:
            synchronous.publish_rgb(0, frame)
            asynchronous.submit_rgb(0, frame)
            packet = asynchronous.wait_for_frame(0, timeout_s=2.0)

            self.assertIsNotNone(packet)
            self.assertEqual(packet[0], synchronous.latest(0))
        finally:
            asynchronous.close()

    def test_frame_requests_are_per_consumer_and_consumed_by_publish(self):
        store = MjpegFrameStore(jpeg_quality=80)
        first_consumer = object()
        second_consumer = object()

        self.assertTrue(store.request_frame(2, first_consumer))
        self.assertTrue(store.request_frame(2, second_consumer))
        self.assertEqual(store.requested_sessions([0, 1, 2, 3]), [2])

        store.cancel_frame_request(2, first_consumer)
        self.assertEqual(store.requested_sessions([2]), [2])
        store.publish_rgb(2, self.frame(90))
        self.assertEqual(store.requested_sessions([2]), [])
        self.assertEqual(store.stats()["requested_sessions"], 0)

        store.close()
        self.assertFalse(store.request_frame(2, first_consumer))


@unittest.skipIf(Sock is None or Client is None, "Flask-Sock is not installed")
class WebSocketStreamIntegrationTest(unittest.TestCase):
    @staticmethod
    def wait_until_requested(store, session_id, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if store.requested_sessions([session_id]) == [session_id]:
                return
            time.sleep(0.01)
        raise AssertionError(f"session {session_id} did not request a frame")

    def test_websocket_delivers_identical_jpeg_and_waits_for_ack(self):
        manager = SessionManager(num_sessions=1)
        token = manager.claim(0)["token"]
        store = MjpegFrameStore(jpeg_quality=80)
        app = create_app(manager, store)
        server, thread = start_flask_server(app, "127.0.0.1", 0)
        port = server.socket.getsockname()[1]
        socket = None
        try:
            store.publish_rgb(0, np.full((48, 64, 3), 5, dtype=np.uint8))
            socket = Client.connect(
                f"ws://127.0.0.1:{port}/ws/stream/0?token={token}"
            )
            self.wait_until_requested(store, 0)
            self.assertIsNone(socket.receive(timeout=0.1))

            store.publish_rgb(0, np.full((48, 64, 3), 40, dtype=np.uint8))
            first_jpeg = store.latest(0)
            self.assertEqual(socket.receive(timeout=2.0), first_jpeg)
            self.assertEqual(store.requested_sessions([0]), [])
            self.assertIsNone(socket.receive(timeout=0.1))

            socket.send("next")
            self.wait_until_requested(store, 0)
            store.publish_rgb(0, np.full((48, 64, 3), 180, dtype=np.uint8))
            second_jpeg = store.latest(0)
            self.assertEqual(socket.receive(timeout=2.0), second_jpeg)
        finally:
            if socket is not None:
                socket.close()
            server.shutdown()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
