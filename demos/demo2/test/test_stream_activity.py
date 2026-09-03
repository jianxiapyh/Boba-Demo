import unittest

from demos.demo2.stream_activity import ActivityStreamPolicy


def claim(claim_id=1, left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0)):
    return {"claim_id": claim_id, "left": left, "right": right}


class ActivityStreamPolicyTest(unittest.TestCase):
    def test_initial_frame_then_idle_until_action(self):
        policy = ActivityStreamPolicy(2)
        policy.observe_inputs({0: claim()}, now=0.0)

        self.assertEqual(policy.eligible_sessions([0], now=0.0), [0])
        policy.mark_published([0], now=0.0)
        self.assertEqual(policy.snapshot()[0]["state"], policy.IDLE)
        self.assertEqual(policy.eligible_sessions([0], now=0.1), [])

        policy.observe_inputs({0: claim(left=(1.0, 0.0, 0.0))}, now=0.2)
        self.assertEqual(policy.snapshot()[0]["state"], policy.ACTIVE)
        self.assertEqual(policy.eligible_sessions([0], now=0.2), [0])

    def test_release_streams_until_five_stable_frames_and_one_final_frame(self):
        policy = ActivityStreamPolicy(
            1,
            motion_threshold=0.01,
            stable_frames=5,
            full_rate_settle_s=1.0,
        )
        policy.observe_inputs({0: claim(left=(1.0, 0.0, 0.0))}, now=0.0)
        policy.observe_inputs({0: claim()}, now=0.1)
        self.assertEqual(policy.snapshot()[0]["state"], policy.SETTLING)

        policy.observe_motion({0: 0.1}, now=0.1)
        for frame in range(4):
            policy.observe_motion({0: 0.005}, now=0.2 + frame * 0.03)
            self.assertEqual(policy.snapshot()[0]["state"], policy.SETTLING)

        policy.observe_motion({0: 0.005}, now=0.32)
        self.assertEqual(policy.snapshot()[0]["state"], policy.FINAL)
        self.assertEqual(policy.eligible_sessions([0], now=0.32), [0])
        policy.mark_published([0], now=0.32)
        self.assertEqual(policy.snapshot()[0]["state"], policy.IDLE)

    def test_long_motion_switches_to_low_rate_recovery(self):
        policy = ActivityStreamPolicy(
            1,
            motion_threshold=0.01,
            stable_frames=5,
            full_rate_settle_s=1.0,
            recovery_fps=2.0,
        )
        policy.observe_inputs({0: claim(left=(0.0, 1.0, 0.0))}, now=0.0)
        policy.observe_inputs({0: claim()}, now=0.1)
        policy.observe_motion({0: 0.1}, now=1.1)

        self.assertEqual(policy.snapshot()[0]["state"], policy.RECOVERY)
        self.assertEqual(policy.eligible_sessions([0], now=1.1), [0])
        policy.mark_published([0], now=1.1)
        self.assertEqual(policy.eligible_sessions([0], now=1.59), [])
        self.assertEqual(policy.eligible_sessions([0], now=1.6), [0])

    def test_new_action_interrupts_settling_or_recovery(self):
        policy = ActivityStreamPolicy(1, full_rate_settle_s=0.0)
        policy.observe_inputs({0: claim(left=(1.0, 0.0, 0.0))}, now=0.0)
        policy.observe_inputs({0: claim()}, now=0.1)
        policy.observe_motion({0: 1.0}, now=0.1)
        self.assertEqual(policy.snapshot()[0]["state"], policy.RECOVERY)

        policy.observe_inputs({0: claim(right=(0.0, 0.0, -1.0))}, now=0.2)
        self.assertEqual(policy.snapshot()[0]["state"], policy.ACTIVE)
        self.assertEqual(policy.snapshot()[0]["stable_frames"], 0)

    def test_release_and_reclaim_reset_state(self):
        policy = ActivityStreamPolicy(1)
        policy.observe_inputs({0: claim(claim_id=1)}, now=0.0)
        policy.mark_published([0], now=0.0)
        policy.observe_inputs({}, now=0.1)
        self.assertEqual(policy.snapshot()[0]["state"], policy.UNCLAIMED)

        policy.observe_inputs({0: claim(claim_id=2)}, now=0.2)
        self.assertEqual(policy.snapshot()[0]["state"], policy.INITIAL)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ActivityStreamPolicy(0)
        with self.assertRaises(ValueError):
            ActivityStreamPolicy(1, motion_threshold=-1)
        with self.assertRaises(ValueError):
            ActivityStreamPolicy(1, stable_frames=0)
        with self.assertRaises(ValueError):
            ActivityStreamPolicy(1, full_rate_settle_s=-1)
        with self.assertRaises(ValueError):
            ActivityStreamPolicy(1, recovery_fps=0)


if __name__ == "__main__":
    unittest.main()
