import unittest

from demos.demo2.throughput import AggregateThroughputMeter


class AggregateThroughputMeterTest(unittest.TestCase):
    def test_reports_batched_instances_per_second(self):
        meter = AggregateThroughputMeter(batch_size=100, window_size=2)

        self.assertEqual(meter.sample(10.0), 0.0)
        self.assertAlmostEqual(meter.sample(10.05), 2000.0)
        self.assertAlmostEqual(meter.sample(10.10), 2000.0)
        self.assertAlmostEqual(meter.sample(10.20), 100 * 2 / 0.15)

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            AggregateThroughputMeter(batch_size=0)
        with self.assertRaisesRegex(ValueError, "window_size"):
            AggregateThroughputMeter(batch_size=1, window_size=0)


if __name__ == "__main__":
    unittest.main()
