import json
import tempfile
import unittest
from pathlib import Path

from navvis_recon.floor_estimator import (
    TraceSample,
    read_trace_csv,
    refined_floor_estimator,
    simple_floor_estimator,
    to_official_json,
)


class FloorEstimatorTest(unittest.TestCase):
    def test_revisited_floor_has_two_nanosecond_ranges(self):
        samples = []
        timestamp = 0
        for height in (0.0, 3.0, 0.0):
            for _ in range(40):
                samples.append(TraceSample(timestamp, 0.0, 0.0, height))
                timestamp += 200_000_000

        floors = simple_floor_estimator(samples)

        self.assertEqual(len(floors), 2)
        self.assertEqual(len(floors[0].time_ranges), 2)
        self.assertEqual(floors[0].time_ranges[0].start_ns, 0)
        self.assertEqual(floors[0].time_ranges[1].start_ns, 16_000_000_000)

    def test_refined_output_uses_official_top_level_schema(self):
        samples = [
            TraceSample(
                index * 200_000_000,
                0.0,
                0.0,
                (0.0 if index < 40 else 3.0) + (0.04 if index % 2 else -0.04),
            )
            for index in range(80)
        ]

        payload = to_official_json(refined_floor_estimator(samples))

        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertEqual(set(payload[0]), {"time_ranges", "z_min", "z_max"})
        self.assertIsInstance(payload[0]["time_ranges"][0][0], int)
        json.dumps(payload)

    def test_trace_reader_rejects_non_increasing_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                "nsecs,  x, y, z\n"
                "100, 0, 0, 1\n"
                "100, 0, 0, 1\n"
            )

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                read_trace_csv(path)

    def test_trace_reader_rejects_empty_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text("nsecs,  x, y, z\n")

            with self.assertRaisesRegex(ValueError, "no usable samples"):
                read_trace_csv(path)


if __name__ == "__main__":
    unittest.main()
