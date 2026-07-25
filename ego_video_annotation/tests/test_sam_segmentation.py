from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.pipeline import build_parser  # noqa: E402
from egoanno.sam_segmentation import sam_split_merge  # noqa: E402


class SaMSplitMergeTests(unittest.TestCase):
    def test_repeated_action_is_one_cluster_but_multiple_contiguous_runs(self) -> None:
        action_a = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (20, 1))
        action_b = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (20, 1))
        features = np.vstack([action_a, action_b, action_a])

        result = sam_split_merge(features, action_count=2, delta=0.3, temporal_lambda=0.001)

        self.assertEqual(result["diagnostics"]["final_distinct_cluster_count"], 2)
        self.assertEqual(result["runs"], [
            {"start": 0, "end": 20, "cluster_id": 1},
            {"start": 20, "end": 40, "cluster_id": 2},
            {"start": 40, "end": 60, "cluster_id": 1},
        ])
        covered = [
            index
            for run in result["runs"]
            for index in range(run["start"], run["end"])
        ]
        self.assertEqual(covered, list(range(60)))

    def test_invalid_known_action_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sam-action-count"):
            sam_split_merge(np.ones((5, 3), dtype=np.float32), action_count=5)

    def test_pipeline_exposes_sam_without_changing_default_method(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["--video", "x.mp4", "--output", "out"])
        sam = parser.parse_args([
            "--video", "x.mp4", "--output", "out",
            "--segmentation-method", "sam", "--sam-action-count", "12",
        ])
        self.assertEqual(default.segmentation_method, "ours")
        self.assertEqual(sam.segmentation_method, "sam")
        self.assertEqual(sam.sam_action_count, 12)
        self.assertEqual(sam.sam_delta, 0.3)
        self.assertEqual(sam.sam_lambda, 0.001)


if __name__ == "__main__":
    unittest.main()
