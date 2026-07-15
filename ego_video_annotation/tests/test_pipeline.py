from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.pipeline import (  # noqa: E402
    _long_no_hand_boundaries,
    HandIdentityTracker,
    compatible_fine_annotations,
    consolidate_boundaries,
    find_velocity_candidates,
    segment_record,
    spans_from_boundaries,
)


def detected_hand(side: str, x: float, score: float = 0.9) -> dict:
    return {
        "raw_side": side,
        "score": score,
        "landmarks_2d_relative": [{"x": x, "y": 0.5, "z": 0.0} for _ in range(21)],
        "landmarks_3d_relative": None,
    }


class HandIdentityTests(unittest.TestCase):
    def test_non_finite_detection_is_dropped_instead_of_crashing(self) -> None:
        tracker = HandIdentityTracker()
        assigned = tracker.assign([detected_hand("left", float("nan"))], 1.0)
        self.assertEqual(assigned, [])

    def test_unexpected_extra_detection_is_capped_at_two(self) -> None:
        tracker = HandIdentityTracker()
        hands = [
            detected_hand("left", 0.2, 0.9),
            detected_hand("right", 0.8, 0.8),
            detected_hand("unknown", 0.5, 0.1),
        ]
        assigned = tracker.assign(hands, 1.0)
        self.assertEqual(len(assigned), 2)
        self.assertEqual({hand["side"] for hand in assigned}, {"left", "right"})


def args() -> Namespace:
    return Namespace(
        sample_fps=8.0,
        velocity_center_s=0.20,
        velocity_context_s=0.75,
        velocity_drop_ratio=0.40,
        velocity_prominence=0.20,
        velocity_min_gap_s=0.80,
        hand_boundary_fusion_s=0.40,
    )


def motion_samples(left: list[float], right: list[float] | None = None) -> list[dict]:
    right = right or [float("nan")] * len(left)
    result = []
    for index, (left_speed, right_speed) in enumerate(zip(left, right)):
        def item(value: float) -> dict:
            return {
                "smoothed_speed": None if not np.isfinite(value) else value,
                "raw_speed": None if not np.isfinite(value) else value,
                "valid": bool(np.isfinite(value)),
            }

        result.append({
            "time_s": index / 8.0,
            "camera_motion_quality": 0.9,
            "hand_motion": {"left": item(left_speed), "right": item(right_speed)},
        })
    return result


class VelocityBoundaryTests(unittest.TestCase):
    def test_significant_valley_is_detected_and_two_hands_are_fused(self) -> None:
        speed = [0.10] * 13 + [0.08, 0.04, 0.01, 0.04, 0.08] + [0.10] * 14
        candidates = find_velocity_candidates(motion_samples(speed, speed), args())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["hands"], ["left", "right"])
        self.assertAlmostEqual(candidates[0]["time_s"], 15 / 8.0, places=3)

    def test_constant_low_speed_does_not_create_boundaries(self) -> None:
        candidates = find_velocity_candidates(motion_samples([0.005] * 40), args())
        self.assertEqual(candidates, [])

    def test_half_open_spans_cover_every_sample(self) -> None:
        boundaries = [{"index": 3}, {"index": 7}]
        spans = spans_from_boundaries(boundaries, 10)
        self.assertEqual(spans, [(0, 3), (3, 7), (7, 10)])
        flattened = [index for start, end in spans for index in range(start, end)]
        self.assertEqual(flattened, list(range(10)))

    def test_long_no_hand_interval_is_isolated_and_not_exportable(self) -> None:
        samples = []
        for index in range(24):
            present = index < 6 or index >= 16
            samples.append({
                "time_s": index / 8.0,
                "hand_present_raw": present,
                "hand_present_smoothed": present,
                "hand_gap_bridged": False,
                "sharpness": 100.0,
                "hand_validity": {
                    side: {"observed": present, "smoothed_presence": present}
                    for side in ("left", "right")
                },
            })
        boundaries = consolidate_boundaries(_long_no_hand_boundaries(samples, 1.0), 0.5)
        self.assertEqual([item["index"] for item in boundaries], [6, 16])
        records = [
            segment_record(span, samples, index + 1, "fine", 3.0, Namespace(
                min_hand_coverage=0.30,
                max_no_hand_gap_s=1.0,
                min_export_duration_s=0.5,
            ))
            for index, span in enumerate(spans_from_boundaries(boundaries, len(samples)))
        ]
        self.assertEqual([item["valid_operation"] for item in records], [True, False, True])


class SemanticMergeTests(unittest.TestCase):
    @staticmethod
    def annotation(key: str, verb: str, obj: str) -> dict:
        return {
            "annotation_source": "vlm",
            "meaningful_action": True,
            "semantic_key": key,
            "action": {"verb": verb},
            "objects": [obj],
            "subtask": f"连续{verb}{obj}",
        }

    def test_repeated_cutting_is_compatible(self) -> None:
        left = self.annotation("右手|切割|蒜", "切割", "蒜")
        right = self.annotation("右手|切割|蒜", "切", "蒜")
        self.assertTrue(compatible_fine_annotations(left, right))

    def test_pick_and_place_are_not_compatible(self) -> None:
        left = self.annotation("右手|拿起|杯子", "拿起", "杯子")
        right = self.annotation("右手|放下|杯子", "放下", "杯子")
        self.assertFalse(compatible_fine_annotations(left, right))


if __name__ == "__main__":
    unittest.main()
