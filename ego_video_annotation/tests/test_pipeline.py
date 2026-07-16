from __future__ import annotations

import sys
import unittest
import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.pipeline import (  # noqa: E402
    _long_no_hand_boundaries,
    HandIdentityTracker,
    annotate_hand_validity,
    attach_hand_motion,
    build_clean_annotations,
    compatible_fine_annotations,
    consolidate_boundaries,
    correct_mediapipe_handedness,
    find_velocity_candidates,
    _refinement_indices,
    normalize_annotation,
    segment_record,
    spans_from_boundaries,
    recaption_merged_fine_segments,
    VideoInfo,
)


def detected_hand(side: str, x: float, score: float = 0.9) -> dict:
    return {
        "raw_side": side,
        "side": side,
        "score": score,
        "landmarks_2d_relative": [{"x": x, "y": 0.5, "z": 0.0} for _ in range(21)],
        "landmarks_3d_relative": None,
    }


class HandIdentityTests(unittest.TestCase):
    def test_non_mirrored_egocentric_handedness_is_swapped(self) -> None:
        self.assertEqual(correct_mediapipe_handedness("Left"), "right")
        self.assertEqual(correct_mediapipe_handedness("Right"), "left")
        self.assertEqual(correct_mediapipe_handedness("unknown"), "unknown")

    def test_identity_tracker_prefers_corrected_detector_side(self) -> None:
        tracker = HandIdentityTracker()
        hand = detected_hand("left", 0.7)
        hand["mediapipe_side"] = "left"
        hand["detector_side"] = "right"
        assigned = tracker.assign([hand], 0.0)
        self.assertEqual(assigned[0]["side"], "right")

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
        velocity_min_window_weight=0.60,
        motion_interpolation_gap_s=0.50,
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

    def test_short_gap_is_interpolated_only_for_motion(self) -> None:
        samples = []
        for index in range(8):
            hands = [] if index in (2, 3) else [detected_hand("left", 0.2 + 0.01 * index)]
            samples.append({
                "time_s": index / 8.0,
                "hands": hands,
                "camera_motion_quality": 1.0,
                "scene_change_diagnostic": 0.0,
                "camera_motion_from_previous": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            })
        annotate_hand_validity(samples, 0.5)
        attach_hand_motion(samples, 0.15, 0.25, 0.5, 0.25)
        missing_track = samples[2]["hand_tracks"]["left"]
        self.assertFalse(missing_track["valid_mask"])
        self.assertIsNone(missing_track["landmarks_2d_relative"])
        self.assertTrue(missing_track["interpolated_for_motion"])
        self.assertIsNotNone(missing_track["boundary_palm_center"])
        self.assertEqual(samples[3]["hand_motion"]["left"]["source"], "interpolated_for_motion")

    def test_vlm_suggested_frame_aligns_to_nearby_weak_valley(self) -> None:
        speed = [0.10] * 6 + [0.08, 0.04, 0.01, 0.04, 0.08] + [0.10] * 7
        samples = motion_samples(speed)
        segment = {
            "sample_start": 0,
            "sample_end": len(samples) - 1,
            "start_s": 0.0,
            "end_s": len(samples) / 8.0,
            "vlm_frame_times_s": [0.0, 0.5, 1.0, 1.5],
            "semantic_annotation": {"suggested_boundary_frames": [3]},
        }
        config = args()
        config.minimum_provisional_duration_s = 0.5
        config.max_vlm_refinement_splits = 2
        config.vlm_refinement_search_s = 1.0
        selected = _refinement_indices(samples, segment, config)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0]["time_s"], 1.0, places=3)
        self.assertTrue(selected[0]["aligned_to_weak_velocity_minimum"])


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


class FineAnnotationSchemaTests(unittest.TestCase):
    def test_coarse_stage_from_vlm_is_not_emitted(self) -> None:
        annotation = normalize_annotation(
            {"coarse_stage": "切配", "subtask": "切蒜", "confidence": 0.9},
            {"hand_coverage": 1.0},
        )
        self.assertNotIn("coarse_stage", annotation)

    @staticmethod
    def segment(segment_id: str, **overrides: object) -> dict:
        result = {
            "id": segment_id,
            "start_s": 1.25,
            "end_s": 3.75,
            "duration_s": 2.5,
            "sample_start": 10,
            "sample_end": 29,
            "hand_coverage": 0.9,
            "valid_operation": True,
            "needs_review": False,
            "caption_zh": "拿起杯子：右手拿起杯子",
            "semantic_annotation": {
                "annotation_source": "vlm",
                "meaningful_action": True,
                "subtask": "拿起杯子",
                "action": {"verb": "拿起", "description": "右手拿起杯子"},
                "objects": ["杯子"],
                "left_hand": {"visible": False, "action": "无", "object": "无"},
                "right_hand": {"visible": True, "action": "拿起", "object": "杯子"},
                "confidence": 0.9,
            },
        }
        result.update(overrides)
        return result

    def test_clean_annotations_keep_only_reviewed_vlm_actions(self) -> None:
        accepted = self.segment("fine_001")
        rejected_review = self.segment("fine_002", needs_review=True)
        rejected_invalid = self.segment("fine_003", valid_operation=False)
        fallback = self.segment("fine_004")
        fallback["semantic_annotation"] = {
            **fallback["semantic_annotation"], "annotation_source": "fallback",
        }
        info = VideoInfo("/dataset/long1.mp4", 50.0, 15000, 1920, 1080, 300.0)

        payload = build_clean_annotations(
            info, [accepted, rejected_review, rejected_invalid, fallback], clips_exported=True,
        )

        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["video"]["file"], "long1.mp4")
        self.assertEqual(payload["hand_data_file"], "hand_landmarks.json")
        self.assertEqual(len(payload["clips"]), 1)
        clip = payload["clips"][0]
        self.assertEqual(clip["id"], "fine_001")
        self.assertEqual(clip["clip_path"], "valid_segments/fine_001_1.2-3.8s.mp4")
        self.assertEqual(clip["sample_range"], {"start": 10, "end": 29})
        self.assertEqual(set(clip["hands"]), {"left", "right"})
        self.assertNotIn("semantic_key", clip)

    def test_smoke_output_uses_null_clip_path(self) -> None:
        info = VideoInfo("short1.mp4", 30.0, 300, 640, 480, 10.0)
        payload = build_clean_annotations(info, [self.segment("fine_001")], clips_exported=False)
        self.assertIsNone(payload["clips"][0]["clip_path"])


class MergedRecaptionTests(unittest.TestCase):
    def test_successful_full_clip_recaption_replaces_old_annotation(self) -> None:
        old = SemanticMergeTests.annotation("右手|切割|蒜", "切割", "蒜")
        old.update({
            "scene": "厨房", "left_hand": {}, "right_hand": {},
            "temporal_evidence": "局部", "contains_multiple_actions": False, "confidence": 0.9,
        })
        segment = {
            "id": "fine_001", "valid_operation": True, "merged_from": ["a", "b"],
            "duration_s": 12.0, "start_s": 0.0, "end_s": 12.0, "hand_coverage": 1.0,
            "semantic_annotation": old, "caption_zh": "旧描述",
        }
        new = {**old, "subtask": "连续切蒜", "action": {"verb": "切割", "description": "完整切蒜过程"}}
        info = VideoInfo("dummy.mp4", 30.0, 360, 640, 480, 12.0)
        config = Namespace(
            merged_recaption_min_duration_s=8.0, fine_frame_count=16,
            vlm_api_base="http://example", vlm_model="model", vlm_image_max_side=768,
        )
        with patch.dict(os.environ, {"VLM_API_KEY": "test-key"}), patch(
            "egoanno.pipeline.sample_segment_frames",
            return_value=([np.zeros((2, 2, 3), np.uint8)], [6.0]),
        ), patch("egoanno.pipeline.vlm_annotation", return_value=new):
            attempted, succeeded = recaption_merged_fine_segments([segment], info, config)
        self.assertEqual((attempted, succeeded), (1, 1))
        self.assertEqual(segment["semantic_annotation"]["subtask"], "连续切蒜")
        self.assertEqual(segment["pre_recaption_annotation"]["subtask"], old["subtask"])


if __name__ == "__main__":
    unittest.main()
