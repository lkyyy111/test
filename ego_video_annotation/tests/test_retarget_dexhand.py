from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.pipeline import build_parser  # noqa: E402
from egoanno.retarget_dexhand import (  # noqa: E402
    DexHandConfig,
    JOINT_NAMES,
    build_dex_hand_mjcf,
    build_dex_hand_targets,
    combine_retarget_videos,
    hand_landmarks_to_joint_targets,
)


def open_hand_points(wrist_x: float = 0.5, wrist_y: float = 0.7, scale: float = 1.0) -> list[dict]:
    # A palm-local open hand whose fingers point upward in image coordinates.
    xy = [
        (0.00, 0.00),
        (0.04, -0.03), (0.08, -0.07), (0.11, -0.10), (0.14, -0.13),
        (0.05, -0.10), (0.05, -0.18), (0.05, -0.25), (0.05, -0.31),
        (0.01, -0.11), (0.01, -0.20), (0.01, -0.28), (0.01, -0.35),
        (-0.03, -0.10), (-0.03, -0.18), (-0.03, -0.25), (-0.03, -0.31),
        (-0.07, -0.08), (-0.08, -0.15), (-0.09, -0.21), (-0.10, -0.26),
    ]
    return [
        {"x": wrist_x + scale * x, "y": wrist_y + scale * y, "z": 0.0}
        for x, y in xy
    ]


def sample(time_s: float, scale: float, side: str = "right") -> dict:
    points = open_hand_points(scale=scale)
    return {
        "time_s": time_s,
        "hands": [{
            "side": side,
            "landmarks_2d_relative": points,
            "landmarks_3d_relative": points,
        }],
    }


class DexHandPoseTests(unittest.TestCase):
    def test_open_hand_produces_twenty_finite_joint_targets(self) -> None:
        hand = sample(0.0, 1.0)["hands"][0]
        targets = hand_landmarks_to_joint_targets(hand, "right")
        self.assertEqual(targets.shape, (20,))
        self.assertTrue(np.isfinite(targets).all())
        self.assertEqual(len(JOINT_NAMES), 20)

    def test_bending_index_increases_pip_and_dip_flexion(self) -> None:
        straight = sample(0.0, 1.0)["hands"][0]
        bent = sample(0.0, 1.0)["hands"][0]
        bent_points = bent["landmarks_3d_relative"]
        bent_points[7] = {"x": 0.11, "y": 0.52, "z": 0.0}
        bent_points[8] = {"x": 0.16, "y": 0.55, "z": 0.0}
        straight_q = hand_landmarks_to_joint_targets(straight, "right")
        bent_q = hand_landmarks_to_joint_targets(bent, "right")
        # index occupies entries 4..7: abd, mcp, pip, dip.
        self.assertGreater(bent_q[6] + bent_q[7], straight_q[6] + straight_q[7])

    def test_smaller_apparent_hand_moves_forward_in_first_person_scene(self) -> None:
        samples = [sample(0.0, 1.0), sample(0.5, 0.75), sample(1.0, 0.50)]
        result = build_dex_hand_targets(
            samples, 1.0, DexHandConfig(control_fps=10.0, smoothing_s=0.0),
        )
        self.assertGreater(result["root_xyz"][-1, 0], result["root_xyz"][0, 0])
        self.assertEqual(result["joint_targets"].shape, (11, 20))

    def test_long_tracking_gap_holds_pose_and_marks_it_unsupported(self) -> None:
        samples = [sample(0.0, 1.0), sample(2.0, 0.5), sample(2.5, 0.5)]
        result = build_dex_hand_targets(
            samples, 2.5,
            DexHandConfig(control_fps=10.0, gap_tolerance_s=0.5, smoothing_s=0.0),
        )
        self.assertFalse(bool(result["supported_mask"][10]))
        self.assertTrue(np.allclose(result["joint_targets"][10], result["joint_targets"][0]))

    def test_generated_model_is_self_contained_twenty_dof_hand(self) -> None:
        root = ET.fromstring(build_dex_hand_mjcf("right"))
        joints = root.findall(".//joint")
        free_joints = root.findall(".//freejoint")
        self.assertEqual(len(joints), 20)
        self.assertEqual(len(free_joints), 1)
        self.assertEqual({joint.attrib["name"] for joint in joints}, set(JOINT_NAMES))


class DexHandCliTests(unittest.TestCase):
    def test_retarget_is_disabled_by_default_for_long_runs(self) -> None:
        args = build_parser().parse_args(["--video", "long1.mp4", "--output", "out"])
        self.assertFalse(args.retarget_dex_hand)
        self.assertEqual(args.retarget_hand, "right")

    def test_bimanual_option_is_available_for_short2(self) -> None:
        args = build_parser().parse_args([
            "--video", "short2.mp4", "--output", "out",
            "--retarget-dex-hand", "--retarget-hand", "both",
        ])
        self.assertTrue(args.retarget_dex_hand)
        self.assertEqual(args.retarget_hand, "both")

    def test_old_franka_switch_is_a_compatibility_alias(self) -> None:
        args = build_parser().parse_args([
            "--video", "short1.mp4", "--output", "out", "--retarget-franka",
        ])
        self.assertTrue(args.retarget_dex_hand)

    def test_two_hand_videos_can_be_composed_side_by_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path, right_path = root / "left.mp4", root / "right.mp4"
            for path, color in ((left_path, (255, 0, 0)), (right_path, (0, 255, 0))):
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120),
                )
                self.assertTrue(writer.isOpened())
                for _ in range(3):
                    writer.write(np.full((120, 160, 3), color, dtype=np.uint8))
                writer.release()
            output = root / "both.mp4"
            metadata = combine_retarget_videos(left_path, right_path, output)
            self.assertEqual(metadata["frame_count"], 3)
            self.assertEqual(metadata["width"], 320)


if __name__ == "__main__":
    unittest.main()
