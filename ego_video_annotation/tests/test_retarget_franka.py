from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.pipeline import build_parser  # noqa: E402
from egoanno.retarget_franka import (  # noqa: E402
    RetargetConfig,
    build_short_lift_targets,
    combine_retarget_videos,
)


def sample(time_s: float, right_xy: tuple[float, float] | None) -> dict:
    hands = []
    if right_xy is not None:
        hands.append({
            "side": "right",
            "landmarks_2d_relative": [
                {"x": right_xy[0], "y": right_xy[1], "z": 0.0}
                for _ in range(21)
            ],
        })
    return {"time_s": time_s, "hands": hands}


class ShortLiftTargetTests(unittest.TestCase):
    def test_vertical_image_motion_maps_to_robot_height(self) -> None:
        samples = [
            sample(0.0, (0.50, 0.80)),
            sample(0.5, (0.48, 0.50)),
            sample(1.0, (0.50, 0.20)),
            sample(1.5, (0.52, 0.50)),
            sample(2.0, (0.50, 0.80)),
        ]
        config = RetargetConfig(control_fps=10.0, smoothing_s=0.0)
        result = build_short_lift_targets(samples, 2.0, config)

        target = result["target_xyz"]
        self.assertTrue(np.allclose(target[:, 0], config.robot_x_m))
        self.assertAlmostEqual(float(target[:, 2].min()), config.robot_z_low_m)
        self.assertAlmostEqual(float(target[:, 2].max()), config.robot_z_high_m)
        self.assertGreater(target[10, 2], target[0, 2])
        self.assertEqual(result["observed_sample_count"], 5)

    def test_long_tracking_gap_holds_last_pose_and_marks_it_unsupported(self) -> None:
        samples = [
            sample(0.0, (0.40, 0.80)),
            sample(2.0, (0.60, 0.20)),
            sample(2.5, (0.60, 0.20)),
        ]
        config = RetargetConfig(
            control_fps=10.0, gap_tolerance_s=0.5, smoothing_s=0.0,
        )
        result = build_short_lift_targets(samples, 2.5, config)

        one_second = 10
        self.assertFalse(bool(result["supported_mask"][one_second]))
        self.assertTrue(np.allclose(result["image_xy"][one_second], [0.40, 0.80]))

    def test_retarget_is_disabled_by_default_for_long_runs(self) -> None:
        args = build_parser().parse_args(["--video", "long1.mp4", "--output", "out"])
        self.assertFalse(args.retarget_franka)
        self.assertEqual(args.retarget_hand, "right")

    def test_bimanual_retarget_option_is_available_for_short2(self) -> None:
        args = build_parser().parse_args([
            "--video", "short2.mp4", "--output", "out",
            "--retarget-franka", "--retarget-hand", "both",
        ])
        self.assertTrue(args.retarget_franka)
        self.assertEqual(args.retarget_hand, "both")

    def test_two_single_arm_videos_can_be_composed_side_by_side(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.mp4"
            right_path = root / "right.mp4"
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
            cap = cv2.VideoCapture(str(output))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 120)
            cap.release()
            self.assertEqual(metadata["frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
