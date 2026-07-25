from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.otas_segmentation import otas_boundary_selection  # noqa: E402
from egoanno.pipeline import build_parser  # noqa: E402


def piecewise_features(lengths: list[int]) -> np.ndarray:
    rows = []
    for index, length in enumerate(lengths):
        basis = np.zeros(len(lengths), dtype=np.float32)
        basis[index] = 1.0
        rows.append(np.tile(basis, (length, 1)))
    return np.vstack(rows)


class OTASBoundaryTests(unittest.TestCase):
    def test_three_stream_changes_create_contiguous_segments(self) -> None:
        features = piecewise_features([20, 20, 20])
        times = np.arange(len(features), dtype=np.float64) / 8.0
        result = otas_boundary_selection(
            {
                "global": features,
                "interaction": features,
                "relation": features,
            },
            times,
            peak_window_s=0.5,
            neighbor_s=0.5,
            min_gap_s=0.5,
            candidate_threshold=1.0,
            strong_local_threshold=2.5,
            smoothing_s=0.0,
        )

        self.assertEqual(
            [item["index"] for item in result["boundaries"]], [20, 40],
        )
        self.assertEqual(result["runs"], [
            {"start": 0, "end": 20},
            {"start": 20, "end": 40},
            {"start": 40, "end": 60},
        ])

    def test_unconfirmed_weak_global_peak_is_rejected(self) -> None:
        global_features = piecewise_features([20, 20])
        static = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (40, 1))
        times = np.arange(40, dtype=np.float64) / 8.0
        result = otas_boundary_selection(
            {
                "global": global_features,
                "interaction": static,
                "relation": static,
            },
            times,
            peak_window_s=0.5,
            neighbor_s=0.5,
            min_gap_s=0.5,
            candidate_threshold=1.0,
            strong_local_threshold=2.5,
            smoothing_s=0.0,
        )
        self.assertEqual(result["boundaries"], [])
        self.assertEqual(result["runs"], [{"start": 0, "end": 40}])

    def test_parser_exposes_otas_without_changing_default(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["--video", "x.mp4", "--output", "out"])
        otas = parser.parse_args([
            "--video", "x.mp4", "--output", "out",
            "--segmentation-method", "otas",
        ])
        self.assertEqual(default.segmentation_method, "ours")
        self.assertEqual(otas.segmentation_method, "otas")
        self.assertEqual(otas.otas_global_weight, 1.0)
        self.assertEqual(otas.otas_interaction_weight, 1.0)
        self.assertEqual(otas.otas_relation_weight, 1.0)
        self.assertEqual(otas.otas_peak_window_s, 0.75)
        self.assertEqual(otas.otas_neighbor_s, 0.75)
        self.assertEqual(otas.otas_min_gap_s, 0.75)


if __name__ == "__main__":
    unittest.main()
