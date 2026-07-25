from __future__ import annotations

import json
import os
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egoanno.vlm_evaluation import (  # noqa: E402
    CLIP_SCHEMA,
    CLIP_SCORE_KEYS,
    JudgeConfig,
    ResponsesJudge,
    aggregate_metrics,
    config_from_args,
)


class MetricAggregationTests(unittest.TestCase):
    def test_scores_follow_documented_weights(self) -> None:
        clips = [
            {
                "duration_s": 1.0,
                "atomicity": 5, "completeness": 4,
                "hand_correctness": 5, "action_correctness": 4,
                "object_correctness": 3, "direction_correctness": 2,
            },
            {
                "duration_s": 3.0,
                "atomicity": 3, "completeness": 5,
                "hand_correctness": 4, "action_correctness": 5,
                "object_correctness": 4, "direction_correctness": 5,
            },
        ]
        boundaries = [{"boundary_validity": 4, "temporal_consistency": 3}]

        metrics = aggregate_metrics(clips, boundaries)

        self.assertEqual(metrics["segmentation_quality"], 81.67)
        self.assertEqual(metrics["caption_factuality"], 86.0)
        self.assertEqual(metrics["temporal_semantic_consistency"], 60.0)
        self.assertEqual(metrics["ego_seg_cap"], 79.07)
        self.assertEqual(metrics["components"]["atomicity"], 70.0)


class FakeResponse:
    status_code = 200
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        scores = {key: 4 for key in CLIP_SCORE_KEYS}
        scores["evidence"] = "画面证据支持该判断"
        return {
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(scores)}],
            }],
        }


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse()


class ResponsesJudgeTests(unittest.TestCase):
    def test_multimodal_responses_request_uses_independent_model_and_schema(self) -> None:
        session = FakeSession()
        config = JudgeConfig(
            api_base="https://example.test/api/v1",
            api_key="secret",
            model="gpt-5.4-mini",
        )
        judge = ResponsesJudge(config, session=session)
        frame = np.zeros((12, 16, 3), dtype=np.uint8)

        result = judge.score(
            "judge", [("[CLIP_FRAME] t=1.0s", frame)],
            "ego_clip_evaluation", CLIP_SCHEMA, CLIP_SCORE_KEYS,
        )

        self.assertEqual(result["action_correctness"], 4)
        self.assertEqual(len(session.calls), 1)
        call = session.calls[0]
        self.assertEqual(call["url"], "https://example.test/api/v1/responses")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        body = call["json"]
        self.assertEqual(body["model"], "gpt-5.4-mini")
        self.assertEqual(body["reasoning"]["effort"], "high")
        self.assertTrue(body["text"]["format"]["strict"])
        types = [item["type"] for item in body["input"][0]["content"]]
        self.assertIn("input_image", types)

    def test_judge_configuration_can_reuse_caption_api_key(self) -> None:
        args = Namespace(
            judge_api_base=None, judge_model=None,
            vlm_api_base="https://example.test/api/v1",
            fine_frame_count=16, vlm_context_s=0.75,
            vlm_image_max_side=768, judge_repeats=3,
            judge_reasoning_effort="high", judge_timeout_s=180,
            judge_boundary_frame_count=8, judge_max_pair_gap_s=0.25,
        )
        with patch.dict(
            os.environ,
            {"VLM_API_KEY": "shared-key", "JUDGE_MODEL": "gpt-5.4-mini"},
            clear=True,
        ):
            config = config_from_args(args)

        self.assertEqual(config.api_key, "shared-key")
        self.assertEqual(config.model, "gpt-5.4-mini")
        self.assertEqual(config.repeats, 3)


if __name__ == "__main__":
    unittest.main()
