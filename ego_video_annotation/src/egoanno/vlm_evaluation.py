from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable

import cv2
import numpy as np
import requests
from tqdm import tqdm


CLIP_SCORE_KEYS = (
    "atomicity",
    "completeness",
    "hand_correctness",
    "action_correctness",
    "object_correctness",
    "direction_correctness",
)
BOUNDARY_SCORE_KEYS = ("boundary_validity", "temporal_consistency")


CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            key: {"type": "integer", "minimum": 0, "maximum": 5}
            for key in CLIP_SCORE_KEYS
        },
        "evidence": {"type": "string"},
    },
    "required": [*CLIP_SCORE_KEYS, "evidence"],
    "additionalProperties": False,
}


BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            key: {"type": "integer", "minimum": 0, "maximum": 5}
            for key in BOUNDARY_SCORE_KEYS
        },
        "evidence": {"type": "string"},
    },
    "required": [*BOUNDARY_SCORE_KEYS, "evidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeConfig:
    api_base: str
    api_key: str
    model: str
    reasoning_effort: str = "high"
    repeats: int = 1
    image_max_side: int = 768
    timeout_s: float = 180.0
    clip_frame_count: int = 16
    context_s: float = 0.75
    boundary_frame_count: int = 8
    max_pair_gap_s: float = 0.25


class JudgeError(RuntimeError):
    pass


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise JudgeError("裁判响应不包含JSON对象")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise JudgeError(f"裁判JSON解析失败：{error}") from error
    if not isinstance(value, dict):
        raise JudgeError("裁判响应不是JSON对象")
    return value


def _response_output_text(payload: dict[str, Any]) -> str:
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                return str(content.get("text") or "")
    raise JudgeError("Responses响应缺少output_text")


def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame, (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _input_image(frame: np.ndarray, max_side: int) -> dict[str, str]:
    resized = _resize(frame, max_side)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise JudgeError("评测帧JPEG编码失败")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + data}


def _normalized_score(value: Any, key: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise JudgeError(f"裁判字段{key}不是整数") from error
    if not 0 <= numeric <= 5:
        raise JudgeError(f"裁判字段{key}超出0到5范围")
    return numeric


class ResponsesJudge:
    def __init__(self, config: JudgeConfig, session: Any = requests) -> None:
        self.config = config
        self.session = session

    def score(
        self,
        prompt: str,
        labeled_frames: list[tuple[str, np.ndarray]],
        schema_name: str,
        schema: dict[str, Any],
        score_keys: tuple[str, ...],
    ) -> dict[str, Any]:
        if not labeled_frames:
            raise JudgeError("没有成功读取任何评测帧")
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for label, frame in labeled_frames:
            content.append({"type": "input_text", "text": label})
            content.append(_input_image(frame, self.config.image_max_side))
        body = {
            "model": self.config.model,
            "reasoning": {"effort": self.config.reasoning_effort},
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 768,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        url = self.config.api_base.rstrip("/") + "/responses"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            response = self.session.post(
                url, headers=headers, json=body, timeout=self.config.timeout_s,
            )
        except requests.RequestException as error:
            raise JudgeError(f"裁判API连接失败：{error}") from error

        # Some OpenAI-compatible gateways implement Responses multimodal input
        # before implementing strict structured output.  Prefer json_schema,
        # but fall back to prompt-constrained JSON for those gateways.
        if getattr(response, "status_code", None) in {400, 404, 422}:
            fallback_body = dict(body)
            fallback_body.pop("text", None)
            fallback_body["input"] = [{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "严格只输出一个JSON对象，不要Markdown代码块。",
                    },
                    *content,
                ],
            }]
            try:
                response = self.session.post(
                    url, headers=headers, json=fallback_body,
                    timeout=self.config.timeout_s,
                )
            except requests.RequestException as error:
                raise JudgeError(f"裁判API兼容模式连接失败：{error}") from error
        try:
            response.raise_for_status()
        except requests.RequestException as error:
            detail = getattr(response, "text", "")[:1000]
            raise JudgeError(f"裁判API请求失败：{error}；{detail}") from error
        try:
            payload = response.json()
        except (ValueError, TypeError) as error:
            raise JudgeError("裁判API返回的不是JSON") from error
        if payload.get("status") not in {None, "completed"} or payload.get("error"):
            raise JudgeError(f"裁判API状态异常：{payload.get('status')} / {payload.get('error')}")
        result = _parse_json_text(_response_output_text(payload))
        normalized = {key: _normalized_score(result.get(key), key) for key in score_keys}
        normalized["evidence"] = str(result.get("evidence") or "").strip()
        return normalized


def _clip_prompt(clip: dict[str, Any]) -> str:
    return (
        "你是独立、严格且不知道方法来源的第一人称操作视频评测员。"
        "只根据按时间顺序提供的图像判断，不偏好待评caption的措辞。"
        "BEFORE/AFTER帧只用于理解边界状态；CLIP帧才属于当前片段。\n"
        f"当前片段时间：{float(clip['start_s']):.3f}s–{float(clip['end_s']):.3f}s。\n"
        f"待评caption：{clip.get('caption', '')}\n"
        f"结构化动作：{json.dumps(clip.get('action', {}), ensure_ascii=False)}\n"
        f"物体：{json.dumps(clip.get('objects', []), ensure_ascii=False)}\n"
        f"左右手：{json.dumps(clip.get('hands', {}), ensure_ascii=False)}\n"
        "请对以下项目各打0到5的整数分：\n"
        "atomicity：5=CLIP帧只有一个连续原子动作；0=含多个明显不同动作。"
        "连续切割、擦拭、清洗、搅拌的往返过程仍算一个动作。\n"
        "completeness：5=动作起止完整且边界自然；0=明显从动作中间开始或结束。\n"
        "hand_correctness：操作手描述是否正确。\n"
        "action_correctness：动作动词和行为是否正确。\n"
        "object_correctness：交互物体是否正确且无幻觉。\n"
        "direction_correctness：拿起/放下、取出/放回、打开/关闭等方向和状态变化是否正确。\n"
        "evidence用一句中文说明最关键的视频证据。严格按JSON Schema输出。"
    )


def _boundary_prompt(previous: dict[str, Any], current: dict[str, Any]) -> str:
    return (
        "你是独立、严格且不知道方法来源的第一人称操作视频边界评测员。"
        "以下图像按时间排序，PREVIOUS来自边界前，CURRENT来自边界后。\n"
        f"边界时间约为{float(current['start_s']):.3f}s。\n"
        f"前片段caption：{previous.get('caption', '')}\n"
        f"后片段caption：{current.get('caption', '')}\n"
        "请各打0到5的整数分：\n"
        "boundary_validity：5=这里确实是两个原子动作/明确状态的自然切换；"
        "0=前后仍是同一个连续动作，属于明显过切。\n"
        "temporal_consistency：5=两个caption与画面中的物体状态、动作方向和先后顺序完全连贯；"
        "0=明显矛盾、方向颠倒或重复描述不成立。\n"
        "连续切割、擦拭、清洗、搅拌的往返不应仅因一次速度停顿而视作新动作。"
        "evidence用一句中文说明判断依据。严格按JSON Schema输出。"
    )


def _median_result(results: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        **{key: float(median([float(result[key]) for result in results])) for key in keys},
        "evidence": [result.get("evidence", "") for result in results],
        "successful_repeats": len(results),
    }


def _weighted_mean(records: list[dict[str, Any]], key: str, weight_key: str) -> float | None:
    valid = [record for record in records if key in record and float(record.get(weight_key, 0)) > 0]
    if not valid:
        return None
    weights = np.asarray([float(record[weight_key]) for record in valid], dtype=float)
    values = np.asarray([float(record[key]) for record in valid], dtype=float)
    return float(np.average(values, weights=weights))


def aggregate_metrics(
    clip_evaluations: list[dict[str, Any]],
    boundary_evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    atomicity = _weighted_mean(clip_evaluations, "atomicity", "duration_s")
    completeness = _weighted_mean(clip_evaluations, "completeness", "duration_s")
    boundary = (
        float(np.mean([item["boundary_validity"] for item in boundary_evaluations]))
        if boundary_evaluations else None
    )
    sq_components = [value for value in (boundary, atomicity, completeness) if value is not None]
    sq = 20.0 * float(np.mean(sq_components)) if sq_components else None

    hand = _weighted_mean(clip_evaluations, "hand_correctness", "duration_s")
    action = _weighted_mean(clip_evaluations, "action_correctness", "duration_s")
    objects = _weighted_mean(clip_evaluations, "object_correctness", "duration_s")
    direction = _weighted_mean(clip_evaluations, "direction_correctness", "duration_s")
    cf = None
    if all(value is not None for value in (hand, action, objects, direction)):
        cf = 20.0 * (
            0.15 * float(hand) + 0.35 * float(action)
            + 0.25 * float(objects) + 0.25 * float(direction)
        )
    tsc = (
        20.0 * float(np.mean([item["temporal_consistency"] for item in boundary_evaluations]))
        if boundary_evaluations else None
    )
    overall = None
    if sq is not None and cf is not None and tsc is not None:
        overall = 0.4 * sq + 0.4 * cf + 0.2 * tsc
    return {
        "segmentation_quality": None if sq is None else round(sq, 2),
        "caption_factuality": None if cf is None else round(cf, 2),
        "temporal_semantic_consistency": None if tsc is None else round(tsc, 2),
        "ego_seg_cap": None if overall is None else round(overall, 2),
        "components": {
            "boundary_validity": None if boundary is None else round(20.0 * boundary, 2),
            "atomicity": None if atomicity is None else round(20.0 * atomicity, 2),
            "completeness": None if completeness is None else round(20.0 * completeness, 2),
            "hand_correctness": None if hand is None else round(20.0 * hand, 2),
            "action_correctness": None if action is None else round(20.0 * action, 2),
            "object_correctness": None if objects is None else round(20.0 * objects, 2),
            "direction_correctness": None if direction is None else round(20.0 * direction, 2),
        },
    }


def _boundary_frames(
    info: Any, previous: dict[str, Any], current: dict[str, Any], frame_count: int,
) -> list[tuple[str, np.ndarray]]:
    frame_count = max(2, int(frame_count))
    before_count = frame_count // 2
    after_count = frame_count - before_count
    boundary_s = 0.5 * (float(previous["end_s"]) + float(current["start_s"]))
    before_start = max(float(previous["start_s"]), boundary_s - 0.75)
    before_end = min(float(previous["end_s"]), boundary_s - 1.0 / float(info.fps))
    after_start = max(float(current["start_s"]), boundary_s)
    after_end = min(float(current["end_s"]), boundary_s + 0.75)
    before_times = np.linspace(before_start, max(before_start, before_end), before_count)
    after_times = np.linspace(after_start, max(after_start, after_end), after_count)
    result: list[tuple[str, np.ndarray]] = []
    cap = cv2.VideoCapture(str(info.path))
    try:
        for role, times in (("PREVIOUS", before_times), ("CURRENT", after_times)):
            for time_s in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_s)) * 1000)
                ok, frame = cap.read()
                if ok:
                    result.append((f"[{role}] t={float(time_s):.3f}s", frame))
    finally:
        cap.release()
    return result


def run_vlm_evaluation(
    info: Any,
    annotations: dict[str, Any],
    config: JudgeConfig,
    sample_clip_frames: Callable[[Any, dict[str, Any], int, float], tuple[list[np.ndarray], list[float]]],
) -> dict[str, Any]:
    clips = list(annotations.get("clips", []))
    if not clips:
        raise RuntimeError("annotations.json没有可供VLM评价的最终片段")
    judge = ResponsesJudge(config)
    clip_evaluations: list[dict[str, Any]] = []
    boundary_evaluations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with tqdm(total=len(clips) * config.repeats, desc="GPT裁判-片段", unit="call") as progress:
        for clip in clips:
            frames, times = sample_clip_frames(
                info, clip, config.clip_frame_count, config.context_s,
            )
            labeled = []
            for frame, time_s in zip(frames, times):
                role = (
                    "CONTEXT_BEFORE" if time_s < float(clip["start_s"]) - 0.02
                    else "CONTEXT_AFTER" if time_s > float(clip["end_s"]) + 0.02
                    else "CLIP_FRAME"
                )
                labeled.append((f"[{role}] t={time_s:.3f}s", frame))
            repeat_results = []
            for repeat_index in range(config.repeats):
                try:
                    repeat_results.append(judge.score(
                        _clip_prompt(clip), labeled, "ego_clip_evaluation",
                        CLIP_SCHEMA, CLIP_SCORE_KEYS,
                    ))
                except JudgeError as error:
                    failures.append({
                        "level": "clip", "clip_id": clip.get("id"),
                        "repeat": repeat_index + 1, "error": str(error),
                    })
                progress.update(1)
            if repeat_results:
                clip_evaluations.append({
                    "clip_id": clip.get("id"),
                    "start_s": clip.get("start_s"), "end_s": clip.get("end_s"),
                    "duration_s": max(float(clip.get("duration_s", 0.0)), 1e-6),
                    **_median_result(repeat_results, CLIP_SCORE_KEYS),
                })

    contiguous_pairs = [
        (previous, current) for previous, current in zip(clips, clips[1:])
        if abs(float(current["start_s"]) - float(previous["end_s"])) <= config.max_pair_gap_s
    ]
    with tqdm(
        total=len(contiguous_pairs) * config.repeats,
        desc="GPT裁判-边界", unit="call",
    ) as progress:
        for previous, current in contiguous_pairs:
            labeled = _boundary_frames(
                info, previous, current, config.boundary_frame_count,
            )
            repeat_results = []
            for repeat_index in range(config.repeats):
                try:
                    repeat_results.append(judge.score(
                        _boundary_prompt(previous, current), labeled,
                        "ego_boundary_evaluation", BOUNDARY_SCHEMA,
                        BOUNDARY_SCORE_KEYS,
                    ))
                except JudgeError as error:
                    failures.append({
                        "level": "boundary",
                        "previous_clip_id": previous.get("id"),
                        "current_clip_id": current.get("id"),
                        "repeat": repeat_index + 1, "error": str(error),
                    })
                progress.update(1)
            if repeat_results:
                boundary_evaluations.append({
                    "previous_clip_id": previous.get("id"),
                    "current_clip_id": current.get("id"),
                    "boundary_s": current.get("start_s"),
                    **_median_result(repeat_results, BOUNDARY_SCORE_KEYS),
                })

    if not clip_evaluations:
        first_error = failures[0]["error"] if failures else "未知错误"
        raise RuntimeError(f"所有片段的VLM评价均失败：{first_error}")
    metrics = aggregate_metrics(clip_evaluations, boundary_evaluations)
    return {
        "schema_version": "1.0",
        "status": "complete" if not failures else "partial",
        "judge": {
            "provider": "OpenAI-compatible Responses API",
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "repeats": config.repeats,
            "clip_frame_protocol": "2 context before + 12 clip + 2 context after when budget=16",
            "boundary_frame_count": config.boundary_frame_count,
            "blind_method_identity": True,
        },
        "metric_definition": {
            "segmentation_quality": "mean of boundary validity, duration-weighted atomicity and completeness",
            "caption_factuality": "0.15 hand + 0.35 action + 0.25 object + 0.25 direction",
            "temporal_semantic_consistency": "mean adjacent-boundary temporal consistency",
            "ego_seg_cap": "0.4 SQ + 0.4 CF + 0.2 TSC",
        },
        "counts": {
            "input_clips": len(clips),
            "evaluated_clips": len(clip_evaluations),
            "contiguous_boundaries": len(contiguous_pairs),
            "evaluated_boundaries": len(boundary_evaluations),
            "failed_calls": len(failures),
        },
        "metrics": metrics,
        "clip_evaluations": clip_evaluations,
        "boundary_evaluations": boundary_evaluations,
        "failures": failures,
    }


def config_from_args(args: Any) -> JudgeConfig:
    api_base = (
        getattr(args, "judge_api_base", None)
        or os.getenv("JUDGE_API_BASE")
        or getattr(args, "vlm_api_base", None)
        or os.getenv("VLM_API_BASE")
    )
    model = getattr(args, "judge_model", None) or os.getenv("JUDGE_MODEL")
    key = os.getenv("JUDGE_API_KEY") or os.getenv("VLM_API_KEY")
    missing = [
        name for name, value in (
            ("judge API base", api_base), ("judge model", model), ("JUDGE_API_KEY/VLM_API_KEY", key),
        ) if not value
    ]
    if missing:
        raise RuntimeError("缺少VLM评测配置：" + "、".join(missing))
    repeats = int(getattr(args, "judge_repeats", 1))
    if repeats <= 0:
        raise ValueError("--judge-repeats必须大于0")
    return JudgeConfig(
        api_base=str(api_base), api_key=str(key), model=str(model),
        reasoning_effort=str(getattr(args, "judge_reasoning_effort", "high")),
        repeats=repeats,
        image_max_side=int(getattr(args, "vlm_image_max_side", 768)),
        timeout_s=float(getattr(args, "judge_timeout_s", 180.0)),
        clip_frame_count=int(getattr(args, "fine_frame_count", 16)),
        context_s=float(getattr(args, "vlm_context_s", 0.75)),
        boundary_frame_count=int(getattr(args, "judge_boundary_frame_count", 8)),
        max_pair_gap_s=float(getattr(args, "judge_max_pair_gap_s", 0.25)),
    )
