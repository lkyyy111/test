"""Reproducible baseline for temporal segmentation, hand tracking and captions."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from tqdm import tqdm


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
COARSE_STAGES = {"准备", "取放", "清洗", "切配", "烹饪", "整理", "移动", "等待", "其他"}


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


class HandTracker:
    """MediaPipe wrapper; the import is delayed to give a useful install error."""

    def __init__(self, confidence: float) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("缺少 mediapipe。请执行 pip install -r requirements.txt") from exc
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=confidence, min_tracking_confidence=confidence,
        )

    @staticmethod
    def _points(landmarks: Any) -> list[dict[str, float]]:
        return [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in landmarks.landmark]

    def process(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        result = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return []
        detected = []
        world = result.multi_hand_world_landmarks or [None] * len(result.multi_hand_landmarks)
        handedness = result.multi_handedness or [None] * len(result.multi_hand_landmarks)
        for landmarks, world_landmarks, label_data in zip(result.multi_hand_landmarks, world, handedness):
            label = "unknown"
            score = 0.0
            if label_data and label_data.classification:
                label = label_data.classification[0].label.lower()
                score = float(label_data.classification[0].score)
            detected.append({
                "side": label,
                "score": score,
                "landmarks_2d_relative": self._points(landmarks),
                "landmarks_3d_relative": self._points(world_landmarks) if world_landmarks else None,
            })
        return detected

    def close(self) -> None:
        self.hands.close()


def probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(str(video_path.resolve()), fps, frames, width, height, frames / fps)


def image_statistics(frame: np.ndarray, previous: np.ndarray | None) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    change = 0.0 if previous is None else float(np.mean(np.abs(small.astype(np.float32) - previous.astype(np.float32))) / 255.0)
    return sharpness, change


def sample_and_track(info: VideoInfo, sample_fps: float, confidence: float) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(info.path)
    stride = max(1, round(info.fps / sample_fps))
    tracker = HandTracker(confidence)
    samples: list[dict[str, Any]] = []
    previous_small: np.ndarray | None = None
    frame_index = 0
    total = max(1, info.frame_count // stride)
    try:
        with tqdm(total=total, desc="手部检测", unit="sample") as progress:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    sharpness, change = image_statistics(frame, previous_small)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    previous_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
                    hands = tracker.process(frame)
                    samples.append({
                        "frame_index": frame_index,
                        "time_s": round(frame_index / info.fps, 4),
                        "sharpness": round(sharpness, 4),
                        "scene_change": round(change, 6),
                        "hand_present_raw": bool(hands),
                        "hands": hands,
                    })
                    progress.update(1)
                frame_index += 1
    finally:
        tracker.close()
        cap.release()
    if not samples:
        raise ValueError("未能从视频抽取任何帧")
    return samples


def smooth_hand_presence(samples: list[dict[str, Any]], sample_fps: float, gap_tolerance_s: float) -> None:
    """Bridge brief occlusion/out-of-view gaps without inventing hand keypoints.

    A first-person hand can leave the image while the same manipulation continues.
    Only the boolean activity signal is bridged; raw detections and landmarks are
    kept unchanged, so downstream users can distinguish measured from inferred state.
    """
    raw = [bool(sample["hand_present_raw"]) for sample in samples]
    smoothed = raw[:]
    max_gap = max(1, round(gap_tolerance_s * sample_fps))
    index = 0
    while index < len(raw):
        if raw[index]:
            index += 1
            continue
        start = index
        while index < len(raw) and not raw[index]:
            index += 1
        end = index
        # Bridge only a bounded gap surrounded by real hand observations.
        if start > 0 and end < len(raw) and end - start <= max_gap and raw[start - 1] and raw[end]:
            for gap_index in range(start, end):
                smoothed[gap_index] = True
    for sample, raw_value, smooth_value in zip(samples, raw, smoothed):
        sample["hand_present_smoothed"] = smooth_value
        sample["hand_gap_bridged"] = bool(smooth_value and not raw_value)


def wrist_movement(previous: dict[str, Any], current: dict[str, Any]) -> float:
    previous_wrist = {h["side"]: h["landmarks_2d_relative"][0] for h in previous["hands"]}
    current_wrist = {h["side"]: h["landmarks_2d_relative"][0] for h in current["hands"]}
    distances = []
    for side in previous_wrist.keys() & current_wrist.keys():
        a, b = previous_wrist[side], current_wrist[side]
        distances.append(float(np.hypot(a["x"] - b["x"], a["y"] - b["y"])))
    return float(np.mean(distances)) if distances else 0.0


def select_boundaries(scores: list[float], minimum_gap: int, threshold: float) -> list[int]:
    candidates = [i for i, value in enumerate(scores) if i > 0 and value >= threshold]
    boundaries: list[int] = []
    for index in candidates:
        if not boundaries or index - boundaries[-1] >= minimum_gap:
            boundaries.append(index)
    return boundaries


def spans_from_boundaries(boundaries: list[int], count: int) -> list[tuple[int, int]]:
    starts = [0] + sorted(set(boundaries))
    ends = starts[1:] + [count]
    return [(start, end) for start, end in zip(starts, ends) if end > start]


def segment_samples(samples: list[dict[str, Any]], sample_fps: float) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    scene = [float(s["scene_change"]) for s in samples]
    movement = [0.0] + [wrist_movement(a, b) for a, b in zip(samples, samples[1:])]
    scene_threshold = max(0.055, float(np.percentile(scene[1:] or [0.0], 90)))
    move_threshold = max(0.018, float(np.percentile(movement[1:] or [0.0], 85)))
    coarse = select_boundaries(scene, max(1, round(6 * sample_fps)), scene_threshold)
    fine_scores = [max(scene[i] / max(scene_threshold, 1e-6), movement[i] / max(move_threshold, 1e-6)) for i in range(len(samples))]
    fine = select_boundaries(fine_scores, max(1, round(2 * sample_fps)), 1.0)
    # Hand appearance/disappearance is deliberately not a semantic cut boundary:
    # first-person hands are often briefly occluded or outside the field of view.
    return spans_from_boundaries(coarse, len(samples)), spans_from_boundaries(fine, len(samples))


def span_for_time(samples: list[dict[str, Any]], start_s: float, end_s: float) -> tuple[int, int]:
    indexes = [i for i, sample in enumerate(samples) if start_s <= sample["time_s"] <= end_s]
    if not indexes:
        nearest = min(range(len(samples)), key=lambda i: abs(samples[i]["time_s"] - start_s))
        return nearest, min(len(samples), nearest + 1)
    return indexes[0], indexes[-1] + 1


def build_semantic_windows(info: VideoInfo, samples: list[dict[str, Any]], window_s: float, stride_s: float) -> list[dict[str, Any]]:
    """Build overlapping context windows; each label belongs to its center anchor."""
    if stride_s <= 0 or window_s <= 0 or stride_s > window_s:
        raise ValueError("粗粒度窗口参数必须满足 0 < stride_s <= window_s")
    windows = []
    last_start = max(0.0, info.duration_s - window_s)
    starts = list(np.arange(0.0, last_start + 1e-6, stride_s))
    if not starts or abs(starts[-1] - last_start) > 1e-3:
        starts.append(last_start)
    index = 1
    for start_s in starts:
        end_s = min(info.duration_s, start_s + window_s)
        start, end = span_for_time(samples, start_s, end_s)
        portion = samples[start:end]
        windows.append({
            "id": f"coarse_window_{index:03d}", "level": "coarse_window", "start_s": round(start_s, 3),
            "end_s": round(end_s, 3), "anchor_time_s": round((start_s + end_s) / 2, 3), "duration_s": round(end_s - start_s, 3),
            "sample_start": start, "sample_end": end - 1,
            "hand_coverage": round(float(np.mean([s["hand_present_smoothed"] for s in portion])), 3),
        })
        index += 1
    return windows


def annotate_semantic_windows(info: VideoInfo, windows: list[dict[str, Any]], api_base: str | None, model: str | None, frame_count: int) -> None:
    """Annotate each center anchor using overlapping multi-frame temporal context."""
    enabled = bool(os.getenv("VLM_API_KEY") and api_base and model)
    for window in windows:
        if not enabled:
            window["semantic_annotation"] = fallback_annotation(window, "未配置 VLM，无法进行语义粗切分")
            window["caption_zh"] = annotation_caption(window["semantic_annotation"])
            continue
        start, end = window["start_s"], window["end_s"]
        end = min(end, max(0.0, info.duration_s - 1.0 / info.fps))
        frame_times = np.linspace(start, end, frame_count).tolist()
        window["context_frame_times_s"] = [round(time_s, 3) for time_s in frame_times]
        frames = [read_frame_at(info.path, time_s) for time_s in frame_times]
        if any(frame is None for frame in frames):
            window["semantic_annotation"] = fallback_annotation(window, "无法读取粗粒度窗口的三帧")
        else:
            window["semantic_annotation"] = vlm_annotation([frame for frame in frames if frame is not None], window, api_base, model, mode="anchor")
        window["caption_zh"] = annotation_caption(window["semantic_annotation"])


def compatible_semantic_windows(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Merge only adjacent windows that are plausibly the same high-level task."""
    a, b = left["semantic_annotation"], right["semantic_annotation"]
    if a["annotation_source"] != "vlm" or b["annotation_source"] != "vlm":
        return False
    if a["coarse_stage"] != b["coarse_stage"]:
        return False
    if a["subtask"].strip() == b["subtask"].strip():
        return True
    # Stable stages can extend across several windows even when the wording varies.
    if a["coarse_stage"] in {"清洗", "切配", "烹饪", "整理", "等待"}:
        return True
    return bool(set(a["objects"]) & set(b["objects"]))


def refine_boundary_index(samples: list[dict[str, Any]], target_s: float, radius_s: float) -> int:
    """Move a coarse VLM boundary to a nearby pause or visual transition."""
    candidates = [i for i, sample in enumerate(samples) if abs(sample["time_s"] - target_s) <= radius_s]
    if not candidates:
        return min(range(len(samples)), key=lambda i: abs(samples[i]["time_s"] - target_s))
    movements = [0.0] + [wrist_movement(a, b) for a, b in zip(samples, samples[1:])]
    scale_scene = max(float(np.percentile([s["scene_change"] for s in samples], 95)), 1e-5)
    scale_motion = max(float(np.percentile(movements, 95)), 1e-5)
    def score(i: int) -> float:
        scene = min(samples[i]["scene_change"] / scale_scene, 1.0)
        pause = 1.0 - min(movements[i] / scale_motion, 1.0)
        return 0.55 * pause + 0.45 * scene
    return max(candidates, key=score)


def semantic_coarse_spans(
    info: VideoInfo, samples: list[dict[str, Any]], windows: list[dict[str, Any]], args: argparse.Namespace,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """Resolve changes between overlapping semantic anchors into exact boundaries."""
    groups: list[list[dict[str, Any]]] = []
    for window in windows:
        if groups and compatible_semantic_windows(groups[-1][-1], window):
            groups[-1].append(window)
        else:
            groups.append([window])
    boundaries: list[int] = []
    transitions = []
    for left, right in zip(groups, groups[1:]):
        # With overlapping windows the boundary is bracketed by their center anchors.
        interval_start = left[-1]["anchor_time_s"]
        interval_end = right[0]["anchor_time_s"]
        boundary, evidence = resolve_semantic_boundary(
            info, samples, interval_start, interval_end, left[-1]["semantic_annotation"],
            right[0]["semantic_annotation"], args,
        )
        transitions.append(evidence)
        if boundary is not None and 0 < boundary < len(samples):
            boundaries.append(boundary)
    boundaries = sorted(set(boundaries))
    return spans_from_boundaries(boundaries, len(samples)), transitions


def annotation_for_coarse_span(span: tuple[int, int], windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the strongest overlapping window as the coarse segment's task label."""
    start, end = span
    candidates = [window for window in windows if window["sample_start"] < end and window["sample_end"] >= start]
    if not candidates:
        return fallback_annotation({"hand_coverage": 0.0}, "没有重叠的语义窗口")
    return max(candidates, key=lambda window: window["semantic_annotation"]["confidence"])["semantic_annotation"]


def fine_spans_within_coarse(samples: list[dict[str, Any]], coarse_spans: list[tuple[int, int]], sample_fps: float) -> list[tuple[int, int]]:
    """Action-level cuts are computed locally, guaranteeing a nested hierarchy."""
    result: list[tuple[int, int]] = []
    minimum_gap = max(1, round(2 * sample_fps))
    for coarse_start, coarse_end in coarse_spans:
        local = samples[coarse_start:coarse_end]
        if len(local) <= minimum_gap:
            result.append((coarse_start, coarse_end))
            continue
        scene = [float(s["scene_change"]) for s in local]
        movement = [0.0] + [wrist_movement(a, b) for a, b in zip(local, local[1:])]
        scene_threshold = max(0.055, float(np.percentile(scene[1:] or [0.0], 90)))
        move_threshold = max(0.018, float(np.percentile(movement[1:] or [0.0], 85)))
        scores = [max(scene[i] / scene_threshold, movement[i] / move_threshold) for i in range(len(local))]
        local_boundaries = select_boundaries(scores, minimum_gap, 1.0)
        result.extend([(coarse_start + a, coarse_start + b) for a, b in spans_from_boundaries(local_boundaries, len(local))])
    return result


def segment_record(span: tuple[int, int], samples: list[dict[str, Any]], index: int, level: str) -> dict[str, Any]:
    start, end = span
    portion = samples[start:end]
    raw_hand_coverage = float(np.mean([bool(s["hand_present_raw"]) for s in portion]))
    hand_coverage = float(np.mean([bool(s["hand_present_smoothed"]) for s in portion]))
    bridged_coverage = float(np.mean([bool(s["hand_gap_bridged"]) for s in portion]))
    sharpness = float(np.mean([s["sharpness"] for s in portion]))
    return {
        "id": f"{level}_{index:03d}", "level": level,
        "start_s": portion[0]["time_s"], "end_s": portion[-1]["time_s"],
        "duration_s": round(portion[-1]["time_s"] - portion[0]["time_s"], 3),
        "sample_start": start, "sample_end": end - 1,
        "hand_coverage": round(hand_coverage, 3), "raw_hand_coverage": round(raw_hand_coverage, 3),
        "bridged_hand_gap_coverage": round(bridged_coverage, 3), "mean_sharpness": round(sharpness, 2),
        "valid_operation": hand_coverage >= 0.30 and len(portion) >= 2,
    }


def fallback_annotation(segment: dict[str, Any], reason: str = "未配置或未成功调用视觉语言模型") -> dict[str, Any]:
    coverage = segment["hand_coverage"]
    if coverage >= 0.75:
        description = "第一视角近景中，手部持续出现并执行与画面内物体的操作交互。"
    elif coverage >= 0.30:
        description = "第一视角场景中出现间歇性手部活动，疑似处于一个操作子步骤。"
    else:
        description = "第一视角场景片段，未检测到足够稳定的手部操作。"
    return {
        "scene": "未知", "coarse_stage": "其他", "subtask": "未知", "action": {"verb": "未知", "description": description},
        "objects": ["未知物体"], "hand_object_relation": "未知", "temporal_evidence": "仅由手部检测与视频统计生成，未进行视觉语言理解。",
        "confidence": 0.0, "annotation_source": "fallback", "failure_reason": reason,
    }


def normalize_annotation(raw: Any, segment: dict[str, Any]) -> dict[str, Any]:
    """Validate VLM output so every segment uses exactly the same schema."""
    if not isinstance(raw, dict):
        return fallback_annotation(segment, "VLM 响应不是 JSON 对象")
    action = raw.get("action", {})
    if not isinstance(action, dict):
        action = {}
    objects = raw.get("objects", ["未知物体"])
    if isinstance(objects, str):
        objects = [objects]
    if not isinstance(objects, list):
        objects = ["未知物体"]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    coarse_stage = str(raw.get("coarse_stage") or "其他").strip()
    if coarse_stage not in COARSE_STAGES:
        coarse_stage = "其他"
    return {
        "scene": str(raw.get("scene") or "未知"), "coarse_stage": coarse_stage,
        "subtask": str(raw.get("subtask") or "未知"),
        "action": {"verb": str(action.get("verb") or "未知"), "description": str(action.get("description") or "未知")},
        "objects": [str(item) for item in objects if str(item).strip()] or ["未知物体"],
        "hand_object_relation": str(raw.get("hand_object_relation") or "未知"),
        "temporal_evidence": str(raw.get("temporal_evidence") or "未提供"),
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "annotation_source": "vlm",
    }


def annotation_caption(annotation: dict[str, Any]) -> str:
    action = annotation["action"]
    objects = "、".join(annotation["objects"])
    return f"{annotation['subtask']}：{action['verb']}（{action['description']}）；物体：{objects}。"


def parse_json_response(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        content = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        return None
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def vlm_annotation(frames: list[np.ndarray], segment: dict[str, Any], api_base: str | None, model: str | None, mode: str = "segment") -> dict[str, Any]:
    key = os.getenv("VLM_API_KEY")
    if not key or not api_base or not model:
        return fallback_annotation(segment)
    if mode == "anchor":
        temporal_instruction = (
            f"以下是同一第一视角视频的一个 {len(frames)} 帧上下文窗口，按时间顺序排列。"
            f"窗口时间：{segment['start_s']:.2f}s 至 {segment['end_s']:.2f}s；"
            f"请只标注窗口中心锚点 {segment['anchor_time_s']:.2f}s 时正在进行的稳定子任务，首尾帧仅作上下文，不要把短暂过渡误作中心任务。\n"
        )
    else:
        temporal_instruction = (
            f"以下是同一第一视角视频片段按时间顺序排列的 {len(frames)} 帧。"
            f"片段时间：{segment['start_s']:.2f}s 至 {segment['end_s']:.2f}s；手部检测覆盖率：{segment['hand_coverage']:.0%}。\n"
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": (
        temporal_instruction + "\n"
        "只根据可见证据输出一个合法 JSON 对象，不要输出 Markdown、解释或额外字段：\n"
        "{\n"
        '  "scene": "场景或工作台描述",\n'
        '  "coarse_stage": "准备|取放|清洗|切配|烹饪|整理|移动|等待|其他",\n'
        '  "subtask": "该片段的子任务；尽量简短且相似片段用相同措辞",\n'
        '  "action": {"verb": "拿起|移动|放下|打开|关闭|切割|清洁|等待|未知", "description": "简短动作描述"},\n'
        '  "objects": ["被操作物体或未知物体"],\n'
        '  "hand_object_relation": "左手/右手与物体的接触、抓取或移动关系；未知则写未知",\n'
        '  "temporal_evidence": "依据三帧观察到的变化",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "规则：不要猜测画面中不可见的信息；看不清物体时写‘未知物体’；动作必须依据三帧之间的变化；不确定时降低 confidence（0 到 1）。"
    )}]
    for index, frame in enumerate(frames, 1):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return fallback_annotation(segment, f"第 {index} 帧编码失败")
        content.extend([
            {"type": "text", "text": f"帧 {index}"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")}},
        ])
    prompt = (
        "你是严谨的第一视角具身操作视频标注员。必须按用户给定 JSON 模式回答。"
    )
    body = {"model": model, "temperature": 0.1, "max_tokens": 180, "messages": [
        {"role": "system", "content": prompt}, {"role": "user", "content": content},
    ]}
    try:
        response = requests.post(api_base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
        response.raise_for_status()
        raw = parse_json_response(response.json()["choices"][0]["message"]["content"])
        return normalize_annotation(raw, segment)
    except (requests.RequestException, KeyError, IndexError, TypeError) as error:
        print(f"[warning] VLM 标注失败，改用回退 JSON：{error}")
        return fallback_annotation(segment, f"VLM 请求失败：{error}")


def normalize_boundary_labels(raw: Any, times: list[float]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("frame_labels"), list):
        return []
    labels = []
    for time_s, item in zip(times, raw["frame_labels"]):
        if not isinstance(item, dict):
            return []
        stage = str(item.get("coarse_stage") or "其他").strip()
        if stage not in COARSE_STAGES:
            stage = "其他"
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        labels.append({"time_s": round(time_s, 3), "coarse_stage": stage, "subtask": str(item.get("subtask") or "未知"),
                       "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3)})
    return labels if len(labels) == len(times) else []


def vlm_boundary_labels(
    frames: list[np.ndarray], times: list[float], left: dict[str, Any], right: dict[str, Any], api_base: str | None, model: str | None,
) -> list[dict[str, Any]]:
    """Classify densely sampled frames only around a semantic transition candidate."""
    key = os.getenv("VLM_API_KEY")
    if not key or not api_base or not model:
        return []
    content: list[dict[str, Any]] = [{"type": "text", "text": (
        f"这些帧按时间顺序覆盖一个第一视角任务切换候选区间。此前稳定子任务是“{left['subtask']}”（{left['coarse_stage']}），"
        f"之后稳定子任务是“{right['subtask']}”（{right['coarse_stage']}）。\n"
        "请逐帧判断该时刻属于哪个粗粒度阶段。只输出合法 JSON，不要 Markdown 或解释：\n"
        '{"frame_labels":[{"coarse_stage":"准备|取放|清洗|切配|烹饪|整理|移动|等待|其他",'
        '"subtask":"该帧的简短任务", "confidence":0.0}]}'
    )}]
    for index, frame in enumerate(frames, 1):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return []
        content.extend([
            {"type": "text", "text": f"第 {index} 帧，时间 {times[index - 1]:.2f}s"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")}},
        ])
    body = {"model": model, "temperature": 0.0, "max_tokens": 300, "messages": [
        {"role": "system", "content": "你是严谨的视频时序标注员。必须严格输出给定 JSON 模式。"},
        {"role": "user", "content": content},
    ]}
    try:
        response = requests.post(api_base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
        response.raise_for_status()
        return normalize_boundary_labels(parse_json_response(response.json()["choices"][0]["message"]["content"]), times)
    except (requests.RequestException, KeyError, IndexError, TypeError) as error:
        print(f"[warning] VLM 边界复核失败，改用视觉微调：{error}")
        return []


def resolve_semantic_boundary(
    info: VideoInfo, samples: list[dict[str, Any]], interval_start: float, interval_end: float,
    left: dict[str, Any], right: dict[str, Any], args: argparse.Namespace,
) -> tuple[int | None, dict[str, Any]]:
    """Confirm a change with dense temporal labels, then snap it to a local visual pause."""
    times = np.linspace(interval_start, interval_end, args.transition_frame_count).tolist()
    frames = [read_frame_at(info.path, time_s) for time_s in times]
    labels = [] if any(frame is None for frame in frames) else vlm_boundary_labels(
        [frame for frame in frames if frame is not None], times, left, right, args.vlm_api_base, args.vlm_model,
    )
    old_stage, new_stage = left["coarse_stage"], right["coarse_stage"]
    boundary_time: float | None = None
    for index in range(1, len(labels) - args.transition_persistence + 1):
        old_seen = any(label["coarse_stage"] == old_stage for label in labels[:index])
        stable_new = all(label["coarse_stage"] == new_stage for label in labels[index:index + args.transition_persistence])
        if old_seen and stable_new:
            boundary_time = (labels[index - 1]["time_s"] + labels[index]["time_s"]) / 2
            break
    if boundary_time is not None:
        index = refine_boundary_index(samples, boundary_time, min(args.boundary_refinement_radius, (interval_end - interval_start) / 2))
        return index, {"candidate_interval_s": [interval_start, interval_end], "from": left["subtask"], "to": right["subtask"],
                       "source": "dense_vlm_and_visual_refinement", "dense_labels": labels, "confirmed": True,
                       "boundary_time_s": samples[index]["time_s"]}
    # Preserve a candidate boundary when a secondary VLM call is unavailable, but flag it for review.
    fallback_time = (interval_start + interval_end) / 2
    index = refine_boundary_index(samples, fallback_time, min(args.boundary_refinement_radius, (interval_end - interval_start) / 2))
    return index, {"candidate_interval_s": [interval_start, interval_end], "from": left["subtask"], "to": right["subtask"],
                   "source": "visual_refinement_fallback", "dense_labels": labels, "confirmed": False,
                   "boundary_time_s": samples[index]["time_s"], "review_reason": "未能以连续密集 VLM 标签确认任务切换"}


def read_frame_at(video_path: str, time_s: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def write_trajectories(samples: list[dict[str, Any]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["time_s", "frame_index", "side", "x", "y", "z", "world_x", "world_y", "world_z"])
        writer.writeheader()
        for sample in samples:
            for hand in sample["hands"]:
                wrist = hand["landmarks_2d_relative"][0]
                world = (hand.get("landmarks_3d_relative") or [{}])[0]
                writer.writerow({"time_s": sample["time_s"], "frame_index": sample["frame_index"], "side": hand["side"], **wrist,
                                 "world_x": world.get("x"), "world_y": world.get("y"), "world_z": world.get("z")})


def draw_hand(frame: np.ndarray, hand: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    points = hand["landmarks_2d_relative"]
    pixel = [(int(p["x"] * width), int(p["y"] * height)) for p in points]
    color = (38, 214, 120) if hand["side"] == "right" else (255, 170, 40)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pixel[a], pixel[b], color, 2, cv2.LINE_AA)
    for point in pixel:
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    cv2.putText(frame, hand["side"], pixel[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def write_overlay(info: VideoInfo, samples: list[dict[str, Any]], output: Path) -> None:
    lookup = {sample["frame_index"]: sample["hands"] for sample in samples}
    cap = cv2.VideoCapture(info.path)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for hand in lookup.get(index, []):
            draw_hand(frame, hand)
        writer.write(frame)
        index += 1
    cap.release()
    writer.release()


def write_valid_clips(info: VideoInfo, segments: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        if not segment["valid_operation"]:
            continue
        cap = cv2.VideoCapture(info.path)
        start_frame = max(0, round(segment["start_s"] * info.fps))
        end_frame = min(info.frame_count, round(segment["end_s"] * info.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        target = output_dir / f"{segment['id']}_{segment['start_s']:.1f}-{segment['end_s']:.1f}s.mp4"
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
        for _ in range(max(0, end_frame - start_frame)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        cap.release()


def run(args: argparse.Namespace) -> None:
    video = Path(args.video).expanduser()
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    info = probe_video(video)
    print(f"[info] {info.duration_s:.1f}s, {info.width}x{info.height}, {info.fps:.2f} FPS")
    samples = sample_and_track(info, args.sample_fps, args.hand_confidence)
    smooth_hand_presence(samples, args.sample_fps, args.hand_gap_tolerance)
    semantic_windows = build_semantic_windows(info, samples, args.coarse_window_s, args.coarse_stride_s)
    print(f"[info] 标注 {len(semantic_windows)} 个粗粒度语义窗口…")
    annotate_semantic_windows(info, semantic_windows, args.vlm_api_base, args.vlm_model, args.coarse_frame_count)
    vlm_response_success_ratio = float(np.mean([w["semantic_annotation"]["annotation_source"] == "vlm" for w in semantic_windows]))
    if vlm_response_success_ratio >= 0.60:
        coarse_spans, transition_evidence = semantic_coarse_spans(info, samples, semantic_windows, args)
        coarse_method = "overlapping_anchor_vlm_with_dense_transition_confirmation"
    else:
        coarse_spans, _ = segment_samples(samples, args.sample_fps)
        transition_evidence = []
        coarse_method = "visual_scene_change_fallback"
        print("[warning] VLM 语义窗口响应成功率不足 60%，粗切分退化为画面变化候选；请复核。")
    fine_spans = fine_spans_within_coarse(samples, coarse_spans, args.sample_fps)
    coarse = [segment_record(span, samples, i + 1, "coarse") for i, span in enumerate(coarse_spans)]
    fine = [segment_record(span, samples, i + 1, "fine") for i, span in enumerate(fine_spans)]
    for record, span in zip(coarse, coarse_spans):
        record["semantic_annotation"] = annotation_for_coarse_span(span, semantic_windows)
        record["caption_zh"] = annotation_caption(record["semantic_annotation"])
        record["segmentation_method"] = coarse_method
    for record in fine:
        parent = next(coarse_record for coarse_record in coarse if coarse_record["sample_start"] <= record["sample_start"] <= coarse_record["sample_end"])
        record["parent_coarse_id"] = parent["id"]
    review_queue = []
    for transition in transition_evidence:
        if not transition["confirmed"]:
            review_queue.append({"level": "coarse_transition", "start_s": transition["candidate_interval_s"][0],
                                 "end_s": transition["candidate_interval_s"][1], "reasons": [transition["review_reason"]]})
    for segment in coarse:
        annotation = segment["semantic_annotation"]
        reasons = []
        if annotation["confidence"] < args.review_confidence:
            reasons.append(f"粗粒度 VLM 置信度 {annotation['confidence']:.2f} 低于阈值 {args.review_confidence:.2f}")
        if annotation["annotation_source"] != "vlm":
            reasons.append(annotation.get("failure_reason", "粗粒度使用回退标注"))
        if reasons:
            review_queue.append({"level": "coarse", "segment_id": segment["id"], "start_s": segment["start_s"], "end_s": segment["end_s"], "reasons": reasons})
    for segment in fine:
        start, end = segment["start_s"], segment["end_s"]
        frame_times = [start, (start + end) / 2, end]
        frames = [read_frame_at(info.path, time_s) for time_s in frame_times]
        if any(frame is None for frame in frames):
            annotation = fallback_annotation(segment, "无法读取片段的起始、中间或结束帧")
        else:
            annotation = vlm_annotation([frame for frame in frames if frame is not None], segment, args.vlm_api_base, args.vlm_model)
        segment["semantic_annotation"] = annotation
        segment["caption_zh"] = annotation_caption(annotation)
        reasons = []
        if annotation["confidence"] < args.review_confidence:
            reasons.append(f"VLM 置信度 {annotation['confidence']:.2f} 低于阈值 {args.review_confidence:.2f}")
        if annotation["annotation_source"] != "vlm":
            reasons.append(annotation.get("failure_reason", "使用回退标注"))
        if "未知" in annotation["action"]["verb"] or any("未知" in item for item in annotation["objects"]):
            reasons.append("动作或关键物体未知")
        segment["needs_review"] = bool(reasons)
        segment["review_reasons"] = reasons
        if reasons:
            review_queue.append({"level": "fine", "segment_id": segment["id"], "start_s": segment["start_s"], "end_s": segment["end_s"], "reasons": reasons})
    payload = {"schema_version": "0.2", "video": asdict(info), "parameters": vars(args), "segmentation": {
        "coarse_method": coarse_method, "semantic_window_count": len(semantic_windows), "vlm_response_success_ratio": round(vlm_response_success_ratio, 3),
        "semantic_window_label": "center_anchor", "transition_evidence": transition_evidence,
        "fine_method": "local_hand_motion_and_visual_change_within_coarse_segments"
    }, "coordinate_system": {
        "image": "x/y normalized to [0,1], z is MediaPipe relative depth", "world": "MediaPipe relative metric hand coordinates; not camera-calibrated"
    }, "semantic_windows": semantic_windows, "coarse_segments": coarse, "fine_segments": fine, "valid_segments": [s["id"] for s in fine if s["valid_operation"]], "review_queue": review_queue}
    (output / "annotations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "hand_landmarks.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trajectories(samples, output / "wrist_trajectories.csv")
    if not args.skip_video_outputs:
        print("[info] 写入骨架叠加视频…")
        write_overlay(info, samples, output / "hand_overlay.mp4")
        print("[info] 导出有效操作片段…")
        write_valid_clips(info, fine, output / "valid_segments")
    print(f"[done] 有效操作片段：{len(payload['valid_segments'])}/{len(fine)}；待复核：{len(review_queue)}；结果目录：{output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一视角视频：切分、语言标注、手部姿态和轨迹")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="分析采样帧率，默认 4")
    parser.add_argument("--hand-confidence", type=float, default=0.55, help="MediaPipe 手部置信度阈值")
    parser.add_argument("--hand-gap-tolerance", type=float, default=1.5, help="短暂手部遮挡/出画的连续容忍时间（秒），默认 1.5")
    parser.add_argument("--coarse-window-s", type=float, default=8.0, help="VLM 语义上下文窗口长度（秒），默认 8")
    parser.add_argument("--coarse-stride-s", type=float, default=4.0, help="相邻粗语义窗口的中心锚点步长（秒），默认 4")
    parser.add_argument("--coarse-frame-count", type=int, default=5, help="每个粗语义窗口取帧数，默认 5")
    parser.add_argument("--transition-frame-count", type=int, default=5, help="候选任务边界的密集 VLM 复核帧数，默认 5")
    parser.add_argument("--transition-persistence", type=int, default=2, help="确认新粗阶段所需连续密集标签数，默认 2")
    parser.add_argument("--boundary-refinement-radius", type=float, default=4.0, help="VLM 粗边界向停顿/画面变化微调的搜索半径（秒），默认 4")
    parser.add_argument("--vlm-api-base", default=None, help="兼容 Chat Completions 的 API 基地址，例如 https://host/v1")
    parser.add_argument("--vlm-model", default=None, help="视觉语言模型名称；密钥从 VLM_API_KEY 读取")
    parser.add_argument("--review-confidence", type=float, default=0.65, help="低于该 VLM 置信度的片段进入 review_queue，默认 0.65")
    parser.add_argument("--skip-video-outputs", action="store_true", help="仅输出 JSON/CSV，不生成 MP4")
    return parser


def main() -> None:
    run(build_parser().parse_args())
