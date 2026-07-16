"""Fine-grained egocentric video annotation pipeline.

The pipeline deliberately treats kinematic boundaries as high-recall candidates:
camera-compensated hand-speed valleys propose atomic clips, a VLM captions those
clips, and adjacent clips with the same semantics are merged.
"""
from __future__ import annotations

import argparse
import base64
import csv
import itertools
import json
import math
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
HAND_SIDES = ("left", "right")
PALM_LANDMARKS = (0, 5, 9, 13, 17)
CAMERA_MOTION_SIZE = (320, 180)


def correct_mediapipe_handedness(label: str) -> str:
    """Map MediaPipe's mirrored-input handedness to a non-mirrored egocentric view."""
    normalized = str(label).lower()
    if normalized == "left":
        return "right"
    if normalized == "right":
        return "left"
    return "unknown"


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


class HandTracker:
    """MediaPipe wrapper returning at most two hands and 21 points per hand."""

    def __init__(self, confidence: float) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("缺少 mediapipe。请执行 pip install -r requirements.txt") from exc
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.last_invalid_count = 0

    @staticmethod
    def _points(landmarks: Any) -> list[dict[str, float]]:
        return [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in landmarks.landmark]

    @staticmethod
    def _valid_points(points: list[dict[str, float]]) -> bool:
        return len(points) == 21 and all(
            math.isfinite(point[axis]) for point in points for axis in ("x", "y", "z")
        )

    def process(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        self.last_invalid_count = 0
        result = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return []
        count = len(result.multi_hand_landmarks)
        world = list(result.multi_hand_world_landmarks or [])
        handedness = list(result.multi_handedness or [])
        world.extend([None] * (count - len(world)))
        handedness.extend([None] * (count - len(handedness)))
        detected: list[dict[str, Any]] = []
        for landmarks, world_landmarks, label_data in zip(result.multi_hand_landmarks, world, handedness):
            points_2d = self._points(landmarks)
            if not self._valid_points(points_2d):
                self.last_invalid_count += 1
                continue
            points_3d = self._points(world_landmarks) if world_landmarks else None
            if points_3d is not None and not self._valid_points(points_3d):
                points_3d = None
            mediapipe_side, score = "unknown", 0.0
            if label_data and label_data.classification:
                mediapipe_side = label_data.classification[0].label.lower()
                score = float(label_data.classification[0].score)
            if mediapipe_side not in HAND_SIDES:
                mediapipe_side = "unknown"
            detector_side = correct_mediapipe_handedness(mediapipe_side)
            if not math.isfinite(score):
                score = 0.0
            detected.append({
                "mediapipe_side": mediapipe_side,
                "detector_side": detector_side,
                "raw_side": mediapipe_side,
                "side": detector_side,
                "score": round(score, 4),
                "landmarks_2d_relative": points_2d,
                "landmarks_3d_relative": points_3d,
            })
        return detected

    def close(self) -> None:
        self.hands.close()


def palm_center(hand: dict[str, Any]) -> np.ndarray:
    points = hand["landmarks_2d_relative"]
    return np.mean([[points[i]["x"], points[i]["y"]] for i in PALM_LANDMARKS], axis=0)


class HandIdentityTracker:
    """Stabilize MediaPipe handedness with short-term spatial continuity."""

    def __init__(self, max_gap_s: float = 1.0) -> None:
        self.max_gap_s = max_gap_s
        self.last: dict[str, tuple[np.ndarray, float]] = {}

    def assign(self, hands: list[dict[str, Any]], time_s: float) -> list[dict[str, Any]]:
        if not hands:
            return hands
        valid_hands_and_centers = []
        for hand in hands:
            center = palm_center(hand)
            if center.shape == (2,) and np.all(np.isfinite(center)):
                valid_hands_and_centers.append((hand, center))
        # MediaPipe is configured for two hands, but cap defensively in case a
        # backend returns malformed extra detections.
        valid_hands_and_centers.sort(key=lambda item: float(item[0].get("score", 0.0)), reverse=True)
        valid_hands_and_centers = valid_hands_and_centers[:len(HAND_SIDES)]
        if not valid_hands_and_centers:
            return []
        hands = [item[0] for item in valid_hands_and_centers]
        centers = [item[1] for item in valid_hands_and_centers]
        assignments = itertools.permutations(HAND_SIDES, len(hands))
        best_slots: tuple[str, ...] | None = None
        best_cost = float("inf")
        for slots in assignments:
            cost = 0.0
            for hand, center, slot in zip(hands, centers, slots):
                raw_side = hand.get("detector_side", hand.get("raw_side", "unknown"))
                raw_score = float(hand.get("score", 0.0))
                if not math.isfinite(raw_score):
                    raw_score = 0.0
                if raw_side in HAND_SIDES and raw_side != slot:
                    cost += 0.25 * max(0.5, raw_score)
                previous = self.last.get(slot)
                if previous and time_s - previous[1] <= self.max_gap_s:
                    cost += float(np.linalg.norm(center - previous[0]))
                elif raw_side not in HAND_SIDES:
                    cost += 0.1
            if math.isfinite(cost) and cost < best_cost:
                best_cost, best_slots = cost, slots
        if best_slots is None:
            # This should only be reachable for unexpected backend values. Keep
            # the pipeline alive with a deterministic raw-label-first mapping.
            available = list(HAND_SIDES)
            fallback: list[str] = []
            for hand in hands:
                raw_side = hand.get("detector_side", hand.get("raw_side"))
                slot = raw_side if raw_side in available else available[0]
                fallback.append(slot)
                available.remove(slot)
            best_slots = tuple(fallback)
        for hand, center, slot in zip(hands, centers, best_slots):
            hand["side"] = slot
            hand["track_id"] = slot
            hand["palm_center"] = {"x": round(float(center[0]), 6), "y": round(float(center[1]), 6)}
            hand["valid_mask"] = True
            self.last[slot] = (center, time_s)
        return hands


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


def image_statistics(gray_small: np.ndarray, previous: np.ndarray | None) -> tuple[float, float]:
    sharpness = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())
    change = 0.0 if previous is None else float(
        np.mean(np.abs(gray_small.astype(np.float32) - previous.astype(np.float32))) / 255.0
    )
    return sharpness, change


def _hand_exclusion_mask(hands: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
    mask = np.full((height, width), 255, dtype=np.uint8)
    for hand in hands:
        points = hand["landmarks_2d_relative"]
        xs = [p["x"] * width for p in points]
        ys = [p["y"] * height for p in points]
        margin = 0.08 * max(width, height)
        x0, x1 = max(0, int(min(xs) - margin)), min(width, int(max(xs) + margin))
        y0, y1 = max(0, int(min(ys) - margin)), min(height, int(max(ys) + margin))
        mask[y0:y1, x0:x1] = 0
    return mask


def estimate_camera_motion(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    previous_hands: list[dict[str, Any]],
) -> tuple[np.ndarray, float, int]:
    """Estimate previous-to-current background affine motion with LK + RANSAC."""
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if previous_gray is None:
        return identity, 0.0, 0
    height, width = previous_gray.shape
    mask = _hand_exclusion_mask(previous_hands, width, height)
    points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7, mask=mask,
    )
    if points is None or len(points) < 8:
        return identity, 0.0, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return identity, 0.0, 0
    good_previous = points[status.reshape(-1) == 1].reshape(-1, 2)
    good_current = tracked[status.reshape(-1) == 1].reshape(-1, 2)
    if len(good_previous) < 8:
        return identity, 0.0, len(good_previous)
    affine, inliers = cv2.estimateAffinePartial2D(
        good_previous, good_current, method=cv2.RANSAC, ransacReprojThreshold=2.5,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if affine is None or inliers is None:
        return identity, 0.0, len(good_previous)
    inlier_ratio = float(np.mean(inliers))
    coverage = min(1.0, len(good_previous) / 50.0)
    quality = inlier_ratio * coverage
    return affine.astype(np.float32), quality, len(good_previous)


def sample_and_track(info: VideoInfo, sample_fps: float, confidence: float) -> list[dict[str, Any]]:
    if sample_fps <= 0:
        raise ValueError("--sample-fps 必须大于 0")
    cap = cv2.VideoCapture(info.path)
    tracker = HandTracker(confidence)
    identity_tracker = HandIdentityTracker()
    samples: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    previous_hands: list[dict[str, Any]] = []
    next_sample_s = 0.0
    frame_index = 0
    total = max(1, math.ceil(info.duration_s * sample_fps))
    try:
        with tqdm(total=total, desc="手部检测", unit="sample") as progress:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                time_s = frame_index / info.fps
                if time_s + 1e-9 >= next_sample_s:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray, CAMERA_MOTION_SIZE, interpolation=cv2.INTER_AREA)
                    sharpness, change = image_statistics(gray_small, previous_gray)
                    hands = identity_tracker.assign(tracker.process(frame), time_s)
                    affine, camera_quality, feature_count = estimate_camera_motion(
                        previous_gray, gray_small, previous_hands,
                    )
                    samples.append({
                        "frame_index": frame_index,
                        "time_s": round(time_s, 6),
                        "sharpness": round(sharpness, 4),
                        "scene_change_diagnostic": round(change, 6),
                        "camera_motion_from_previous": [[round(float(v), 7) for v in row] for row in affine],
                        "camera_motion_quality": round(camera_quality, 4),
                        "camera_feature_count": feature_count,
                        "invalid_hand_detection_count": tracker.last_invalid_count,
                        "hand_present_raw": bool(hands),
                        "hands": hands,
                    })
                    previous_gray, previous_hands = gray_small, hands
                    progress.update(1)
                    next_sample_s += 1.0 / sample_fps
                    while next_sample_s <= time_s + 1e-9:
                        next_sample_s += 1.0 / sample_fps
                frame_index += 1
    finally:
        tracker.close()
        cap.release()
    if not samples:
        raise ValueError("未能从视频抽取任何帧")
    return samples


def _hand_for_side(sample: dict[str, Any], side: str) -> dict[str, Any] | None:
    return next((hand for hand in sample["hands"] if hand.get("side") == side), None)


def annotate_hand_validity(samples: list[dict[str, Any]], gap_tolerance_s: float) -> None:
    """Attach per-hand observed/smoothed masks without inventing landmarks."""
    for sample in samples:
        sample["hand_validity"] = {}
    for side in HAND_SIDES:
        observed = [_hand_for_side(sample, side) is not None for sample in samples]
        smoothed = observed[:]
        index = 0
        while index < len(samples):
            if observed[index]:
                index += 1
                continue
            start = index
            while index < len(samples) and not observed[index]:
                index += 1
            end = index
            if start > 0 and end < len(samples):
                left_t = samples[start - 1]["time_s"]
                right_t = samples[end]["time_s"]
                if right_t - left_t <= gap_tolerance_s:
                    for gap_index in range(start, end):
                        smoothed[gap_index] = True
        for sample, raw, smooth in zip(samples, observed, smoothed):
            sample["hand_validity"][side] = {
                "observed": raw,
                "smoothed_presence": smooth,
                "interpolated_presence": bool(smooth and not raw),
                "valid_for_boundary": raw,
            }
    for sample in samples:
        raw = any(sample["hand_validity"][side]["observed"] for side in HAND_SIDES)
        smooth = any(sample["hand_validity"][side]["smoothed_presence"] for side in HAND_SIDES)
        sample["hand_present_raw"] = raw
        sample["hand_present_smoothed"] = smooth
        sample["hand_gap_bridged"] = bool(smooth and not raw)
        # Dense, side-stable representation: every timestamp always contains two
        # slots. Missing observations keep a false mask and null landmarks; we do
        # not fabricate 21 points across an occlusion.
        sample["hand_tracks"] = {}
        for side in HAND_SIDES:
            hand = _hand_for_side(sample, side)
            validity = sample["hand_validity"][side]
            sample["hand_tracks"][side] = {
                **validity,
                "valid_mask": bool(validity["observed"]),
                "landmarks_2d_relative": hand["landmarks_2d_relative"] if hand else None,
                "landmarks_3d_relative": hand.get("landmarks_3d_relative") if hand else None,
            }


def _smooth_finite_runs(values: np.ndarray, times: np.ndarray, radius_s: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    index = 0
    while index < len(values):
        if not finite[index]:
            index += 1
            continue
        start = index
        while index < len(values) and finite[index]:
            index += 1
        end = index
        run = values[start:end].astype(np.float64)
        median_filtered = run.copy()
        for local in range(len(run)):
            a, b = max(0, local - 1), min(len(run), local + 2)
            median_filtered[local] = float(np.median(run[a:b]))
        for local, global_index in enumerate(range(start, end)):
            distances = np.abs(times[start:end] - times[global_index])
            neighbors = distances <= radius_s
            if not np.any(neighbors):
                result[global_index] = median_filtered[local]
                continue
            sigma = max(radius_s / 2.0, 1e-4)
            weights = np.exp(-0.5 * (distances[neighbors] / sigma) ** 2)
            result[global_index] = float(np.average(median_filtered[neighbors], weights=weights))
    return result


def _camera_step_usable(sample: dict[str, Any], min_camera_quality: float) -> bool:
    quality = float(sample["camera_motion_quality"])
    low_scene_change = float(sample["scene_change_diagnostic"]) <= 0.025
    return quality >= min_camera_quality or low_scene_change


def _apply_affine(affine: np.ndarray, point: np.ndarray) -> np.ndarray:
    return affine @ np.array([point[0], point[1], 1.0], dtype=np.float64)


def _boundary_centers(
    samples: list[dict[str, Any]], side: str, min_camera_quality: float,
    max_gap_s: float, max_displacement: float,
) -> tuple[list[np.ndarray | None], list[str | None], np.ndarray]:
    """Bridge short palm gaps only for motion analysis, never as real pose observations."""
    width, height = CAMERA_MOTION_SIZE
    diagonal = math.hypot(width, height)
    times = np.array([sample["time_s"] for sample in samples], dtype=np.float64)
    centers: list[np.ndarray | None] = []
    sources: list[str | None] = []
    weights = np.zeros(len(samples), dtype=np.float64)
    for sample in samples:
        hand = _hand_for_side(sample, side)
        if hand is None:
            centers.append(None)
            sources.append(None)
        else:
            centers.append(palm_center(hand) * np.array([width, height], dtype=np.float64))
            sources.append("observed")
            weights[len(centers) - 1] = 1.0

    index = 0
    while index < len(samples):
        if centers[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < len(samples) and centers[index] is None:
            index += 1
        gap_end = index
        left, right = gap_start - 1, gap_end
        if left < 0 or right >= len(samples) or times[right] - times[left] > max_gap_s:
            continue
        if not all(_camera_step_usable(samples[i], min_camera_quality) for i in range(left + 1, right + 1)):
            continue

        forward: dict[int, np.ndarray] = {left: centers[left].copy()}  # type: ignore[union-attr]
        for i in range(left + 1, right + 1):
            affine = np.asarray(samples[i]["camera_motion_from_previous"], dtype=np.float64)
            forward[i] = _apply_affine(affine, forward[i - 1])
        residual = float(np.linalg.norm(centers[right] - forward[right]) / diagonal)  # type: ignore[operator]
        if residual > max_displacement:
            continue

        backward: dict[int, np.ndarray] = {right: centers[right].copy()}  # type: ignore[union-attr]
        invertible = True
        for i in range(right - 1, left, -1):
            affine = np.vstack([
                np.asarray(samples[i + 1]["camera_motion_from_previous"], dtype=np.float64),
                [0.0, 0.0, 1.0],
            ])
            try:
                inverse = np.linalg.inv(affine)[:2]
            except np.linalg.LinAlgError:
                invertible = False
                break
            backward[i] = _apply_affine(inverse, backward[i + 1])
        if not invertible:
            continue
        for i in range(gap_start, gap_end):
            alpha = float((times[i] - times[left]) / max(times[right] - times[left], 1e-6))
            centers[i] = (1.0 - alpha) * forward[i] + alpha * backward[i]
            sources[i] = "interpolated_for_motion"
            weights[i] = 0.4
    return centers, sources, weights


def attach_hand_motion(
    samples: list[dict[str, Any]], min_camera_quality: float, smoothing_radius_s: float,
    interpolation_gap_s: float, interpolation_max_displacement: float,
) -> None:
    """Compute camera-compensated palm speed with low-weight short-gap bridging."""
    width, height = CAMERA_MOTION_SIZE
    diagonal = math.hypot(width, height)
    times = np.array([sample["time_s"] for sample in samples], dtype=np.float64)
    for sample in samples:
        sample["hand_motion"] = {
            side: {
                "raw_speed": None, "smoothed_speed": None, "valid": False,
                "source": None, "weight": 0.0,
            } for side in HAND_SIDES
        }
    for side in HAND_SIDES:
        centers, center_sources, center_weights = _boundary_centers(
            samples, side, min_camera_quality, interpolation_gap_s,
            interpolation_max_displacement,
        )
        raw = np.full(len(samples), np.nan, dtype=np.float64)
        speed_weights = np.zeros(len(samples), dtype=np.float64)
        speed_sources: list[str | None] = [None] * len(samples)
        for index, sample in enumerate(samples):
            track = sample["hand_tracks"][side]
            center = centers[index]
            track["boundary_palm_center"] = (
                {
                    "x": round(float(center[0] / width), 6),
                    "y": round(float(center[1] / height), 6),
                } if center is not None else None
            )
            track["interpolated_for_motion"] = center_sources[index] == "interpolated_for_motion"
            track["boundary_motion_available"] = center is not None
        for index in range(1, len(samples)):
            previous_center, current_center = centers[index - 1], centers[index]
            if previous_center is None or current_center is None:
                continue
            dt = times[index] - times[index - 1]
            if dt <= 0 or not _camera_step_usable(samples[index], min_camera_quality):
                continue
            affine = np.asarray(samples[index]["camera_motion_from_previous"], dtype=np.float64)
            predicted = _apply_affine(affine, previous_center)
            raw[index] = float(np.linalg.norm(current_center - predicted) / diagonal / dt)
            speed_weights[index] = min(center_weights[index - 1], center_weights[index])
            speed_sources[index] = (
                "observed" if center_sources[index - 1] == center_sources[index] == "observed"
                else "interpolated_for_motion"
            )
        smoothed = _smooth_finite_runs(raw, times, smoothing_radius_s)
        for index, sample in enumerate(samples):
            item = sample["hand_motion"][side]
            if np.isfinite(raw[index]):
                item["raw_speed"] = round(float(raw[index]), 7)
            if np.isfinite(smoothed[index]):
                item["smoothed_speed"] = round(float(smoothed[index]), 7)
                item["valid"] = True
                item["source"] = speed_sources[index]
                item["weight"] = round(float(speed_weights[index]), 3)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    cutoff = float(np.sum(ordered_weights)) / 2.0
    return float(ordered_values[min(int(np.searchsorted(np.cumsum(ordered_weights), cutoff)), len(values) - 1)])


def _series_velocity_candidates(
    samples: list[dict[str, Any]], args: argparse.Namespace, label: str,
    speeds: np.ndarray, weights: np.ndarray, observed: np.ndarray,
    hands_at_index: list[list[str]], source: str,
) -> list[dict[str, Any]]:
    times = np.array([sample["time_s"] for sample in samples], dtype=np.float64)
    finite_values = speeds[np.isfinite(speeds)]
    if len(finite_values) < 6:
        return []
    q10, q90 = np.percentile(finite_values, [10, 90])
    speed_range = max(float(q90 - q10), 1e-6)
    move_threshold = float(q10 + 0.30 * speed_range)
    candidates: list[dict[str, Any]] = []
    for index in range(1, len(samples) - 1):
        if not np.isfinite(speeds[index]) or weights[index] <= 0:
            continue
        center = np.where(np.abs(times - times[index]) <= args.velocity_center_s)[0]
        left = np.where(
            (times >= times[index] - args.velocity_context_s)
            & (times <= times[index] - args.velocity_center_s)
        )[0]
        right = np.where(
            (times >= times[index] + args.velocity_center_s)
            & (times <= times[index] + args.velocity_context_s)
        )[0]

        def window(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            keep = np.isfinite(speeds[indices]) & (weights[indices] > 0)
            values, value_weights = speeds[indices][keep], weights[indices][keep]
            coverage = float(np.sum(value_weights) / max(1, len(indices)))
            return values, value_weights, coverage

        center_values, _, center_coverage = window(center)
        left_values, left_weights, left_coverage = window(left)
        right_values, right_weights, right_coverage = window(right)
        if (
            len(center_values) == 0
            or len(left_values) == 0
            or len(right_values) == 0
            or min(left_coverage, right_coverage) < args.velocity_min_window_weight
        ):
            continue
        support_s = args.motion_interpolation_gap_s
        real_before = np.any(observed & (times >= times[index] - support_s) & (times <= times[index]))
        real_after = np.any(observed & (times >= times[index]) & (times <= times[index] + support_s))
        if not real_before or not real_after:
            continue
        valley = float(np.min(center_values))
        if speeds[index] > valley + 1e-10:
            continue
        before = _weighted_median(left_values, left_weights)
        after = _weighted_median(right_values, right_weights)
        drop_before = max(0.0, (before - valley) / max(before, 1e-6))
        drop_after = max(0.0, (after - valley) / max(after, 1e-6))
        prominence = max(0.0, (min(before, after) - valley) / speed_range)
        if before <= move_threshold or after <= move_threshold:
            continue
        if min(drop_before, drop_after) < args.velocity_drop_ratio:
            continue
        if prominence < args.velocity_prominence:
            continue
        support_weight = min(left_coverage, right_coverage, max(center_coverage, weights[index]))
        shape_score = 0.5 * min(drop_before, drop_after) + 0.5 * min(prominence / 0.5, 1.0)
        score = shape_score * (0.7 + 0.3 * support_weight)
        candidates.append({
            "index": index,
            "time_s": round(float(times[index]), 6),
            "hands": hands_at_index[index],
            "source": source,
            "score": round(float(score), 4),
            "hard_boundary": False,
            "evidence": [{
                "hand": label,
                "speed_before": round(before, 7),
                "speed_minimum": round(valley, 7),
                "speed_after": round(after, 7),
                "drop_before": round(drop_before, 4),
                "drop_after": round(drop_after, 4),
                "normalized_prominence": round(prominence, 4),
                "move_threshold": round(move_threshold, 7),
                "left_window_weight": round(left_coverage, 3),
                "right_window_weight": round(right_coverage, 3),
                "center_window_weight": round(center_coverage, 3),
                "motion_source": "interpolated_supported" if weights[index] < 1.0 else "observed",
                "camera_motion_quality": samples[index]["camera_motion_quality"],
            }],
        })
    retained: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["time_s"]):
        if not retained or candidate["time_s"] - retained[-1]["time_s"] >= args.velocity_min_gap_s:
            retained.append(candidate)
        elif candidate["score"] > retained[-1]["score"]:
            retained[-1] = candidate
    return retained


def find_velocity_candidates(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Find weighted per-hand and global activity valleys, then fuse nearby proposals."""
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    hands_for_side = {side: [[side] for _ in samples] for side in HAND_SIDES}
    all_candidates: list[dict[str, Any]] = []
    for side in HAND_SIDES:
        speeds = np.array([
            sample["hand_motion"][side]["smoothed_speed"]
            if sample["hand_motion"][side]["smoothed_speed"] is not None else np.nan
            for sample in samples
        ], dtype=np.float64)
        weights = np.array([
            float(sample["hand_motion"][side].get("weight", 1.0 if np.isfinite(speeds[index]) else 0.0))
            for index, sample in enumerate(samples)
        ], dtype=np.float64)
        observed = np.array([
            bool(sample.get("hand_validity", {}).get(side, {}).get("observed", np.isfinite(speeds[index])))
            for index, sample in enumerate(samples)
        ], dtype=bool)
        series[side] = (speeds, weights, observed)
        all_candidates.extend(_series_velocity_candidates(
            samples, args, side, speeds, weights, observed, hands_for_side[side],
            "camera_compensated_wrist_speed_minimum",
        ))

    normalized: dict[str, np.ndarray] = {}
    for side, (speeds, _, _) in series.items():
        finite = speeds[np.isfinite(speeds)]
        scale_low, scale_high = np.percentile(finite, [10, 90]) if len(finite) >= 6 else (0.0, 1.0)
        # Keep values below the 10th percentile distinct; clipping them to zero
        # would create an artificial flat valley and shift a boundary earlier.
        normalized[side] = (speeds - scale_low) / max(float(scale_high - scale_low), 1e-6)
    global_speed = np.full(len(samples), np.nan, dtype=np.float64)
    global_weight = np.zeros(len(samples), dtype=np.float64)
    global_observed = np.zeros(len(samples), dtype=bool)
    global_hands: list[list[str]] = []
    for index in range(len(samples)):
        available = [side for side in HAND_SIDES if np.isfinite(normalized[side][index])]
        global_hands.append(available)
        if not available:
            continue
        active = max(available, key=lambda side: normalized[side][index])
        global_speed[index] = normalized[active][index]
        global_weight[index] = series[active][1][index]
        global_observed[index] = any(series[side][2][index] for side in available)
    all_candidates.extend(_series_velocity_candidates(
        samples, args, "global_activity", global_speed, global_weight, global_observed,
        global_hands, "camera_compensated_global_activity_minimum",
    ))

    # Fuse per-hand/global candidates that describe the same pause.
    fused: list[dict[str, Any]] = []
    pending = sorted(all_candidates, key=lambda item: item["time_s"])
    while pending:
        first = pending.pop(0)
        group = [first]
        while pending and pending[0]["time_s"] - first["time_s"] <= args.hand_boundary_fusion_s:
            group.append(pending.pop(0))
        best = max(group, key=lambda item: item["score"])
        fused.append({
            **best,
            "hands": sorted({hand for item in group for hand in item["hands"]}),
            "score": round(max(item["score"] for item in group) + (0.1 if len(group) > 1 else 0.0), 4),
            "evidence": [evidence for item in group for evidence in item["evidence"]],
        })
    return fused


def _long_no_hand_boundaries(
    samples: list[dict[str, Any]], threshold_s: float, video_duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Technical boundaries isolate long no-hand intervals from exportable clips."""
    result: list[dict[str, Any]] = []
    if len(samples) > 1:
        sample_period_s = float(np.median(np.diff([sample["time_s"] for sample in samples])))
    else:
        sample_period_s = 0.0
    sampled_end_s = samples[-1]["time_s"] + sample_period_s
    stream_end_s = min(sampled_end_s, video_duration_s) if video_duration_s is not None else sampled_end_s
    index = 0
    while index < len(samples):
        if samples[index]["hand_present_smoothed"]:
            index += 1
            continue
        start = index
        while index < len(samples) and not samples[index]["hand_present_smoothed"]:
            index += 1
        end = index
        start_s = samples[start]["time_s"]
        end_s = samples[end]["time_s"] if end < len(samples) else stream_end_s
        if end_s - start_s < threshold_s:
            continue
        for boundary_index in (start, end):
            if 0 < boundary_index < len(samples):
                result.append({
                    "index": boundary_index,
                    "time_s": samples[boundary_index]["time_s"],
                    "hands": [],
                    "source": "long_no_hand_export_boundary",
                    "score": 1.0,
                    "hard_boundary": True,
                    "evidence": [{"no_hand_interval_s": [start_s, end_s]}],
                })
    return result


def _add_analysis_windows(
    samples: list[dict[str, Any]], candidates: list[dict[str, Any]], max_duration_s: float,
) -> list[dict[str, Any]]:
    """Bound very long VLM inputs without claiming the added cuts are semantic."""
    by_index = {candidate["index"]: candidate for candidate in candidates}
    boundaries = [0] + sorted(by_index) + [len(samples)]
    additions: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        start_s = samples[start]["time_s"] if start < len(samples) else samples[-1]["time_s"]
        end_s = samples[end]["time_s"] if end < len(samples) else samples[-1]["time_s"]
        next_time = start_s + max_duration_s
        while next_time < end_s - 1e-6:
            index = min(range(start + 1, end), key=lambda i: abs(samples[i]["time_s"] - next_time))
            additions.append({
                "index": index,
                "time_s": samples[index]["time_s"],
                "hands": [],
                "source": "vlm_analysis_window",
                "score": 0.0,
                "hard_boundary": False,
                "evidence": [{"semantic_boundary": False}],
            })
            next_time += max_duration_s
    return candidates + additions


def consolidate_boundaries(candidates: list[dict[str, Any]], min_gap_s: float) -> list[dict[str, Any]]:
    """Combine duplicate technical/kinematic boundaries and suppress tiny fragments."""
    combined: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        index = candidate["index"]
        if index not in combined:
            combined[index] = candidate
            continue
        old = combined[index]
        winner = candidate if (candidate["hard_boundary"], candidate["score"]) > (old["hard_boundary"], old["score"]) else old
        winner = dict(winner)
        winner["evidence"] = old["evidence"] + candidate["evidence"]
        winner["hard_boundary"] = old["hard_boundary"] or candidate["hard_boundary"]
        combined[index] = winner
    accepted: list[dict[str, Any]] = []
    for candidate in sorted(combined.values(), key=lambda item: item["time_s"]):
        if not accepted or candidate["time_s"] - accepted[-1]["time_s"] >= min_gap_s:
            accepted.append(candidate)
            continue
        previous = accepted[-1]
        if candidate["hard_boundary"] and not previous["hard_boundary"]:
            accepted[-1] = candidate
        elif candidate["hard_boundary"] == previous["hard_boundary"] and candidate["score"] > previous["score"]:
            accepted[-1] = candidate
    return accepted


def spans_from_boundaries(boundaries: list[dict[str, Any]], count: int) -> list[tuple[int, int]]:
    starts = [0] + sorted({item["index"] for item in boundaries if 0 < item["index"] < count})
    ends = starts[1:] + [count]
    return [(start, end) for start, end in zip(starts, ends) if end > start]


def _longest_false_gap(samples: list[dict[str, Any]], start: int, end: int, end_s: float) -> float:
    longest = 0.0
    index = start
    while index < end:
        if samples[index]["hand_present_smoothed"]:
            index += 1
            continue
        gap_start = samples[index]["time_s"]
        while index < end and not samples[index]["hand_present_smoothed"]:
            index += 1
        gap_end = samples[index]["time_s"] if index < end else end_s
        longest = max(longest, gap_end - gap_start)
    return longest


def segment_record(
    span: tuple[int, int], samples: list[dict[str, Any]], index: int, level: str,
    video_duration_s: float, args: argparse.Namespace,
) -> dict[str, Any]:
    """Create a half-open segment record so adjacent clips have no fixed sample gap."""
    start, end = span
    portion = samples[start:end]
    start_s = float(samples[start]["time_s"])
    end_s = float(samples[end]["time_s"]) if end < len(samples) else float(video_duration_s)
    raw_coverage = float(np.mean([sample["hand_present_raw"] for sample in portion]))
    smoothed_coverage = float(np.mean([sample["hand_present_smoothed"] for sample in portion]))
    bridged_coverage = float(np.mean([sample["hand_gap_bridged"] for sample in portion]))
    per_hand = {
        side: {
            "raw_coverage": round(float(np.mean([
                sample["hand_validity"][side]["observed"] for sample in portion
            ])), 3),
            "smoothed_coverage": round(float(np.mean([
                sample["hand_validity"][side]["smoothed_presence"] for sample in portion
            ])), 3),
        }
        for side in HAND_SIDES
    }
    longest_gap = _longest_false_gap(samples, start, end, end_s)
    duration = max(0.0, end_s - start_s)
    valid = (
        smoothed_coverage >= args.min_hand_coverage
        and longest_gap <= args.max_no_hand_gap_s
        and duration >= args.min_export_duration_s
        and sum(sample["hand_present_raw"] for sample in portion) >= 2
    )
    return {
        "id": f"{level}_{index:03d}",
        "level": level,
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "duration_s": round(duration, 3),
        "sample_start": start,
        "sample_end": end - 1,
        "hand_coverage": round(smoothed_coverage, 3),
        "raw_hand_coverage": round(raw_coverage, 3),
        "bridged_hand_gap_coverage": round(bridged_coverage, 3),
        "per_hand_coverage": per_hand,
        "longest_no_hand_gap_s": round(longest_gap, 3),
        "mean_sharpness": round(float(np.mean([sample["sharpness"] for sample in portion])), 2),
        "valid_operation": bool(valid),
    }


def fallback_annotation(segment: dict[str, Any], reason: str = "未配置或未成功调用视觉语言模型") -> dict[str, Any]:
    coverage = float(segment.get("hand_coverage", 0.0))
    description = (
        "片段中手部持续或间歇出现，具体动作需要复核。"
        if coverage >= 0.30 else "片段中未检测到足够稳定的手部活动。"
    )
    return {
        "scene": "未知",
        "subtask": "未知",
        "semantic_key": "未知",
        "left_hand": {"visible": False, "action": "未知", "object": "未知物体", "description": "未知"},
        "right_hand": {"visible": False, "action": "未知", "object": "未知物体", "description": "未知"},
        "action": {"verb": "未知", "description": description},
        "objects": ["未知物体"],
        "temporal_evidence": "未获得可靠的视觉语言标注。",
        "meaningful_action": bool(coverage >= 0.30),
        "contains_multiple_actions": False,
        "action_sequence": [],
        "suggested_boundary_frames": [],
        "confidence": 0.0,
        "annotation_source": "fallback",
        "failure_reason": reason,
    }


def _normalize_hand_annotation(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    return {
        "visible": bool(raw.get("visible", False)),
        "action": str(raw.get("action") or "未知").strip(),
        "object": str(raw.get("object") or "未知物体").strip(),
        "description": str(raw.get("description") or "未知").strip(),
    }


def normalize_annotation(raw: Any, segment: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return fallback_annotation(segment, "VLM 响应不是 JSON 对象")
    action = raw.get("action") if isinstance(raw.get("action"), dict) else {}
    objects = raw.get("objects", ["未知物体"])
    if isinstance(objects, str):
        objects = [objects]
    if not isinstance(objects, list):
        objects = ["未知物体"]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    left = _normalize_hand_annotation(raw.get("left_hand"))
    right = _normalize_hand_annotation(raw.get("right_hand"))
    semantic_key = str(raw.get("semantic_key") or "").strip()
    if not semantic_key:
        semantic_key = f"左:{left['action']}:{left['object']}|右:{right['action']}:{right['object']}"
    action_sequence = raw.get("action_sequence")
    if not isinstance(action_sequence, list):
        action_sequence = []
    normalized_sequence = []
    for item in action_sequence:
        if not isinstance(item, dict):
            continue
        try:
            start_frame = int(item.get("start_frame", 1))
            end_frame = int(item.get("end_frame", start_frame))
        except (TypeError, ValueError):
            continue
        normalized_sequence.append({
            "action": str(item.get("action") or "未知").strip(),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    boundary_frames = raw.get("suggested_boundary_frames")
    if not isinstance(boundary_frames, list):
        boundary_frames = []
    normalized_boundaries = []
    for item in boundary_frames:
        try:
            frame_number = int(item)
        except (TypeError, ValueError):
            continue
        if frame_number >= 2:
            normalized_boundaries.append(frame_number)
    return {
        "scene": str(raw.get("scene") or "未知").strip(),
        "subtask": str(raw.get("subtask") or "未知").strip(),
        "semantic_key": semantic_key,
        "left_hand": left,
        "right_hand": right,
        "action": {
            "verb": str(action.get("verb") or "未知").strip(),
            "description": str(action.get("description") or "未知").strip(),
        },
        "objects": [str(item).strip() for item in objects if str(item).strip()] or ["未知物体"],
        "temporal_evidence": str(raw.get("temporal_evidence") or "未提供").strip(),
        "meaningful_action": bool(raw.get("meaningful_action", True)),
        "contains_multiple_actions": bool(raw.get("contains_multiple_actions", False)),
        "action_sequence": normalized_sequence,
        "suggested_boundary_frames": sorted(set(normalized_boundaries)),
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "annotation_source": "vlm",
    }


def annotation_caption(annotation: dict[str, Any]) -> str:
    return f"{annotation['subtask']}：{annotation['action']['description']}"


def parse_json_response(content: Any) -> Any:
    if isinstance(content, list):
        content = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    if not isinstance(content, str):
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, cleaned, flags=re.DOTALL)
            if not match:
                continue
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        return None


def _resize_for_vlm(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def vlm_annotation(
    frames: list[np.ndarray], frame_times: list[float], segment: dict[str, Any],
    api_base: str | None, model: str | None, image_max_side: int,
) -> dict[str, Any]:
    key = os.getenv("VLM_API_KEY")
    if not key or not api_base or not model:
        return fallback_annotation(segment)
    caption_context = (
        "这是多个相邻同语义候选合并后的完整细粒度片段。请基于全部时间范围重新生成整体描述，"
        "描述必须覆盖整段，而不是只描述某个局部窗口；若合并后实际包含不同语义动作，"
        "仍应如实设置 contains_multiple_actions=true。"
        if segment.get("merged_recaption_request") else ""
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": (
        f"以下 {len(frames)} 帧按时间顺序来自同一第一视角候选细粒度片段，时间 "
        f"{segment['start_s']:.2f}s–{segment['end_s']:.2f}s，手部覆盖率 {segment['hand_coverage']:.0%}。\n"
        f"{caption_context}"
        "连续切割、连续清洗、连续擦拭、连续搅拌等周期运动应视为一个语义动作，不要按每次往返拆开。"
        "只输出合法 JSON，不要 Markdown、解释或额外字段：\n"
        "{\n"
        '  "scene":"场景",\n'
        '  "subtask":"简短且稳定的子任务名称",\n'
        '  "semantic_key":"规范化语义键，例如 右手|切割|蒜；相同连续动作必须用相同键",\n'
        '  "left_hand":{"visible":true,"action":"动作或无","object":"物体或未知物体","description":"简述"},\n'
        '  "right_hand":{"visible":true,"action":"动作或无","object":"物体或未知物体","description":"简述"},\n'
        '  "action":{"verb":"拿起|持有|移动|放下|打开|关闭|倒入|取出|切割|清洗|擦拭|搅拌|等待|其他|未知","description":"整段动作描述"},\n'
        '  "objects":["关键物体"],\n'
        '  "temporal_evidence":"从首尾和中间帧看到的变化",\n'
        '  "meaningful_action":true,\n'
        '  "contains_multiple_actions":false,\n'
        '  "action_sequence":[{"action":"动作名称","start_frame":1,"end_frame":8}],\n'
        '  "suggested_boundary_frames":[],\n'
        '  "confidence":0.0\n'
        "}\n"
        "若整段只有一个连续语义动作，contains_multiple_actions=false 且 suggested_boundary_frames=[]。"
        "若包含多个依次发生的动作，contains_multiple_actions=true，并用 action_sequence 描述各动作覆盖的帧号；"
        "suggested_boundary_frames 填写新动作开始的帧号（2 到总帧数之间的整数）。"
        "不要把连续切割/清洗的每次往返当成多个动作。不要猜测不可见信息；不确定时写未知并降低 confidence。"
    )}]
    for index, (frame, time_s) in enumerate(zip(frames, frame_times), 1):
        resized = _resize_for_vlm(frame, image_max_side)
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return fallback_annotation(segment, f"第 {index} 帧编码失败")
        content.extend([
            {"type": "text", "text": f"帧 {index}/{len(frames)}，时间 {time_s:.3f}s"},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
            }},
        ])
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 768,
        "messages": [
            {"role": "system", "content": "你是严谨的第一视角具身操作视频标注员，必须严格输出指定 JSON。"},
            {"role": "user", "content": content},
        ],
    }
    try:
        response = requests.post(
            api_base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120,
        )
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        raw = parse_json_response(raw_content)
        if raw is None:
            annotation = fallback_annotation(segment, "VLM 响应不是 JSON 对象")
            annotation["raw_response"] = str(raw_content)[:4000]
            return annotation
        return normalize_annotation(raw, segment)
    except (requests.RequestException, KeyError, IndexError, TypeError) as error:
        print(f"[warning] VLM 标注失败，改用回退 JSON：{error}")
        return fallback_annotation(segment, f"VLM 请求失败：{error}")


def read_frame_at(video_path: str, time_s: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def sample_segment_frames(
    info: VideoInfo, segment: dict[str, Any], frame_count: int,
) -> tuple[list[np.ndarray], list[float]]:
    safe_end = min(segment["end_s"], max(segment["start_s"], info.duration_s - 1.0 / info.fps))
    duration = max(0.0, safe_end - segment["start_s"])
    available = max(1, int(math.floor(duration * info.fps)) + 1)
    count = min(max(1, frame_count), available)
    times = np.linspace(segment["start_s"], safe_end, count).tolist()
    unique_times: list[float] = []
    seen_frames: set[int] = set()
    for time_s in times:
        frame_index = min(info.frame_count - 1, max(0, round(time_s * info.fps)))
        if frame_index not in seen_frames:
            unique_times.append(frame_index / info.fps)
            seen_frames.add(frame_index)
    frames = [read_frame_at(info.path, time_s) for time_s in unique_times]
    valid = [(frame, time_s) for frame, time_s in zip(frames, unique_times) if frame is not None]
    return [item[0] for item in valid], [item[1] for item in valid]


def annotate_candidate_segment(
    segment: dict[str, Any], info: VideoInfo, args: argparse.Namespace,
) -> None:
    if segment["hand_coverage"] < args.min_hand_coverage:
        annotation = fallback_annotation(segment, "手部长时间不可见，跳过 VLM 标注")
        annotation["annotation_source"] = "skipped_no_hand"
        frame_times: list[float] = []
    else:
        frames, frame_times = sample_segment_frames(info, segment, args.fine_frame_count)
        annotation = (
            vlm_annotation(
                frames, frame_times, segment, args.vlm_api_base, args.vlm_model,
                args.vlm_image_max_side,
            ) if frames else fallback_annotation(segment, "无法读取候选片段帧")
        )
    segment["vlm_frame_times_s"] = [round(time_s, 3) for time_s in frame_times]
    segment["semantic_annotation"] = annotation
    segment["caption_zh"] = annotation_caption(annotation)


def _weak_velocity_minima(
    samples: list[dict[str, Any]], segment: dict[str, Any], args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Rank sub-threshold local minima for aligning a VLM-requested refinement."""
    times = np.array([sample["time_s"] for sample in samples], dtype=np.float64)
    start = segment["sample_start"]
    end = segment["sample_end"] + 1
    result: list[dict[str, Any]] = []
    for side in HAND_SIDES:
        speeds = np.array([
            sample["hand_motion"][side]["smoothed_speed"]
            if sample["hand_motion"][side]["smoothed_speed"] is not None else np.nan
            for sample in samples
        ], dtype=np.float64)
        finite = speeds[start:end][np.isfinite(speeds[start:end])]
        if len(finite) < 4:
            continue
        q10, q90 = np.percentile(finite, [10, 90])
        scale = max(float(q90 - q10), 1e-6)
        for index in range(start + 1, end - 1):
            time_s = times[index]
            if (
                time_s - segment["start_s"] < args.minimum_provisional_duration_s
                or segment["end_s"] - time_s < args.minimum_provisional_duration_s
                or not np.isfinite(speeds[index])
            ):
                continue
            center = np.where((np.abs(times - time_s) <= args.velocity_center_s) & (np.arange(len(samples)) >= start) & (np.arange(len(samples)) < end))[0]
            left = np.where((times >= time_s - args.velocity_context_s) & (times < time_s) & (np.arange(len(samples)) >= start))[0]
            right = np.where((times > time_s) & (times <= time_s + args.velocity_context_s) & (np.arange(len(samples)) < end))[0]
            center_values = speeds[center][np.isfinite(speeds[center])]
            left_values = speeds[left][np.isfinite(speeds[left])]
            right_values = speeds[right][np.isfinite(speeds[right])]
            if not len(center_values) or not len(left_values) or not len(right_values):
                continue
            valley = float(np.min(center_values))
            if speeds[index] > valley + 1e-10:
                continue
            prominence = max(0.0, (min(float(np.median(left_values)), float(np.median(right_values))) - valley) / scale)
            motion_weight = float(samples[index]["hand_motion"][side].get("weight", 0.0))
            result.append({
                "index": index,
                "time_s": float(time_s),
                "hand": side,
                "score": prominence + 0.1 * motion_weight,
                "prominence": prominence,
            })
    return result


def _refinement_indices(
    samples: list[dict[str, Any]], segment: dict[str, Any], args: argparse.Namespace,
) -> list[dict[str, Any]]:
    annotation = segment["semantic_annotation"]
    frame_times = segment.get("vlm_frame_times_s", [])
    suggested = annotation.get("suggested_boundary_frames", [])
    targets = [
        frame_times[frame_number - 1]
        for frame_number in suggested
        if 2 <= frame_number <= len(frame_times)
    ]
    if not targets:
        targets = [(segment["start_s"] + segment["end_s"]) / 2.0]
    weak = _weak_velocity_minima(samples, segment, args)
    selected: list[dict[str, Any]] = []
    for target in targets[:args.max_vlm_refinement_splits]:
        nearby = [
            item for item in weak
            if abs(item["time_s"] - target) <= args.vlm_refinement_search_s
        ]
        if nearby:
            choice = max(
                nearby,
                key=lambda item: item["score"]
                - 0.25 * abs(item["time_s"] - target) / max(args.vlm_refinement_search_s, 1e-6),
            )
            candidate = {
                **choice,
                "target_time_s": round(float(target), 3),
                "aligned_to_weak_velocity_minimum": True,
            }
        else:
            allowable = [
                index for index in range(segment["sample_start"] + 1, segment["sample_end"] + 1)
                if samples[index]["time_s"] - segment["start_s"] >= args.minimum_provisional_duration_s
                and segment["end_s"] - samples[index]["time_s"] >= args.minimum_provisional_duration_s
            ]
            if not allowable:
                continue
            index = min(allowable, key=lambda item: abs(samples[item]["time_s"] - target))
            candidate = {
                "index": index,
                "time_s": float(samples[index]["time_s"]),
                "hand": "vlm_only",
                "score": 0.0,
                "prominence": 0.0,
                "target_time_s": round(float(target), 3),
                "aligned_to_weak_velocity_minimum": False,
            }
        if any(abs(candidate["time_s"] - item["time_s"]) < args.minimum_provisional_duration_s for item in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: item["index"])


def refine_multi_action_segments(
    provisional: list[dict[str, Any]], samples: list[dict[str, Any]],
    info: VideoInfo, args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Split each VLM-flagged multi-action candidate once, then caption its children."""
    refined: list[dict[str, Any]] = []
    added_boundaries: list[dict[str, Any]] = []
    refined_parent_count = 0
    multi_count = sum(
        segment["semantic_annotation"].get("annotation_source") == "vlm"
        and segment["semantic_annotation"].get("contains_multiple_actions")
        for segment in provisional
    )
    if multi_count:
        print(f"[info] VLM 标记 {multi_count} 个多动作候选，正在自动补切并重新描述子片段…")
    for segment in provisional:
        annotation = segment["semantic_annotation"]
        if annotation.get("annotation_source") != "vlm" or not annotation.get("contains_multiple_actions"):
            refined.append(segment)
            continue
        choices = _refinement_indices(samples, segment, args)
        indices = [choice["index"] for choice in choices]
        starts = [segment["sample_start"]] + indices
        ends = indices + [segment["sample_end"] + 1]
        if not indices or any(end <= start for start, end in zip(starts, ends)):
            refined.append(segment)
            continue
        refined_parent_count += 1
        for choice in choices:
            added_boundaries.append({
                "index": choice["index"],
                "time_s": round(float(choice["time_s"]), 6),
                "hands": [] if choice["hand"] == "vlm_only" else [choice["hand"]],
                "source": "vlm_multi_action_refinement",
                "score": round(float(annotation.get("confidence", 0.0)), 4),
                "hard_boundary": False,
                "evidence": [{
                    "parent_segment_id": segment["id"],
                    "vlm_target_time_s": choice["target_time_s"],
                    "aligned_to_weak_velocity_minimum": choice["aligned_to_weak_velocity_minimum"],
                    "weak_velocity_prominence": round(float(choice["prominence"]), 4),
                }],
            })
        for start, end in zip(starts, ends):
            child = segment_record((start, end), samples, len(refined) + 1, "candidate_fine", info.duration_s, args)
            child["refined_from"] = segment["id"]
            annotate_candidate_segment(child, info, args)
            refined.append(child)
    for index, segment in enumerate(refined, 1):
        segment["id"] = f"candidate_fine_{index:03d}"
    return refined, added_boundaries, refined_parent_count


def _norm_text(value: str) -> str:
    return re.sub(r"[\s，。；、,:：;（）()]+", "", value.lower())


def _canonical_verb(value: str) -> str:
    text = _norm_text(value)
    aliases = {
        "切": "切割", "切菜": "切割", "清洁": "清洗", "洗": "清洗",
        "拿": "拿起", "取": "取出", "放": "放下", "抓取": "拿起",
    }
    return aliases.get(text, text)


def compatible_fine_annotations(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("annotation_source") != "vlm" or right.get("annotation_source") != "vlm":
        return False
    if not left.get("meaningful_action", True) or not right.get("meaningful_action", True):
        return False
    if "未知" not in left["semantic_key"] and _norm_text(left["semantic_key"]) == _norm_text(right["semantic_key"]):
        return True
    if _canonical_verb(left["action"]["verb"]) != _canonical_verb(right["action"]["verb"]):
        return False
    left_objects = {_norm_text(item) for item in left["objects"] if "未知" not in item}
    right_objects = {_norm_text(item) for item in right["objects"] if "未知" not in item}
    if left_objects and right_objects:
        return bool(left_objects & right_objects)
    return _norm_text(left["subtask"]) == _norm_text(right["subtask"]) and "未知" not in left["subtask"]


def merge_fine_segments(
    provisional: list[dict[str, Any]], boundary_by_index: dict[int, dict[str, Any]],
    samples: list[dict[str, Any]], info: VideoInfo, args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    removed: list[dict[str, Any]] = []
    for segment in provisional:
        if not groups:
            groups.append([segment])
            continue
        boundary = boundary_by_index.get(segment["sample_start"])
        can_merge = (
            not (boundary and boundary.get("hard_boundary"))
            and compatible_fine_annotations(groups[-1][-1]["semantic_annotation"], segment["semantic_annotation"])
        )
        if can_merge:
            groups[-1].append(segment)
            if boundary:
                removed.append({**boundary, "decision": "removed_by_same_semantics"})
        else:
            groups.append([segment])
    final: list[dict[str, Any]] = []
    for index, group in enumerate(groups, 1):
        span = (group[0]["sample_start"], group[-1]["sample_end"] + 1)
        record = segment_record(span, samples, index, "fine", info.duration_s, args)
        best = max(group, key=lambda item: item["semantic_annotation"].get("confidence", 0.0))
        record["semantic_annotation"] = best["semantic_annotation"]
        record["caption_zh"] = annotation_caption(record["semantic_annotation"])
        record["merged_from"] = [item["id"] for item in group]
        record["refined_from"] = sorted({
            item["refined_from"] for item in group if item.get("refined_from")
        })
        record["segmentation_method"] = "velocity_minima_then_vlm_semantic_merge"
        final.append(record)
    return final, removed


def recaption_merged_fine_segments(
    fine: list[dict[str, Any]], info: VideoInfo, args: argparse.Namespace,
) -> tuple[int, int]:
    """Re-caption long merged clips over their full duration; keep old labels on failure."""
    targets = [
        segment for segment in fine
        if segment["valid_operation"]
        and len(segment.get("merged_from", [])) > 1
        and segment["duration_s"] >= args.merged_recaption_min_duration_s
    ]
    if not targets or not os.getenv("VLM_API_KEY") or not args.vlm_api_base or not args.vlm_model:
        return 0, 0
    print(f"[info] 对 {len(targets)} 个合并后长片段重新进行整体 VLM 描述…")
    success = 0
    for segment in tqdm(targets, desc="合并长片段重描述", unit="clip"):
        previous = segment["semantic_annotation"]
        frames, frame_times = sample_segment_frames(info, segment, args.fine_frame_count)
        if not frames:
            segment["merged_recaption"] = {
                "status": "failed", "reason": "无法读取合并后片段帧", "frame_times_s": [],
            }
            continue
        request_segment = {**segment, "merged_recaption_request": True}
        annotation = vlm_annotation(
            frames, frame_times, request_segment, args.vlm_api_base, args.vlm_model,
            args.vlm_image_max_side,
        )
        if annotation.get("annotation_source") != "vlm":
            segment["merged_recaption"] = {
                "status": "failed",
                "reason": annotation.get("failure_reason", "VLM 重描述失败"),
                "frame_times_s": [round(time_s, 3) for time_s in frame_times],
            }
            continue
        annotation["annotation_stage"] = "merged_full_clip_recaption"
        segment["pre_recaption_annotation"] = previous
        segment["semantic_annotation"] = annotation
        segment["caption_zh"] = annotation_caption(annotation)
        segment["merged_recaption"] = {
            "status": "success",
            "frame_times_s": [round(time_s, 3) for time_s in frame_times],
        }
        success += 1
    return len(targets), success


def review_reasons(annotation: dict[str, Any], threshold: float) -> list[str]:
    reasons: list[str] = []
    if annotation.get("confidence", 0.0) < threshold:
        reasons.append(f"VLM 置信度 {annotation.get('confidence', 0.0):.2f} 低于阈值 {threshold:.2f}")
    if annotation.get("annotation_source") != "vlm":
        reasons.append(annotation.get("failure_reason", "使用回退标注"))
    if "未知" in annotation.get("action", {}).get("verb", "未知"):
        reasons.append("动作未知")
    if annotation.get("contains_multiple_actions"):
        reasons.append("VLM 判断候选片段可能包含多个动作，可能漏切")
    return reasons


def clip_filename(segment: dict[str, Any]) -> str:
    return f"{segment['id']}_{segment['start_s']:.1f}-{segment['end_s']:.1f}s.mp4"


def build_clean_annotations(
    info: VideoInfo, fine: list[dict[str, Any]], clips_exported: bool,
) -> dict[str, Any]:
    """Build the compact, training-facing annotation file from reviewed fine clips."""
    clips: list[dict[str, Any]] = []
    for segment in fine:
        annotation = segment["semantic_annotation"]
        if not (
            segment["valid_operation"]
            and annotation.get("annotation_source") == "vlm"
            and annotation.get("meaningful_action", False)
            and not segment.get("needs_review", False)
        ):
            continue
        hands = {
            output_side: {
                "visible": bool(annotation[annotation_side].get("visible", False)),
                "action": str(annotation[annotation_side].get("action") or "未知"),
                "object": str(annotation[annotation_side].get("object") or "未知物体"),
            }
            for output_side, annotation_side in (("left", "left_hand"), ("right", "right_hand"))
        }
        clips.append({
            "id": segment["id"],
            "clip_path": f"valid_segments/{clip_filename(segment)}" if clips_exported else None,
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
            "duration_s": segment["duration_s"],
            "sample_range": {
                "start": segment["sample_start"],
                "end": segment["sample_end"],
            },
            "caption": segment["caption_zh"],
            "subtask": annotation["subtask"],
            "action": {
                "verb": annotation["action"]["verb"],
                "description": annotation["action"]["description"],
            },
            "objects": annotation["objects"],
            "hands": hands,
            "quality": {
                "hand_coverage": segment["hand_coverage"],
                "vlm_confidence": annotation["confidence"],
            },
        })
    return {
        "schema_version": "1.0",
        "video": {
            "id": Path(info.path).stem,
            "file": Path(info.path).name,
            "fps": info.fps,
            "duration_s": info.duration_s,
        },
        "hand_data_file": "hand_landmarks.json",
        "clips": clips,
    }


def write_trajectories(samples: list[dict[str, Any]], output: Path) -> None:
    fields = [
        "time_s", "frame_index", "side", "valid_mask", "x", "y", "z",
        "world_x", "world_y", "world_z", "raw_speed", "smoothed_speed",
        "motion_source", "motion_weight", "boundary_x", "boundary_y",
        "interpolated_for_motion",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            for side in HAND_SIDES:
                hand = _hand_for_side(sample, side)
                wrist = hand["landmarks_2d_relative"][0] if hand else {}
                world = ((hand.get("landmarks_3d_relative") or [{}])[0] if hand else {})
                motion = sample["hand_motion"][side]
                track = sample["hand_tracks"][side]
                boundary_center = track.get("boundary_palm_center") or {}
                writer.writerow({
                    "time_s": sample["time_s"], "frame_index": sample["frame_index"],
                    "side": side, "valid_mask": bool(hand),
                    "x": wrist.get("x"), "y": wrist.get("y"), "z": wrist.get("z"),
                    "world_x": world.get("x"), "world_y": world.get("y"), "world_z": world.get("z"),
                    "raw_speed": motion["raw_speed"], "smoothed_speed": motion["smoothed_speed"],
                    "motion_source": motion.get("source"), "motion_weight": motion.get("weight", 0.0),
                    "boundary_x": boundary_center.get("x"), "boundary_y": boundary_center.get("y"),
                    "interpolated_for_motion": track.get("interpolated_for_motion", False),
                })


def draw_hand(frame: np.ndarray, hand: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    pixel = [(int(p["x"] * width), int(p["y"] * height)) for p in hand["landmarks_2d_relative"]]
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
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(f"无法创建骨架叠加视频：{output}")
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for hand in lookup.get(frame_index, []):
            draw_hand(frame, hand)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()


def write_valid_clips(info: VideoInfo, segments: list[dict[str, Any]], output_dir: Path) -> None:
    """Export only final fine-grained clips that pass the hand-visibility filter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise RuntimeError(f"导出片段时无法重新打开输入视频：{info.path}")
    try:
        for segment in segments:
            if not segment["valid_operation"]:
                continue
            start_frame = max(0, int(math.floor(segment["start_s"] * info.fps)))
            end_frame = min(info.frame_count, int(math.ceil(segment["end_s"] * info.fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            target = output_dir / clip_filename(segment)
            writer = cv2.VideoWriter(
                str(target), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(f"无法创建有效片段视频：{target}")
            try:
                for _ in range(max(0, end_frame - start_frame)):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(frame)
            finally:
                writer.release()
    finally:
        cap.release()


def run(args: argparse.Namespace) -> None:
    video = Path(args.video).expanduser()
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    info = probe_video(video)
    print(f"[info] {info.duration_s:.1f}s, {info.width}x{info.height}, {info.fps:.2f} FPS")

    samples = sample_and_track(info, args.sample_fps, args.hand_confidence)
    annotate_hand_validity(samples, args.hand_gap_tolerance)
    attach_hand_motion(
        samples, args.min_camera_motion_quality, args.velocity_smoothing_radius_s,
        args.motion_interpolation_gap_s, args.motion_interpolation_max_displacement,
    )
    actual_sample_fps = (
        (len(samples) - 1) / max(samples[-1]["time_s"] - samples[0]["time_s"], 1e-6)
        if len(samples) > 1 else args.sample_fps
    )

    velocity_candidates = find_velocity_candidates(samples, args)
    technical_candidates = _long_no_hand_boundaries(
        samples, args.max_no_hand_gap_s, info.duration_s,
    )
    candidates = consolidate_boundaries(
        velocity_candidates + technical_candidates,
        min(args.velocity_min_gap_s, args.max_no_hand_gap_s / 2),
    )
    candidates = _add_analysis_windows(samples, candidates, args.max_provisional_segment_s)
    candidates = consolidate_boundaries(candidates, args.minimum_provisional_duration_s)
    boundary_by_index = {item["index"]: item for item in candidates}
    provisional_spans = spans_from_boundaries(candidates, len(samples))
    initial_provisional = [
        segment_record(span, samples, index + 1, "candidate_fine", info.duration_s, args)
        for index, span in enumerate(provisional_spans)
    ]
    print(f"[info] 腕速候选边界 {len(velocity_candidates)} 个；待描述候选细片段 {len(initial_provisional)} 个…")

    for segment in tqdm(initial_provisional, desc="VLM细粒度标注", unit="clip"):
        annotate_candidate_segment(segment, info, args)

    provisional, refinement_boundaries, refined_parent_count = refine_multi_action_segments(
        initial_provisional, samples, info, args,
    )
    if refinement_boundaries:
        for boundary in refinement_boundaries:
            boundary_by_index[boundary["index"]] = boundary
        candidates = sorted(boundary_by_index.values(), key=lambda item: item["time_s"])
        print(
            f"[info] VLM 多动作补切 {refined_parent_count} 个父片段，"
            f"新增 {len(refinement_boundaries)} 个候选边界；重新描述后共 {len(provisional)} 个候选细片段。"
        )

    fine, removed_boundaries = merge_fine_segments(
        provisional, boundary_by_index, samples, info, args,
    )
    merged_recaption_count, merged_recaption_success_count = recaption_merged_fine_segments(
        fine, info, args,
    )
    review_queue: list[dict[str, Any]] = []
    for segment in fine:
        if not segment["valid_operation"]:
            segment["needs_review"] = False
            segment["review_reasons"] = []
            continue
        reasons = review_reasons(segment["semantic_annotation"], args.review_confidence)
        segment["needs_review"] = bool(reasons)
        segment["review_reasons"] = reasons
        if reasons:
            review_queue.append({
                "level": "fine", "segment_id": segment["id"],
                "start_s": segment["start_s"], "end_s": segment["end_s"], "reasons": reasons,
            })
    valid_ids = [segment["id"] for segment in fine if segment["valid_operation"]]
    refinement_children = [segment for segment in provisional if "refined_from" in segment]
    annotation_attempts = initial_provisional + refinement_children
    vlm_eligible = [segment for segment in annotation_attempts if segment["hand_coverage"] >= args.min_hand_coverage]
    vlm_segments = [
        segment for segment in annotation_attempts
        if segment["semantic_annotation"]["annotation_source"] == "vlm"
    ]
    diagnostics = {
        "schema_version": "0.6",
        "video": asdict(info),
        "parameters": vars(args),
        "segmentation": {
            "order": "fine_only",
            "fine_method": "weighted_interpolated_per_hand_and_global_speed_minima_then_vlm_refine_and_merge",
            "actual_sample_fps": round(actual_sample_fps, 4),
            "velocity_candidate_count": len(velocity_candidates),
            "initial_provisional_fine_count": len(initial_provisional),
            "provisional_fine_count": len(provisional),
            "vlm_multi_action_refined_parent_count": refined_parent_count,
            "vlm_refinement_boundary_count": len(refinement_boundaries),
            "final_fine_count": len(fine),
            "removed_semantic_boundary_count": len(removed_boundaries),
            "vlm_eligible_segment_count": len(vlm_eligible),
            "vlm_successful_segment_count": len(vlm_segments),
            "vlm_fine_response_success_ratio": round(len(vlm_segments) / max(1, len(vlm_eligible)), 3),
            "merged_recaption_segment_count": merged_recaption_count,
            "merged_recaption_success_count": merged_recaption_success_count,
            "boundary_candidates": candidates,
            "removed_boundaries": removed_boundaries,
        },
        "coordinate_system": {
            "image": "MediaPipe x/y normalized to [0,1]; z is relative depth",
            "motion": "2D palm speed after background affine camera-motion compensation, normalized by analysis-frame diagonal per second",
            "world": "MediaPipe relative hand coordinates; not calibrated metric world space",
        },
        "fine_segments": fine,
        "valid_segments": valid_ids,
        "review_queue": review_queue,
    }
    clean_annotations = build_clean_annotations(info, fine, clips_exported=not args.skip_video_outputs)
    (output / "annotations.json").write_text(
        json.dumps(clean_annotations, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "annotations_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "hand_landmarks.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trajectories(samples, output / "wrist_trajectories.csv")
    if not args.skip_video_outputs:
        print("[info] 写入骨架叠加视频…")
        write_overlay(info, samples, output / "hand_overlay.mp4")
        print("[info] 按最终细粒度边界导出有效操作片段…")
        write_valid_clips(info, fine, output / "valid_segments")
    print(
        f"[done] 最终细片段：{len(fine)}；干净标注：{len(clean_annotations['clips'])}；"
        f"有效视频：{len(valid_ids)}；待复核：{len(review_queue)}；结果目录：{output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第一视角视频：细粒度切分、语言标注、21点手姿态和轨迹")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--sample-fps", type=float, default=8.0, help="手部与运动分析采样帧率，默认 8")
    parser.add_argument("--hand-confidence", type=float, default=0.55, help="MediaPipe 手部检测/跟踪阈值")
    parser.add_argument("--hand-gap-tolerance", type=float, default=0.5, help="仅桥接手部存在状态的短缺失时长，默认 0.5 秒")
    parser.add_argument("--min-camera-motion-quality", type=float, default=0.15, help="背景相机运动补偿最低质量")
    parser.add_argument("--motion-interpolation-gap-s", type=float, default=0.5, help="仅供边界计算的掌心短缺失补全上限，默认 0.5 秒")
    parser.add_argument("--motion-interpolation-max-displacement", type=float, default=0.25, help="允许补全的最大相机补偿位移，按画面对角线归一化")
    parser.add_argument("--velocity-smoothing-radius-s", type=float, default=0.25, help="速度高斯平滑半径，默认 0.25 秒")
    parser.add_argument("--velocity-context-s", type=float, default=0.75, help="速度谷前后参考窗口，默认 0.75 秒")
    parser.add_argument("--velocity-center-s", type=float, default=0.20, help="速度谷中心排除半径，默认 0.20 秒")
    parser.add_argument("--velocity-drop-ratio", type=float, default=0.25, help="速度谷相对下降阈值，默认 0.25")
    parser.add_argument("--velocity-prominence", type=float, default=0.10, help="速度谷归一化显著性阈值，默认 0.10")
    parser.add_argument("--velocity-min-gap-s", type=float, default=0.60, help="同一速度序列候选最短间隔，默认 0.6 秒")
    parser.add_argument("--velocity-min-window-weight", type=float, default=0.60, help="速度谷前后窗口最低加权覆盖率，默认 0.60")
    parser.add_argument("--hand-boundary-fusion-s", type=float, default=0.40, help="左右手候选融合容差，默认 0.4 秒")
    parser.add_argument("--minimum-provisional-duration-s", type=float, default=0.60, help="候选片段最短时长，默认 0.6 秒")
    parser.add_argument("--max-provisional-segment-s", type=float, default=8.0, help="仅为VLM分析插入的最长候选窗口，默认 8 秒")
    parser.add_argument("--max-vlm-refinement-splits", type=int, default=2, help="每个多动作候选最多自动补切边界数，默认 2")
    parser.add_argument("--vlm-refinement-search-s", type=float, default=1.0, help="VLM建议时间附近搜索弱腕速谷的半径，默认 1 秒")
    parser.add_argument("--min-hand-coverage", type=float, default=0.30, help="有效片段最低平滑手部覆盖率，默认 0.30")
    parser.add_argument("--max-no-hand-gap-s", type=float, default=1.0, help="有效视频允许的最长双手不可见间隔，默认 1 秒")
    parser.add_argument("--min-export-duration-s", type=float, default=0.5, help="导出有效视频的最短时长，默认 0.5 秒")
    parser.add_argument("--fine-frame-count", type=int, default=16, help="每个候选细片段最多发送给VLM的均匀帧数，默认 16")
    parser.add_argument("--merged-recaption-min-duration-s", type=float, default=8.0, help="合并后触发整体重描述的最短片段时长，默认 8 秒")
    parser.add_argument("--vlm-image-max-side", type=int, default=768, help="发送给VLM前的图像最长边，默认 768")
    parser.add_argument("--vlm-api-base", default=None, help="兼容 Chat Completions 的 API 基地址")
    parser.add_argument("--vlm-model", default=None, help="视觉语言模型名称；密钥从 VLM_API_KEY 读取")
    parser.add_argument("--review-confidence", type=float, default=0.65, help="低于该VLM置信度的有效片段进入复核队列")
    parser.add_argument("--skip-video-outputs", action="store_true", help="仅输出 JSON/CSV，不生成 MP4")
    return parser


def main() -> None:
    run(build_parser().parse_args())
