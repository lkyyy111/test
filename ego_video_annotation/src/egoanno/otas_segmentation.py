"""Training-free OTAS-inspired boundary baseline for raw egocentric video.

The original OTAS system (Li et al., WACV 2024) learns three future-frame
prediction streams: global visual, local human-object interaction, and object
relation features.  It then confirms/corrects global boundary candidates with
the two object-centric streams.  Reproducing those learned features requires
the Breakfast training set, Detectron2 masks, a relation graph, and checkpoints
that are not part of this standalone pipeline.

This module preserves OTAS's three-stream boundary-fusion idea while replacing
the learned features with deterministic raw-video proxies:

* global: whole-frame HSV and low-frequency DCT appearance;
* interaction: the same descriptor in an expanded region around visible hands;
* relation: fixed-size two-hand geometry and pose descriptors from MediaPipe.

No VLM is used to change the resulting boundaries.  An aligned ``.npz`` file
with ``global``, ``interaction``, and ``relation`` arrays can be supplied to
replace these proxies with externally extracted OTAS features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


PAPER_TITLE = (
    "OTAS: Unsupervised Boundary Detection for Object-Centric Temporal "
    "Action Segmentation (WACV 2024)"
)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _appearance_descriptor(frame: np.ndarray) -> np.ndarray:
    """Compact, deterministic appearance descriptor for one image/crop."""
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [16, 8], [0, 180, 0, 256],
    ).reshape(-1)
    histogram = _l2_normalize(histogram)
    dct_input = cv2.resize(
        gray, (32, 32), interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0
    dct = cv2.dct(dct_input)[:8, :8].reshape(-1)
    dct[0] = 0.0
    return _l2_normalize(np.concatenate([histogram, _l2_normalize(dct)]))


def _observed_hands(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        hand for hand in sample.get("hands", [])
        if hand.get("landmarks_2d_relative")
    ]


def _interaction_crop(frame: np.ndarray, sample: dict[str, Any]) -> np.ndarray | None:
    """Crop an expanded hand neighborhood that is likely to contain objects."""
    hands = _observed_hands(sample)
    if not hands:
        return None
    height, width = frame.shape[:2]
    xs: list[float] = []
    ys: list[float] = []
    for hand in hands:
        xs.extend(float(point["x"]) * width for point in hand["landmarks_2d_relative"])
        ys.extend(float(point["y"]) * height for point in hand["landmarks_2d_relative"])
    hand_width = max(xs) - min(xs)
    hand_height = max(ys) - min(ys)
    # The generous expansion is intentional: OTAS uses human-object masks,
    # while MediaPipe only gives hands. Nearby manipulated objects must remain.
    margin_x = max(0.10 * width, 1.25 * hand_width)
    margin_y = max(0.10 * height, 1.25 * hand_height)
    x0 = max(0, int(np.floor(min(xs) - margin_x)))
    x1 = min(width, int(np.ceil(max(xs) + margin_x)))
    y0 = max(0, int(np.floor(min(ys) - margin_y)))
    y1 = min(height, int(np.ceil(max(ys) + margin_y)))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def _hand_pose_descriptor(hand: dict[str, Any] | None) -> np.ndarray:
    """Return a side-stable absolute-position + normalized-pose descriptor."""
    # 1 observed + 2 center + 2 box size + 42 normalized xy = 47 values.
    if hand is None or not hand.get("landmarks_2d_relative"):
        return np.zeros(47, dtype=np.float32)
    points = np.asarray([
        [float(point["x"]), float(point["y"])]
        for point in hand["landmarks_2d_relative"]
    ], dtype=np.float32)
    center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
    size = np.ptp(points, axis=0)
    scale = max(float(np.linalg.norm(size)), 1e-4)
    normalized_pose = ((points - center) / scale).reshape(-1)
    return np.concatenate([
        np.asarray([1.0, center[0], center[1], size[0], size[1]], dtype=np.float32),
        normalized_pose,
    ])


def _relation_descriptor(sample: dict[str, Any]) -> np.ndarray:
    hands = _observed_hands(sample)
    by_side = {str(hand.get("side")): hand for hand in hands}
    left = _hand_pose_descriptor(by_side.get("left"))
    right = _hand_pose_descriptor(by_side.get("right"))
    # Extra pairwise geometry makes bimanual convergence/separation explicit.
    left_center = left[1:3]
    right_center = right[1:3]
    both = bool(left[0] and right[0])
    pair = np.concatenate([
        np.asarray([float(both)], dtype=np.float32),
        (right_center - left_center) if both else np.zeros(2, dtype=np.float32),
        np.asarray([
            float(np.linalg.norm(right_center - left_center)) if both else 0.0,
        ], dtype=np.float32),
    ])
    return np.concatenate([left, right, pair]).astype(np.float32)


def _normalize_feature_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def _validate_feature_stream(
    value: np.ndarray, name: str, sample_count: int,
) -> np.ndarray:
    features = np.asarray(value, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != sample_count:
        raise ValueError(
            f"OTAS 特征 {name!r} 必须是与采样帧对齐的 N×D 矩阵；"
            f"需要 {sample_count} 行，实际形状为 {features.shape}"
        )
    if features.shape[1] < 1 or not np.all(np.isfinite(features)):
        raise ValueError(f"OTAS 特征 {name!r} 为空或包含 NaN/无穷值")
    return _normalize_feature_rows(features)


def extract_otas_features(
    info: Any, samples: list[dict[str, Any]], feature_file: str | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    """Return aligned global, interaction, and relation feature streams."""
    if feature_file:
        path = Path(feature_file).expanduser()
        with np.load(path) as payload:
            missing = [
                name for name in ("global", "interaction", "relation")
                if name not in payload
            ]
            if missing:
                raise ValueError(
                    "--otas-feature-file 缺少数组：" + ", ".join(missing)
                )
            streams = {
                name: _validate_feature_stream(payload[name], name, len(samples))
                for name in ("global", "interaction", "relation")
            }
        return streams, f"precomputed:{path}"

    target_frames = [int(sample["frame_index"]) for sample in samples]
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"OTAS 无法打开视频：{info.path}")
    global_features: list[np.ndarray] = []
    interaction_features: list[np.ndarray] = []
    relation_features: list[np.ndarray] = []
    target_index = 0
    frame_index = 0
    try:
        with tqdm(total=len(samples), desc="OTAS三路特征", unit="frame") as progress:
            while target_index < len(target_frames):
                ok = cap.grab()
                if not ok:
                    break
                if frame_index == target_frames[target_index]:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    sample = samples[target_index]
                    global_features.append(_appearance_descriptor(frame))
                    crop = _interaction_crop(frame, sample)
                    if crop is None:
                        local = np.zeros(192, dtype=np.float32)
                    else:
                        local = _appearance_descriptor(crop)
                    presence = np.asarray([
                        float(any(hand.get("side") == "left" for hand in _observed_hands(sample))),
                        float(any(hand.get("side") == "right" for hand in _observed_hands(sample))),
                    ], dtype=np.float32)
                    interaction_features.append(_l2_normalize(np.concatenate([local, presence])))
                    relation_features.append(_relation_descriptor(sample))
                    target_index += 1
                    progress.update(1)
                frame_index += 1
    finally:
        cap.release()
    if len(global_features) != len(samples):
        raise RuntimeError(
            f"OTAS 特征只读取到 {len(global_features)}/{len(samples)} 个采样帧"
        )
    return {
        "global": _normalize_feature_rows(np.stack(global_features)),
        "interaction": _normalize_feature_rows(np.stack(interaction_features)),
        "relation": _normalize_feature_rows(np.stack(relation_features)),
    }, "opencv_global_hand_neighborhood_mediapipe_relation_proxies"


def _adjacent_cosine_change(features: np.ndarray) -> np.ndarray:
    normalized = _normalize_feature_rows(features)
    cosine = np.clip(np.sum(normalized[:-1] * normalized[1:], axis=1), -1.0, 1.0)
    return np.concatenate([[0.0], 1.0 - cosine]).astype(np.float64)


def _smooth_curve(values: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0 or len(values) < 3:
        return values.astype(np.float64, copy=True)
    kernel_radius = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(radius / 2.0, 0.75)
    kernel = np.exp(-0.5 * (kernel_radius / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _mean_normalize_curve(values: np.ndarray) -> np.ndarray:
    positive = values[1:][np.isfinite(values[1:])]
    scale = float(np.mean(positive)) if len(positive) else 0.0
    return values / max(scale, 1e-8)


def _local_peaks(values: np.ndarray, order: int, threshold: float) -> list[int]:
    order = max(1, int(order))
    peaks: list[int] = []
    for index in range(1, len(values)):
        start = max(1, index - order)
        end = min(len(values), index + order + 1)
        window = values[start:end]
        if values[index] < threshold or values[index] < float(np.max(window)):
            continue
        # Resolve a flat maximum deterministically to its earliest sample.
        equal = np.flatnonzero(np.isclose(window, values[index], rtol=1e-7, atol=1e-9))
        if len(equal) and start + int(equal[0]) != index:
            continue
        peaks.append(index)
    return peaks


def _suppress_close_boundaries(
    records: list[dict[str, Any]], times: np.ndarray, min_gap_s: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (-float(item["score"]), int(item["index"]))):
        time_s = float(times[int(record["index"])])
        if any(abs(time_s - float(times[int(item["index"])])) < min_gap_s for item in selected):
            continue
        selected.append(record)
    return sorted(selected, key=lambda item: int(item["index"]))


def otas_boundary_selection(
    streams: dict[str, np.ndarray], times: np.ndarray,
    global_weight: float = 1.0, interaction_weight: float = 1.0,
    relation_weight: float = 1.0, peak_window_s: float = 0.75,
    neighbor_s: float = 0.75, min_gap_s: float = 0.75,
    candidate_threshold: float = 1.0, strong_local_threshold: float = 2.5,
    smoothing_s: float = 0.25,
) -> dict[str, Any]:
    """Fuse three temporal-difference streams using the OTAS decision pattern."""
    times = np.asarray(times, dtype=np.float64)
    sample_count = len(times)
    if sample_count < 3:
        raise ValueError("OTAS 至少需要 3 个采样帧")
    if np.any(np.diff(times) <= 0):
        raise ValueError("OTAS 采样时间必须严格递增")
    weights = np.asarray(
        [global_weight, interaction_weight, relation_weight], dtype=np.float64,
    )
    if np.any(weights < 0) or float(np.sum(weights)) <= 0:
        raise ValueError("OTAS 三路权重必须非负且至少一个大于 0")
    if min(peak_window_s, neighbor_s, min_gap_s) <= 0 or smoothing_s < 0:
        raise ValueError("OTAS 时间窗口参数必须为正，平滑时间不能为负")
    if candidate_threshold <= 0 or strong_local_threshold <= 0:
        raise ValueError("OTAS 候选阈值必须大于 0")

    median_step = float(np.median(np.diff(times)))
    smooth_radius = int(round(smoothing_s / median_step))
    peak_order = max(1, int(round(peak_window_s / median_step)))
    neighbor_radius = max(1, int(round(neighbor_s / median_step)))
    curves: dict[str, np.ndarray] = {}
    for name in ("global", "interaction", "relation"):
        features = _validate_feature_stream(streams[name], name, sample_count)
        curves[name] = _mean_normalize_curve(
            _smooth_curve(_adjacent_cosine_change(features), smooth_radius),
        )
    fused = (
        global_weight * curves["global"]
        + interaction_weight * curves["interaction"]
        + relation_weight * curves["relation"]
    ) / float(np.sum(weights))

    peaks = {
        name: _local_peaks(curve, peak_order, candidate_threshold)
        for name, curve in curves.items()
    }
    local_candidates = sorted(set(peaks["interaction"] + peaks["relation"]))
    records: list[dict[str, Any]] = []
    used_local: set[int] = set()
    for global_index in peaks["global"]:
        neighbors = [
            index for index in local_candidates
            if abs(index - global_index) <= neighbor_radius
        ]
        if neighbors:
            chosen = max([global_index, *neighbors], key=lambda index: float(fused[index]))
            used_local.update(neighbors)
            evidence = ["global_change", "object_centric_support"]
            source = "otas_global_confirmed_or_corrected"
        else:
            continue
        records.append({
            "index": int(chosen),
            "source": source,
            "hard_boundary": False,
            "score": round(float(fused[chosen]), 6),
            "evidence": evidence,
        })

    # As in the official fusion code, an unusually strong object-centric peak
    # may contribute a boundary even when the global stream misses it.
    for local_index in local_candidates:
        if local_index in used_local or float(fused[local_index]) < strong_local_threshold:
            continue
        records.append({
            "index": int(local_index),
            "source": "otas_strong_object_centric",
            "hard_boundary": False,
            "score": round(float(fused[local_index]), 6),
            "evidence": ["strong_object_centric_change"],
        })
    records = _suppress_close_boundaries(records, times, min_gap_s)
    return {
        "boundaries": records,
        "runs": [
            {"start": start, "end": end}
            for start, end in zip(
                [0, *[int(item["index"]) for item in records]],
                [*[int(item["index"]) for item in records], sample_count],
            )
            if end > start
        ],
        "curves": curves,
        "fused_curve": fused,
        "peaks": peaks,
        "parameters": {
            "global_weight": global_weight,
            "interaction_weight": interaction_weight,
            "relation_weight": relation_weight,
            "peak_window_s": peak_window_s,
            "neighbor_s": neighbor_s,
            "min_gap_s": min_gap_s,
            "candidate_threshold": candidate_threshold,
            "strong_local_threshold": strong_local_threshold,
            "smoothing_s": smoothing_s,
        },
    }


def segment_video_with_otas(
    info: Any, samples: list[dict[str, Any]], feature_file: str | None = None,
    global_weight: float = 1.0, interaction_weight: float = 1.0,
    relation_weight: float = 1.0, peak_window_s: float = 0.75,
    neighbor_s: float = 0.75, min_gap_s: float = 0.75,
    candidate_threshold: float = 1.0, strong_local_threshold: float = 2.5,
    smoothing_s: float = 0.25,
) -> dict[str, Any]:
    streams, feature_source = extract_otas_features(info, samples, feature_file)
    result = otas_boundary_selection(
        streams, np.asarray([sample["time_s"] for sample in samples]),
        global_weight, interaction_weight, relation_weight, peak_window_s,
        neighbor_s, min_gap_s, candidate_threshold, strong_local_threshold,
        smoothing_s,
    )
    diagnostics = {
        "paper": PAPER_TITLE,
        "implementation": "OTAS-inspired training-free raw-video proxy",
        "exact_reproduction": False,
        "difference_from_paper": (
            "The paper's trained future-frame prediction, Detectron2 masks, "
            "and GAT object graph are replaced by deterministic appearance, "
            "expanded hand-neighborhood, and MediaPipe relation descriptors."
        ),
        "feature_source": feature_source,
        "sample_count": len(samples),
        "feature_dimensions": {
            name: int(features.shape[1]) for name, features in streams.items()
        },
        "boundary_count": len(result["boundaries"]),
        "final_segment_count": len(result["runs"]),
        "parameters": result["parameters"],
        "stream_peak_indices": result["peaks"],
        "boundary_indices": [int(item["index"]) for item in result["boundaries"]],
        "temporal_difference_curves": {
            name: [round(float(value), 8) for value in curve]
            for name, curve in result["curves"].items()
        },
        "fused_curve": [round(float(value), 8) for value in result["fused_curve"]],
    }
    return {
        "boundaries": result["boundaries"],
        "runs": result["runs"],
        "diagnostics": diagnostics,
    }
