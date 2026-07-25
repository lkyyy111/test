"""SaM-inspired split-and-merge baseline for raw egocentric videos.

The split and merge equations follow Xing and Zhao, "Unsupervised Action
Segmentation via Fast Learning of Semantically Consistent Actoms" (AAAI 2024).
The paper consumes dataset-provided IDT or HOF+VGG features.  For a standalone
raw-video pipeline we provide a deterministic, training-free HOF+appearance
descriptor, while also accepting an aligned precomputed feature matrix.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def _frame_descriptor(
    frame: np.ndarray, previous_gray: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact HOF + appearance descriptor and current gray frame."""
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Coarse HSV statistics preserve scene/object appearance without depending
    # on a learned model or an extra PyTorch installation.
    color_hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
    color_hist = _l2_normalize(color_hist.astype(np.float32))

    # Low-frequency DCT coefficients encode spatial layout more compactly than
    # flattened pixels.  Drop the DC term so brightness changes dominate less.
    dct_input = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = cv2.dct(dct_input)[:8, :8].reshape(-1)
    dct[0] = 0.0
    dct = _l2_normalize(dct)

    # Histogram of optical flow approximates the HOF component used by the
    # paper on instructional video features.
    flow_hist = np.zeros(10, dtype=np.float32)
    if previous_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        bins = np.floor(angle / 45.0).astype(np.int32) % 8
        histogram = np.bincount(bins.reshape(-1), weights=magnitude.reshape(-1), minlength=8)
        flow_hist[:8] = _l2_normalize(histogram.astype(np.float32))
        flow_hist[8] = float(np.mean(magnitude))
        flow_hist[9] = float(np.percentile(magnitude, 90))
        flow_hist = _l2_normalize(flow_hist)

    descriptor = _l2_normalize(np.concatenate([color_hist, dct, flow_hist]))
    return descriptor.astype(np.float32), gray


def extract_video_features(
    info: Any, samples: list[dict[str, Any]], feature_file: str | None = None,
) -> tuple[np.ndarray, str]:
    """Extract one feature row per existing 8 FPS analysis sample."""
    if feature_file:
        path = Path(feature_file).expanduser()
        features = np.asarray(np.load(path), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(samples):
            raise ValueError(
                f"--sam-feature-file 必须是与采样帧对齐的 N×D 矩阵；"
                f"当前需要 {len(samples)} 行，实际形状为 {features.shape}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("--sam-feature-file 含 NaN 或无穷值")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        return features, f"precomputed:{path}"

    target_frames = [int(sample["frame_index"]) for sample in samples]
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"SaM 无法打开视频：{info.path}")
    descriptors: list[np.ndarray] = []
    previous_gray: np.ndarray | None = None
    target_index = 0
    frame_index = 0
    try:
        with tqdm(total=len(samples), desc="SaM视觉特征", unit="frame") as progress:
            while target_index < len(target_frames):
                ok = cap.grab()
                if not ok:
                    break
                if frame_index == target_frames[target_index]:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    descriptor, previous_gray = _frame_descriptor(frame, previous_gray)
                    descriptors.append(descriptor)
                    target_index += 1
                    progress.update(1)
                frame_index += 1
    finally:
        cap.release()
    if len(descriptors) != len(samples):
        raise RuntimeError(
            f"SaM 特征只读取到 {len(descriptors)}/{len(samples)} 个采样帧"
        )
    return np.stack(descriptors), "opencv_hof_hsv_dct"


def _moving_average(features: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(features)))
    if window == 1:
        return features.copy()
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(features, ((left, right), (0, 0)), mode="edge")
    cumulative = np.vstack([
        np.zeros((1, features.shape[1]), dtype=np.float64),
        np.cumsum(padded, axis=0, dtype=np.float64),
    ])
    return ((cumulative[window:] - cumulative[:-window]) / window).astype(np.float32)


def _angles_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(np.sum(left_norm * right_norm, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _contiguous_runs(labels: np.ndarray) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            runs.append({
                "start": start,
                "end": index,
                "cluster_id": int(labels[start]),
            })
            start = index
    return runs


def sam_split_merge(
    features: np.ndarray, action_count: int, delta: float = 0.3,
    temporal_lambda: float = 0.001,
) -> dict[str, Any]:
    """Apply SaM local-minimum splitting and spatio-temporal merging."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) < 3:
        raise ValueError("SaM 至少需要 3 个二维特征样本")
    if not np.all(np.isfinite(features)):
        raise ValueError("SaM 输入特征含 NaN 或无穷值")
    if action_count < 1 or action_count >= len(features):
        raise ValueError("--sam-action-count 必须大于 0 且小于采样帧数")
    if not 0 < delta <= 1:
        raise ValueError("--sam-delta 必须在 (0, 1] 内")
    if temporal_lambda < 0:
        raise ValueError("--sam-lambda 不能为负数")

    sample_count = len(features)
    window = max(1, int(math.ceil(delta * sample_count / action_count)))
    smoothed = _moving_average(features, window)
    adjacent_angles = _angles_degrees(smoothed[:-1], smoothed[1:])
    angle_variance = max(float(np.var(adjacent_angles)), 1e-6)
    similarities = np.exp(-adjacent_angles / angle_variance)

    # Equation (4) of the paper: one local similarity minimum in each window.
    initial_boundaries: list[int] = []
    for start in range(0, sample_count - 1, window):
        end = min(start + window, sample_count - 1)
        if end <= start:
            continue
        boundary = start + int(np.argmin(similarities[start:end])) + 1
        if 0 < boundary < sample_count:
            initial_boundaries.append(boundary)
    initial_boundaries = sorted(set(initial_boundaries))
    endpoints = [0, *initial_boundaries, sample_count]
    clusters = [
        {"members": np.arange(start, end, dtype=np.int32), "source_actoms": [index]}
        for index, (start, end) in enumerate(zip(endpoints, endpoints[1:]), 1)
        if end > start
    ]
    initial_actom_count = len(clusters)
    if initial_actom_count < action_count:
        raise ValueError(
            f"SaM 初始 actom 只有 {initial_actom_count} 个，无法合并到 K={action_count}；"
            "请减小 --sam-action-count 或 --sam-delta"
        )

    merge_history: list[dict[str, Any]] = []
    while len(clusters) > action_count:
        centroids = np.stack([
            np.mean(smoothed[cluster["members"]], axis=0) for cluster in clusters
        ])
        times = np.asarray([
            float(np.mean(cluster["members"])) / sample_count for cluster in clusters
        ])
        centroid_angles = _angles_degrees(
            np.repeat(centroids, len(clusters), axis=0),
            np.tile(centroids, (len(clusters), 1)),
        ).reshape(len(clusters), len(clusters))
        spatial = np.exp(-centroid_angles / angle_variance)
        temporal = np.exp(-temporal_lambda * np.abs(times[:, None] - times[None, :]))
        graph = spatial * temporal
        np.fill_diagonal(graph, -np.inf)
        first, second = np.unravel_index(int(np.argmax(graph)), graph.shape)
        if second < first:
            first, second = second, first
        merge_history.append({
            "remaining_before": len(clusters),
            "left_source_actoms": list(clusters[first]["source_actoms"]),
            "right_source_actoms": list(clusters[second]["source_actoms"]),
            "spatio_temporal_similarity": round(float(graph[first, second]), 8),
        })
        clusters[first] = {
            "members": np.sort(np.concatenate([
                clusters[first]["members"], clusters[second]["members"],
            ])),
            "source_actoms": sorted([
                *clusters[first]["source_actoms"], *clusters[second]["source_actoms"],
            ]),
        }
        del clusters[second]

    clusters.sort(key=lambda cluster: int(np.min(cluster["members"])))
    labels = np.zeros(sample_count, dtype=np.int32)
    for cluster_id, cluster in enumerate(clusters, 1):
        labels[cluster["members"]] = cluster_id
    runs = _contiguous_runs(labels)
    final_boundaries = [run["end"] for run in runs[:-1]]
    boundary_records = [{
        "index": index,
        "source": "sam_actom_label_transition",
        "hard_boundary": False,
        "score": round(float(1.0 - similarities[index - 1]), 6),
        "evidence": ["SaM 合并后相邻采样帧的 actom 类别发生变化"],
    } for index in final_boundaries]
    return {
        "runs": runs,
        "boundaries": boundary_records,
        "diagnostics": {
            "paper": "Unsupervised Action Segmentation via Fast Learning of Semantically Consistent Actoms (AAAI 2024)",
            "sample_count": sample_count,
            "requested_action_count_k": action_count,
            "final_distinct_cluster_count": len(clusters),
            "final_contiguous_run_count": len(runs),
            "delta": delta,
            "temporal_lambda": temporal_lambda,
            "local_window_samples": window,
            "angle_variance": angle_variance,
            "initial_actom_count": initial_actom_count,
            "initial_boundaries": initial_boundaries,
            "final_boundaries": final_boundaries,
            "adjacent_similarity": [round(float(value), 8) for value in similarities],
            "merge_history": merge_history,
        },
    }


def segment_video_with_sam(
    info: Any, samples: list[dict[str, Any]], action_count: int,
    delta: float = 0.3, temporal_lambda: float = 0.001,
    feature_file: str | None = None,
) -> dict[str, Any]:
    features, feature_source = extract_video_features(info, samples, feature_file)
    result = sam_split_merge(features, action_count, delta, temporal_lambda)
    result["diagnostics"]["feature_source"] = feature_source
    result["diagnostics"]["feature_dimension"] = int(features.shape[1])
    return result
