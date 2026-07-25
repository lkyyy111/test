from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_BASE_YAW = {
    "thumb": 0.82,
    "index": 0.16,
    "middle": 0.0,
    "ring": -0.12,
    "pinky": -0.25,
}
JOINT_NAMES = tuple(
    f"{finger}_{joint}"
    for finger in FINGER_LANDMARKS
    for joint in ("abd", "mcp", "pip", "dip")
)


@dataclass(frozen=True)
class DexHandConfig:
    hand: str = "right"
    control_fps: float = 50.0
    gap_tolerance_s: float = 0.5
    smoothing_s: float = 0.18
    forward_range_m: float = 0.18
    lateral_range_m: float = 0.12
    vertical_range_m: float = 0.10
    root_height_m: float = 0.22
    render_fps: float = 30.0
    render_width: int = 640
    render_height: int = 480


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return vector / norm


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    a, b = _normalize(first), _normalize(second)
    if a is None or b is None:
        return 0.0
    return float(math.acos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def _points_array(hand: dict[str, Any], key: str) -> np.ndarray | None:
    points = hand.get(key)
    if not points or len(points) != 21:
        return None
    array = np.asarray(
        [[point.get("x"), point.get("y"), point.get("z")] for point in points],
        dtype=float,
    )
    return array if array.shape == (21, 3) and np.isfinite(array).all() else None


def _hand_for_side(sample: dict[str, Any], side: str) -> dict[str, Any] | None:
    return next(
        (hand for hand in sample.get("hands", []) if hand.get("side") == side),
        None,
    )


def hand_landmarks_to_joint_targets(hand: dict[str, Any], side: str) -> np.ndarray:
    """Convert one MediaPipe hand skeleton to 20 anatomical joint targets.

    The targets describe finger flexion and spread in a palm-local frame. They
    do not depend on the human hand's physical size, which makes the mapping
    suitable for a robot hand with different link lengths.
    """
    points = _points_array(hand, "landmarks_3d_relative")
    if points is None:
        points = _points_array(hand, "landmarks_2d_relative")
    if points is None:
        raise ValueError("a valid set of 21 MediaPipe landmarks is required")

    forward = _normalize(points[9] - points[0])
    lateral = _normalize(points[5] - points[17])
    if forward is None or lateral is None:
        raise ValueError("degenerate palm landmarks")
    lateral = _normalize(lateral - float(np.dot(lateral, forward)) * forward)
    if lateral is None:
        raise ValueError("degenerate palm coordinate frame")

    mirror = -1.0 if side == "left" else 1.0
    targets: list[float] = []
    for finger, indices in FINGER_LANDMARKS.items():
        a, b, c, d = indices
        first = points[b] - points[a]
        # Thumb starts at landmark 1 rather than the wrist; for the other
        # fingers wrist->MCP supplies a stable palm-to-finger reference.
        parent = points[a] - points[0]
        if finger == "thumb":
            parent = forward
        first_unit = _normalize(first)
        if first_unit is None:
            spread = FINGER_BASE_YAW[finger]
        else:
            spread = math.atan2(
                float(np.dot(first_unit, lateral)),
                float(np.dot(first_unit, forward)),
            )
        abd = mirror * (spread - FINGER_BASE_YAW[finger])
        mcp = _angle_between(parent, first)
        pip = _angle_between(points[b] - points[a], points[c] - points[b])
        dip = _angle_between(points[c] - points[b], points[d] - points[c])
        targets.extend([
            float(np.clip(abd, -0.70, 0.70)),
            float(np.clip(mcp, 0.0, 1.45)),
            float(np.clip(pip, 0.0, 1.65)),
            float(np.clip(dip, 0.0, 1.35)),
        ])
    return np.asarray(targets, dtype=float)


def _resample_with_gap_hold(
    observed_times: np.ndarray,
    observed_values: np.ndarray,
    target_times: np.ndarray,
    gap_tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty((len(target_times), observed_values.shape[1]), dtype=float)
    supported = np.zeros(len(target_times), dtype=bool)
    for target_index, time_s in enumerate(target_times):
        after = int(np.searchsorted(observed_times, time_s, side="left"))
        if after == 0:
            result[target_index] = observed_values[0]
            supported[target_index] = abs(time_s - observed_times[0]) <= gap_tolerance_s
        elif after >= len(observed_times):
            result[target_index] = observed_values[-1]
            supported[target_index] = abs(time_s - observed_times[-1]) <= gap_tolerance_s
        else:
            before = after - 1
            left_time, right_time = observed_times[before], observed_times[after]
            gap_s = right_time - left_time
            if gap_s <= gap_tolerance_s + 1e-9:
                alpha = (time_s - left_time) / max(gap_s, 1e-9)
                result[target_index] = (
                    (1.0 - alpha) * observed_values[before]
                    + alpha * observed_values[after]
                )
                supported[target_index] = True
            else:
                # A long occlusion holds the last pose but remains explicitly
                # unsupported: no unobserved hand motion is fabricated.
                result[target_index] = observed_values[before]
    return result, supported


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.column_stack([
        np.convolve(padded[:, column], kernel, mode="valid")
        for column in range(values.shape[1])
    ])


def build_dex_hand_targets(
    samples: list[dict[str, Any]],
    duration_s: float,
    config: DexHandConfig,
) -> dict[str, Any]:
    """Build a 50 Hz root trajectory and 20-DoF pose from MediaPipe samples."""
    if config.hand not in {"left", "right"}:
        raise ValueError("retarget hand must be 'left' or 'right'")
    if config.control_fps <= 0 or duration_s <= 0:
        raise ValueError("control_fps and duration_s must be positive")

    observations: list[np.ndarray] = []
    rejected = 0
    for sample in samples:
        hand = _hand_for_side(sample, config.hand)
        if hand is None:
            continue
        points_2d = _points_array(hand, "landmarks_2d_relative")
        if points_2d is None:
            rejected += 1
            continue
        try:
            joints = hand_landmarks_to_joint_targets(hand, config.hand)
        except ValueError:
            rejected += 1
            continue
        wrist = points_2d[0, :2]
        palm_width = float(np.linalg.norm(points_2d[5, :2] - points_2d[17, :2]))
        palm_length = float(np.linalg.norm(points_2d[9, :2] - points_2d[0, :2]))
        scale = 0.5 * (palm_width + palm_length)
        if not np.isfinite(scale) or scale < 1e-5:
            rejected += 1
            continue
        observations.append(np.concatenate([
            [float(sample["time_s"]), wrist[0], wrist[1], scale], joints,
        ]))
    if len(observations) < 3:
        raise RuntimeError(
            f"{config.hand} hand has only {len(observations)} usable 21-point samples; "
            "at least 3 are required for dexterous-hand retargeting"
        )

    observed = np.asarray(observations, dtype=float)
    control_times = np.arange(
        0.0, duration_s + 0.5 / config.control_fps,
        1.0 / config.control_fps, dtype=float,
    )
    resampled, supported = _resample_with_gap_hold(
        observed[:, 0], observed[:, 1:], control_times, config.gap_tolerance_s,
    )
    smooth_window = max(1, int(round(config.smoothing_s * config.control_fps)))
    resampled = _moving_average(resampled, smooth_window)
    image_x, image_y, hand_scale = resampled[:, 0], resampled[:, 1], resampled[:, 2]
    joint_targets = resampled[:, 3:]

    u_low, u_high = np.quantile(observed[:, 1], [0.05, 0.95])
    v_low, v_high = np.quantile(observed[:, 2], [0.05, 0.95])
    scale_low, scale_high = np.quantile(observed[:, 3], [0.05, 0.95])
    u_center, v_center = 0.5 * (u_low + u_high), 0.5 * (v_low + v_high)
    u_half_span = max(0.5 * (u_high - u_low), 0.025)
    v_half_span = max(0.5 * (v_high - v_low), 0.025)
    scale_span = max(scale_high - scale_low, 0.015)

    # In an egocentric camera, a smaller apparent palm generally means the hand
    # has moved farther away. This yields a bounded relative-depth trajectory,
    # not a metric depth estimate.
    forward_fraction = np.clip((scale_high - hand_scale) / scale_span, 0.0, 1.0)
    root_xyz = np.column_stack([
        config.forward_range_m * forward_fraction,
        np.clip(-(image_x - u_center) / u_half_span, -1.0, 1.0)
        * config.lateral_range_m,
        config.root_height_m
        + np.clip((v_center - image_y) / v_half_span, -1.0, 1.0)
        * config.vertical_range_m,
    ])
    return {
        "times_s": control_times,
        "image_xy_scale": np.column_stack([image_x, image_y, hand_scale]),
        "root_xyz": root_xyz,
        "joint_targets": joint_targets,
        "supported_mask": supported,
        "observed_sample_count": len(observations),
        "rejected_sample_count": rejected,
        "observed_coverage": len(observations) / max(1, len(samples)),
        "mapping_stats": {
            "image_u_05": float(u_low),
            "image_u_95": float(u_high),
            "image_v_05": float(v_low),
            "image_v_95": float(v_high),
            "hand_scale_05": float(scale_low),
            "hand_scale_95": float(scale_high),
            "depth_source": "inverse_relative_palm_scale",
        },
    }


def _finger_xml(
    side: str,
    finger: str,
    base_x: float,
    anatomical_y: float,
    lengths: tuple[float, float, float],
    radius: float,
) -> str:
    mirror = -1.0 if side == "left" else 1.0
    world_y = mirror * anatomical_y
    base_yaw = mirror * FINGER_BASE_YAW[finger]
    color = "0.18 0.48 0.82 1" if side == "left" else "0.92 0.42 0.20 1"
    l1, l2, l3 = lengths
    return f"""
      <body name="{side}_{finger}_1" pos="{base_x:.4f} {world_y:.4f} 0" euler="0 0 {base_yaw:.6f}">
        <joint name="{finger}_abd" type="hinge" axis="0 0 1" range="-0.70 0.70" damping="0.15"/>
        <joint name="{finger}_mcp" type="hinge" axis="0 -1 0" range="0 1.45" damping="0.15"/>
        <geom type="capsule" fromto="0 0 0 {l1:.4f} 0 0" size="{radius:.4f}" rgba="{color}"/>
        <body name="{side}_{finger}_2" pos="{l1:.4f} 0 0">
          <joint name="{finger}_pip" type="hinge" axis="0 -1 0" range="0 1.65" damping="0.12"/>
          <geom type="capsule" fromto="0 0 0 {l2:.4f} 0 0" size="{radius * 0.9:.4f}" rgba="{color}"/>
          <body name="{side}_{finger}_3" pos="{l2:.4f} 0 0">
            <joint name="{finger}_dip" type="hinge" axis="0 -1 0" range="0 1.35" damping="0.10"/>
            <geom type="capsule" fromto="0 0 0 {l3:.4f} 0 0" size="{radius * 0.8:.4f}" rgba="{color}"/>
          </body>
        </body>
      </body>"""


def build_dex_hand_mjcf(side: str) -> str:
    """Return a self-contained five-finger, 20-DoF MuJoCo hand model."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    mirror = -1.0 if side == "left" else 1.0
    color = "0.12 0.36 0.72 1" if side == "left" else "0.82 0.30 0.10 1"
    fingers = "".join([
        _finger_xml(side, "thumb", 0.015, 0.050, (0.043, 0.034, 0.028), 0.010),
        _finger_xml(side, "index", 0.078, 0.032, (0.050, 0.032, 0.025), 0.009),
        _finger_xml(side, "middle", 0.086, 0.010, (0.055, 0.036, 0.027), 0.0095),
        _finger_xml(side, "ring", 0.080, -0.014, (0.050, 0.033, 0.025), 0.009),
        _finger_xml(side, "pinky", 0.067, -0.034, (0.041, 0.027, 0.022), 0.008),
    ])
    return f"""<mujoco model="egocentric_dex_hand_{side}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/></visual>
  <worldbody>
    <light pos="-0.2 -0.3 1.2" dir="0.2 0.1 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="0.4 0.5 0.7" dir="-0.2 -0.3 -1" diffuse="0.45 0.50 0.60"/>
    <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 0" rgba="0.10 0.12 0.16 1"/>
    <camera name="first_person" pos="-0.48 0 0.30" xyaxes="0 -1 0 0.108 0 0.994" fovy="52"/>
    <body name="{side}_palm" pos="0 0 0.22">
      <freejoint name="{side}_root"/>
      <geom type="box" pos="0.032 0 0" size="0.058 0.046 0.012" rgba="{color}"/>
      <geom type="capsule" fromto="-0.060 0 0 -0.005 0 0" size="0.022" rgba="{color}"/>
      {fingers}
    </body>
  </worldbody>
</mujoco>"""


def _load_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required only when dexterous-hand retargeting is enabled. "
            "Install it with: pip install -r requirements-retarget.txt"
        ) from exc
    return mujoco


def _name_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"dexterous hand model is missing required object: {name}")
    return object_id


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _apply_pose(
    mujoco: Any,
    model: Any,
    data: Any,
    root_qpos: int,
    joint_qpos: np.ndarray,
    root_xyz: np.ndarray,
    joint_targets: np.ndarray,
) -> None:
    data.qpos[root_qpos:root_qpos + 3] = root_xyz
    data.qpos[root_qpos + 3:root_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[joint_qpos] = joint_targets
    mujoco.mj_forward(model, data)


def _render_retarget(
    mujoco: Any,
    model: Any,
    side: str,
    root_xyz: np.ndarray,
    joint_targets: np.ndarray,
    supported: np.ndarray,
    output: Path,
    config: DexHandConfig,
) -> None:
    data = mujoco.MjData(model)
    root_id = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_root")
    root_qpos = int(model.jnt_qposadr[root_id])
    joint_ids = [
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ]
    joint_qpos = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    renderer = mujoco.Renderer(
        model, height=config.render_height, width=config.render_width,
    )
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), config.render_fps,
        (config.render_width, config.render_height),
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"unable to create dexterous-hand video: {output}")
    frame_count = max(
        1,
        int(round((len(joint_targets) - 1) / config.control_fps * config.render_fps)) + 1,
    )
    try:
        for frame_index in range(frame_count):
            time_s = frame_index / config.render_fps
            index = min(len(joint_targets) - 1, int(round(time_s * config.control_fps)))
            _apply_pose(
                mujoco, model, data, root_qpos, joint_qpos,
                root_xyz[index], joint_targets[index],
            )
            renderer.update_scene(data, camera="first_person")
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            state = "TRACKED" if supported[index] else "HAND MISSING - HOLD"
            state_color = (80, 230, 120) if supported[index] else (80, 170, 255)
            cv2.putText(
                frame_bgr, f"{side.upper()} DEX HAND  t={time_s:.2f}s", (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr, state, (18, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, state_color, 2, cv2.LINE_AA,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()
        renderer.close()


def combine_retarget_videos(
    left_video: Path, right_video: Path, output: Path,
) -> dict[str, Any]:
    """Write a synchronized first-person view of two retargeted hands."""
    left_cap = cv2.VideoCapture(str(left_video))
    right_cap = cv2.VideoCapture(str(right_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        raise RuntimeError("unable to open left/right dex-hand videos for composition")
    left_fps = float(left_cap.get(cv2.CAP_PROP_FPS))
    right_fps = float(right_cap.get(cv2.CAP_PROP_FPS))
    fps = min(value for value in (left_fps, right_fps) if value > 0)
    width = max(
        int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    height = max(
        int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * width, height),
    )
    if not writer.isOpened():
        left_cap.release()
        right_cap.release()
        raise RuntimeError(f"unable to create bimanual dex-hand video: {output}")
    frame_count = 0
    try:
        while True:
            left_ok, left_frame = left_cap.read()
            right_ok, right_frame = right_cap.read()
            if not left_ok and not right_ok:
                break
            if not left_ok:
                left_frame = np.zeros((height, width, 3), dtype=np.uint8)
            if not right_ok:
                right_frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(cv2.hconcat([
                cv2.resize(left_frame, (width, height)),
                cv2.resize(right_frame, (width, height)),
            ]))
            frame_count += 1
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()
    return {"fps": fps, "frame_count": frame_count, "width": 2 * width, "height": height}


def run_dex_hand_retarget(
    info: Any,
    samples: list[dict[str, Any]],
    output_dir: Path,
    args: Any,
) -> dict[str, Any]:
    config = DexHandConfig(
        hand=args.retarget_hand,
        control_fps=args.retarget_control_fps,
        gap_tolerance_s=args.retarget_gap_tolerance,
        forward_range_m=args.dex_hand_forward_range,
        lateral_range_m=args.dex_hand_lateral_range,
        vertical_range_m=args.dex_hand_vertical_range,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = build_dex_hand_targets(samples, float(info.duration_s), config)
    model_path = output_dir / "dex_hand_model.xml"
    model_path.write_text(build_dex_hand_mjcf(config.hand), encoding="utf-8")

    times = trajectory["times_s"]
    root_xyz = trajectory["root_xyz"]
    joints = trajectory["joint_targets"]
    image = trajectory["image_xy_scale"]
    supported = trajectory["supported_mask"]
    _write_csv(
        output_dir / "root_trajectory.csv",
        ["time_s", "image_x", "image_y", "hand_scale", "root_x", "root_y", "root_z", "supported_mask"],
        [{
            "time_s": float(times[index]),
            "image_x": float(image[index, 0]),
            "image_y": float(image[index, 1]),
            "hand_scale": float(image[index, 2]),
            "root_x": float(root_xyz[index, 0]),
            "root_y": float(root_xyz[index, 1]),
            "root_z": float(root_xyz[index, 2]),
            "supported_mask": bool(supported[index]),
        } for index in range(len(times))],
    )
    _write_csv(
        output_dir / "joint_trajectory.csv",
        ["time_s", *JOINT_NAMES],
        [{"time_s": float(times[index]), **{
            name: float(joints[index, joint_index])
            for joint_index, name in enumerate(JOINT_NAMES)
        }} for index in range(len(times))],
    )

    metrics = {
        "schema_version": "1.0",
        "profile": "egocentric_five_finger_dexterous_hand",
        "source_video": str(info.path),
        "hand": config.hand,
        "model": "procedural_20dof_five_finger_hand",
        "coordinate_note": (
            "Root x is bounded relative depth estimated from inverse apparent palm scale; "
            "it is not calibrated metric camera depth. Finger joints are palm-local pose targets."
        ),
        "config": asdict(config),
        "observed_sample_count": trajectory["observed_sample_count"],
        "rejected_sample_count": trajectory["rejected_sample_count"],
        "observed_coverage": round(float(trajectory["observed_coverage"]), 4),
        "supported_control_ratio": round(float(np.mean(supported)), 4),
        "mapping_stats": trajectory["mapping_stats"],
        "outputs": {
            "model": "dex_hand_model.xml",
            "root_trajectory": "root_trajectory.csv",
            "joint_trajectory": "joint_trajectory.csv",
            "rendered_video": None if args.skip_video_outputs else "retarget.mp4",
        },
    }
    (output_dir / "retarget_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not args.skip_video_outputs:
        mujoco = _load_mujoco()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        _render_retarget(
            mujoco, model, config.hand, root_xyz, joints, supported,
            output_dir / "retarget.mp4", config,
        )
    return metrics
