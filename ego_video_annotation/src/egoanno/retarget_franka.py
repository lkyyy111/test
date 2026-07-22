from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))


@dataclass(frozen=True)
class RetargetConfig:
    hand: str = "right"
    control_fps: float = 50.0
    gap_tolerance_s: float = 0.5
    smoothing_s: float = 0.20
    robot_x_m: float = 0.50
    robot_y_range_m: float = 0.10
    robot_z_low_m: float = 0.25
    robot_z_high_m: float = 0.50
    ik_damping: float = 0.05
    ik_orientation_weight: float = 0.20
    ik_max_iterations: int = 80
    ik_max_joint_step: float = 0.10
    ik_nullspace_gain: float = 0.05
    render_fps: float = 30.0
    render_width: int = 640
    render_height: int = 480


def _finite_wrist(sample: dict[str, Any], side: str) -> tuple[float, float] | None:
    for hand in sample.get("hands", []):
        if hand.get("side") != side:
            continue
        landmarks = hand.get("landmarks_2d_relative") or []
        if not landmarks:
            return None
        wrist = landmarks[0]
        x, y = wrist.get("x"), wrist.get("y")
        if x is None or y is None or not np.isfinite([x, y]).all():
            return None
        return float(x), float(y)
    return None


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
            continue
        if after >= len(observed_times):
            result[target_index] = observed_values[-1]
            supported[target_index] = abs(time_s - observed_times[-1]) <= gap_tolerance_s
            continue
        before = after - 1
        left_time, right_time = observed_times[before], observed_times[after]
        gap_s = right_time - left_time
        if gap_s <= gap_tolerance_s + 1e-9:
            alpha = (time_s - left_time) / max(gap_s, 1e-9)
            result[target_index] = (
                (1.0 - alpha) * observed_values[before] + alpha * observed_values[after]
            )
            supported[target_index] = True
        else:
            # Long missing intervals must not invent a hand path. Hold the last
            # observed target until tracking resumes and keep the support mask false.
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


def build_short_lift_targets(
    samples: list[dict[str, Any]],
    duration_s: float,
    config: RetargetConfig,
) -> dict[str, Any]:
    """Map one observed wrist path to a vertical Franka workspace plane.

    This profile is intentionally a 2.5D trajectory-shape retarget. MediaPipe
    world landmarks are hand-relative and therefore are not used as absolute
    robot translations.
    """
    if config.hand not in {"left", "right"}:
        raise ValueError("retarget hand must be 'left' or 'right'")
    if config.control_fps <= 0 or duration_s <= 0:
        raise ValueError("control_fps and duration_s must be positive")
    if config.robot_z_high_m <= config.robot_z_low_m:
        raise ValueError("robot_z_high_m must be greater than robot_z_low_m")

    observations: list[tuple[float, float, float]] = []
    for sample in samples:
        wrist = _finite_wrist(sample, config.hand)
        if wrist is not None:
            observations.append((float(sample["time_s"]), wrist[0], wrist[1]))
    if len(observations) < 3:
        raise RuntimeError(
            f"{config.hand} hand has only {len(observations)} valid wrist samples; "
            "at least 3 are required for Franka retargeting"
        )

    observed = np.asarray(observations, dtype=float)
    control_times = np.arange(
        0.0, duration_s + 0.5 / config.control_fps, 1.0 / config.control_fps,
        dtype=float,
    )
    image_xy, supported = _resample_with_gap_hold(
        observed[:, 0], observed[:, 1:3], control_times, config.gap_tolerance_s,
    )
    smooth_window = max(1, int(round(config.smoothing_s * config.control_fps)))
    image_xy = _moving_average(image_xy, smooth_window)

    u, v = image_xy[:, 0], image_xy[:, 1]
    u_low, u_high = np.quantile(u, [0.05, 0.95])
    v_top, v_bottom = np.quantile(v, [0.05, 0.95])
    u_center = 0.5 * (u_low + u_high)
    u_half_span = max(0.5 * (u_high - u_low), 0.02)
    v_span = max(v_bottom - v_top, 0.03)

    target_y = np.clip(
        (u - u_center) / u_half_span, -1.0, 1.0,
    ) * config.robot_y_range_m
    lift_fraction = np.clip((v_bottom - v) / v_span, 0.0, 1.0)
    target_z = (
        config.robot_z_low_m
        + lift_fraction * (config.robot_z_high_m - config.robot_z_low_m)
    )
    target_xyz = np.column_stack([
        np.full(len(control_times), config.robot_x_m), target_y, target_z,
    ])

    return {
        "times_s": control_times,
        "image_xy": image_xy,
        "target_xyz": target_xyz,
        "supported_mask": supported,
        "observed_sample_count": len(observations),
        "observed_coverage": len(observations) / max(1, len(samples)),
        "mapping_stats": {
            "image_u_05": float(u_low),
            "image_u_95": float(u_high),
            "image_v_05": float(v_top),
            "image_v_95": float(v_bottom),
        },
    }


def _resolve_franka_model(explicit_path: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie:
        root = Path(menagerie).expanduser()
        candidates.extend([
            root / "franka_emika_panda" / "scene.xml",
            root / "franka_emika_panda" / "panda.xml",
        ])
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        project_root / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml",
        project_root / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    try:
        from robot_descriptions import panda_mj_description

        model_path = Path(panda_mj_description.MJCF_PATH)
        if model_path.is_file():
            return model_path.resolve()
    except (ImportError, ModuleNotFoundError):
        pass

    raise RuntimeError(
        "Franka MJCF model was not found. Install requirements-retarget.txt, "
        "pass --franka-model /path/to/franka_emika_panda/scene.xml, or set "
        "MUJOCO_MENAGERIE_PATH to a mujoco_menagerie checkout."
    )


def _load_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required only when --retarget-franka is enabled. "
            "Install it with: pip install -r requirements-retarget.txt"
        ) from exc
    return mujoco


def _name_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"Franka model is missing required object: {name}")
    return object_id


def _reset_home(mujoco: Any, model: Any, data: Any, joint_ids: list[int]) -> np.ndarray:
    home_key = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
    else:
        mujoco.mj_resetData(model, data)
        fallback = np.asarray([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.8])
        for joint_id, value in zip(joint_ids, fallback):
            qpos_index = int(model.jnt_qposadr[joint_id])
            low, high = model.jnt_range[joint_id]
            data.qpos[qpos_index] = np.clip(value, low, high)
    mujoco.mj_forward(model, data)
    return np.asarray([
        data.qpos[int(model.jnt_qposadr[joint_id])] for joint_id in joint_ids
    ], dtype=float)


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * sum(
        (np.cross(current[:, axis], target[:, axis]) for axis in range(3)),
        start=np.zeros(3, dtype=float),
    )


def _solve_ik_trajectory(
    mujoco: Any,
    model: Any,
    targets: np.ndarray,
    config: RetargetConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = mujoco.MjData(model)
    joint_ids = [
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINT_NAMES
    ]
    qpos_indices = np.asarray([model.jnt_qposadr[index] for index in joint_ids], dtype=int)
    dof_indices = np.asarray([model.jnt_dofadr[index] for index in joint_ids], dtype=int)
    hand_body = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    home_q = _reset_home(mujoco, model, data, joint_ids)
    fixed_orientation = np.asarray(data.xmat[hand_body], dtype=float).reshape(3, 3).copy()
    joint_low = np.asarray([model.jnt_range[index][0] for index in joint_ids]) + 1e-5
    joint_high = np.asarray([model.jnt_range[index][1] for index in joint_ids]) - 1e-5

    q_trajectory = np.empty((len(targets), 7), dtype=float)
    kinematic_xyz = np.empty((len(targets), 3), dtype=float)
    position_iterations: list[int] = []
    q = home_q.copy()
    for target_index, target in enumerate(targets):
        iterations_used = config.ik_max_iterations
        for iteration in range(config.ik_max_iterations):
            data.qpos[qpos_indices] = q
            mujoco.mj_forward(model, data)
            current_position = np.asarray(data.xpos[hand_body], dtype=float)
            current_orientation = np.asarray(data.xmat[hand_body], dtype=float).reshape(3, 3)
            position_error = target - current_position
            orientation_error = _rotation_error(current_orientation, fixed_orientation)
            if np.linalg.norm(position_error) < 5e-4 and np.linalg.norm(orientation_error) < 5e-3:
                iterations_used = iteration + 1
                break

            jacobian_position = np.zeros((3, model.nv), dtype=float)
            jacobian_rotation = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacBody(
                model, data, jacobian_position, jacobian_rotation, hand_body,
            )
            jacobian = np.vstack([
                jacobian_position[:, dof_indices],
                config.ik_orientation_weight * jacobian_rotation[:, dof_indices],
            ])
            error = np.concatenate([
                position_error,
                config.ik_orientation_weight * orientation_error,
            ])
            regularized = (
                jacobian @ jacobian.T
                + (config.ik_damping ** 2) * np.eye(jacobian.shape[0])
            )
            try:
                pseudo_inverse = jacobian.T @ np.linalg.solve(
                    regularized, np.eye(regularized.shape[0]),
                )
            except np.linalg.LinAlgError:
                pseudo_inverse = np.linalg.pinv(jacobian)
            delta_q = pseudo_inverse @ error
            nullspace = np.eye(7) - pseudo_inverse @ jacobian
            delta_q += nullspace @ (config.ik_nullspace_gain * (home_q - q))
            step_norm = np.linalg.norm(delta_q)
            if step_norm > config.ik_max_joint_step:
                delta_q *= config.ik_max_joint_step / step_norm
            q = np.clip(q + delta_q, joint_low, joint_high)

        data.qpos[qpos_indices] = q
        mujoco.mj_forward(model, data)
        q_trajectory[target_index] = q
        kinematic_xyz[target_index] = data.xpos[hand_body]
        position_iterations.append(iterations_used)

    position_error = np.linalg.norm(kinematic_xyz - targets, axis=1)
    return q_trajectory, kinematic_xyz, {
        "mean_iterations": float(np.mean(position_iterations)),
        "max_iterations": int(max(position_iterations)),
        "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "position_max_error_m": float(np.max(position_error)),
    }


def _simulate_joint_tracking(
    mujoco: Any,
    model: Any,
    q_trajectory: np.ndarray,
    control_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    joint_ids = [
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINT_NAMES
    ]
    actuator_ids = [
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ARM_ACTUATOR_NAMES
    ]
    qpos_indices = np.asarray([model.jnt_qposadr[index] for index in joint_ids], dtype=int)
    hand_body = _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    data.qpos[qpos_indices] = q_trajectory[0]
    data.ctrl[actuator_ids] = q_trajectory[0]
    if model.nu > 7:
        data.ctrl[7] = 0.0  # closed standard Panda gripper; no dexterous retarget
    mujoco.mj_forward(model, data)

    achieved_q = np.empty_like(q_trajectory)
    achieved_xyz = np.empty((len(q_trajectory), 3), dtype=float)
    # q_trajectory has already been sampled at the requested controller rate.
    substeps = max(1, int(round((1.0 / control_fps) / model.opt.timestep)))
    for index, target_q in enumerate(q_trajectory):
        data.ctrl[actuator_ids] = target_q
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        achieved_q[index] = data.qpos[qpos_indices]
        achieved_xyz[index] = data.xpos[hand_body]
    return achieved_q, achieved_xyz


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_retarget(
    mujoco: Any,
    model: Any,
    q_trajectory: np.ndarray,
    targets: np.ndarray,
    output: Path,
    control_fps: float,
    config: RetargetConfig,
) -> None:
    data = mujoco.MjData(model)
    joint_ids = [
        _name_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINT_NAMES
    ]
    qpos_indices = np.asarray([model.jnt_qposadr[index] for index in joint_ids], dtype=int)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.render_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.render_height)
    renderer = mujoco.Renderer(
        model, height=config.render_height, width=config.render_width,
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.40, 0.0, 0.35]
    camera.distance = 1.45
    camera.azimuth = 135.0
    camera.elevation = -20.0
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), config.render_fps,
        (config.render_width, config.render_height),
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"unable to create MuJoCo retarget video: {output}")
    frame_count = max(1, int(round((len(q_trajectory) - 1) / control_fps * config.render_fps)) + 1)
    try:
        for frame_index in range(frame_count):
            time_s = frame_index / config.render_fps
            trajectory_index = min(
                len(q_trajectory) - 1, int(round(time_s * control_fps)),
            )
            data.qpos[qpos_indices] = q_trajectory[trajectory_index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if renderer.scene.ngeom < renderer.scene.maxgeom:
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.asarray([0.025, 0.0, 0.0]),
                    pos=targets[trajectory_index],
                    mat=np.eye(3).reshape(-1),
                    rgba=np.asarray([0.15, 0.65, 1.0, 0.9]),
                )
                renderer.scene.ngeom += 1
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame_bgr, f"target t={time_s:.2f}s", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()
        renderer.close()


def combine_retarget_videos(
    left_video: Path, right_video: Path, output: Path,
) -> dict[str, Any]:
    """Write a synchronized side-by-side view of two independent Frankas."""
    left_cap = cv2.VideoCapture(str(left_video))
    right_cap = cv2.VideoCapture(str(right_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        left_cap.release()
        right_cap.release()
        raise RuntimeError("unable to open left/right Franka videos for composition")

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
        raise RuntimeError(f"unable to create bimanual Franka video: {output}")

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
            left_frame = cv2.resize(left_frame, (width, height))
            right_frame = cv2.resize(right_frame, (width, height))
            cv2.putText(
                left_frame, "LEFT HAND -> FRANKA", (18, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                right_frame, "RIGHT HAND -> FRANKA", (18, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(cv2.hconcat([left_frame, right_frame]))
            frame_count += 1
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": 2 * width,
        "height": height,
    }


def run_franka_retarget(
    info: Any,
    samples: list[dict[str, Any]],
    output_dir: Path,
    args: Any,
) -> dict[str, Any]:
    config = RetargetConfig(
        hand=args.retarget_hand,
        control_fps=args.retarget_control_fps,
        gap_tolerance_s=args.retarget_gap_tolerance,
        robot_x_m=args.retarget_robot_x,
        robot_y_range_m=args.retarget_y_range,
        robot_z_low_m=args.retarget_z_low,
        robot_z_high_m=args.retarget_z_high,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = build_short_lift_targets(samples, float(info.duration_s), config)
    model_path = _resolve_franka_model(args.franka_model)
    mujoco = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    q_trajectory, kinematic_xyz, ik_metrics = _solve_ik_trajectory(
        mujoco, model, trajectory["target_xyz"], config,
    )

    # Simulate at the real controller period. Keep this value local rather than
    # changing the source MJCF timestep.
    achieved_q, achieved_xyz = _simulate_joint_tracking(
        mujoco, model, q_trajectory, config.control_fps,
    )

    times = trajectory["times_s"]
    target_xyz = trajectory["target_xyz"]
    errors = np.linalg.norm(achieved_xyz - target_xyz, axis=1)
    _write_csv(
        output_dir / "target_trajectory.csv",
        ["time_s", "image_x", "image_y", "target_x", "target_y", "target_z", "supported_mask"],
        [
            {
                "time_s": float(times[index]),
                "image_x": float(trajectory["image_xy"][index, 0]),
                "image_y": float(trajectory["image_xy"][index, 1]),
                "target_x": float(target_xyz[index, 0]),
                "target_y": float(target_xyz[index, 1]),
                "target_z": float(target_xyz[index, 2]),
                "supported_mask": bool(trajectory["supported_mask"][index]),
            }
            for index in range(len(times))
        ],
    )
    _write_csv(
        output_dir / "joint_trajectory.csv",
        ["time_s", *ARM_JOINT_NAMES],
        [
            {"time_s": float(times[index]), **{
                name: float(achieved_q[index, joint_index])
                for joint_index, name in enumerate(ARM_JOINT_NAMES)
            }}
            for index in range(len(times))
        ],
    )
    _write_csv(
        output_dir / "achieved_trajectory.csv",
        ["time_s", "achieved_x", "achieved_y", "achieved_z", "target_error_m"],
        [
            {
                "time_s": float(times[index]),
                "achieved_x": float(achieved_xyz[index, 0]),
                "achieved_y": float(achieved_xyz[index, 1]),
                "achieved_z": float(achieved_xyz[index, 2]),
                "target_error_m": float(errors[index]),
            }
            for index in range(len(times))
        ],
    )

    metrics = {
        "schema_version": "0.1",
        "profile": "short_single_hand_vertical_plane",
        "source_video": str(info.path),
        "franka_model": str(model_path),
        "config": asdict(config),
        "observed_sample_count": trajectory["observed_sample_count"],
        "observed_coverage": round(float(trajectory["observed_coverage"]), 4),
        "supported_control_ratio": round(float(np.mean(trajectory["supported_mask"])), 4),
        "mapping_stats": trajectory["mapping_stats"],
        "ik": ik_metrics,
        "simulation": {
            "position_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "position_mean_error_m": float(np.mean(errors)),
            "position_p95_error_m": float(np.quantile(errors, 0.95)),
            "position_max_error_m": float(np.max(errors)),
        },
        "rendered_video": None if args.skip_video_outputs else "retarget.mp4",
    }
    (output_dir / "retarget_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not args.skip_video_outputs:
        _render_retarget(
            mujoco, model, achieved_q, target_xyz, output_dir / "retarget.mp4",
            config.control_fps, config,
        )
    return metrics
