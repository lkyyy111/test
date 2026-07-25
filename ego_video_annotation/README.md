# 第一视角具身操作视频自动标注

该项目完成细粒度视频切分、语言标注、手部姿态与轨迹提取。默认使用 MediaPipe 同时检测最多两只手，为每只手输出 21 个关键点、左右手轨迹和逐帧有效性 mask；当前版本不生成 MANO mesh。

## 当前细粒度切分策略

管线不使用画面变化作为动作边界，也不生成粗粒度窗口。当前顺序是：

1. 默认以约 8 FPS 读取真实时间戳，检测左右手 21 点。
   所有输入按非镜像第一人称视频处理：MediaPipe 的原始 handedness 会先交换左右手，再进入时序身份跟踪。`mediapipe_side` 保留模型原始标签，`detector_side` 是交换后的标签，`side` 是最终稳定身份。
2. 使用背景 LK 光流、RANSAC 仿射变换估计相邻采样帧的相机运动，并从掌心位移中扣除相机运动。
3. 对不超过 0.5 秒且前后身份连续的短丢手，仅补全相机补偿后的边界专用掌心轨迹；21 点与 `valid_mask` 仍保持缺失，不把插值当成真实观测。
4. 左右手分别计算并平滑掌心速度，并构造取当前活跃手最大归一化速度的全局活动曲线；在前后 0.75 秒加权窗口内寻找局部极小值。
5. 真实速度点权重为 1，插值速度点权重为 0.4；默认速度谷下降不低于 25%，prominence 不低于 0.10，窗口加权覆盖率不低于 0.60。左右手及全局候选在 0.4 秒内融合。
6. 候选边界先产生偏高召回率的临时细片段；最长分析窗默认 8 秒。每段仍使用最多 16 帧的固定预算：视频条件允许时，默认取片段前 0.75/0.25 秒各一帧、片段内部均匀 12 帧、片段后 0.25/0.75 秒各一帧。边界外帧仅帮助 VLM 判断动作方向，不改变片段边界，也不能作为补切帧。
7. 若 VLM 判断候选包含多个动作，它会返回动作序列和建议帧号；管线在建议时间附近对齐弱腕速谷，最多补切 2 个边界，并重新描述子片段。
8. 相邻片段若是同一操作手、同一动作和同一物体，则删除中间运动边界，合并为最终细粒度片段。连续切割、清洗、擦拭和搅拌不会按每次往返作为最终动作。
9. 对由多个候选合并且时长默认不低于 8 秒的最终细片段，重新在完整时间范围均匀取帧做一次整体 caption；成功才覆盖局部 caption，旧标注保留用于审计。

画面变化只保留为诊断字段和相机补偿的 fallback 判断，不参与动作边界打分。

## 手部缺失与有效视频

输出同时保留：

- `hand_present_raw`：采样帧确实检测到至少一只手；
- `hand_validity.left/right.observed`：对应手确实检测到；
- `smoothed_presence`：只桥接默认不超过 0.5 秒的短暂缺失；
- `interpolated_presence`：只代表活动状态被桥接，不会虚构 21 点；
- `valid_for_boundary`：该帧能否用于速度边界判断。
- `hand_tracks.left/right.boundary_palm_center`：真实或短缺失补全的边界专用掌心位置；
- `interpolated_for_motion`：该边界专用掌心是否来自低权重插值，`true` 时 `valid_mask` 仍为 `false`。

手部缺失不会被当成速度为零。双手长时间不可见时，管线会增加技术性隔离边界，将无手区间从可导出的动作片段中分开；这类边界不是语义动作边界，也不会参与相邻语义合并。

`valid_segments/` 只按最终细粒度边界导出满足以下条件的片段：

- 平滑手部覆盖率默认不低于 30%；
- 双手连续不可见默认不超过 1 秒；
- 至少有两个真实手部采样点；
- 片段时长默认不低于 0.5 秒。

## 环境

```bash
conda create -n egoanno python=3.10 -y
conda activate egoanno
pip install -r requirements.txt
```

Linux服务器生成中文字幕视频还需要中文字体；如果系统没有，可安装：

```bash
apt-get update
apt-get install -y fonts-noto-cjk
```

也可以使用 `--subtitle-font /path/to/font.ttf` 指定已有字体。

当前基础管线不依赖 PyTorch；MediaPipe 和 OpenCV 可在 CPU 上运行，外部 VLM API 不占用本地显存。

## 运行

```bash
python run_pipeline.py \
  --video ../data/long1.mp4 \
  --output outputs/long1 \
  --vlm-api-base "$VLM_API_BASE" \
  --vlm-model "$VLM_MODEL"
```

API 密钥从 `VLM_API_KEY` 环境变量读取。未配置 API 时仍会生成手部、轨迹、运动边界和固定回退 JSON，但不会执行 VLM 语义合并。

常用参数：

```bash
--sample-fps 8
--velocity-context-s 0.75
--velocity-drop-ratio 0.25
--velocity-prominence 0.10
--velocity-min-gap-s 0.60
--velocity-min-window-weight 0.60
--motion-interpolation-gap-s 0.5
--max-provisional-segment-s 8
--merged-recaption-min-duration-s 8
--fine-frame-count 16
--vlm-context-s 0.75
--hand-gap-tolerance 0.5
--max-no-hand-gap-s 1.0
--min-hand-coverage 0.30
```

先做不生成视频的烟雾测试：

```bash
python run_pipeline.py \
  --video ../data/short1.mp4 \
  --output outputs/short1_smoke \
  --vlm-api-base "$VLM_API_BASE" \
  --vlm-model "$VLM_MODEL" \
  --skip-video-outputs
```

### 同一次运行自动评测

标注模型与裁判模型分开设置。Qwen 继续生成标注，`gpt-5.4-mini` 通过 Responses API 只负责盲评最终结果；`JUDGE_API_KEY` 未单独设置时会复用 `VLM_API_KEY`。

```bash
export VLM_API_BASE="https://www.autodl.art/api/v1"
export VLM_MODEL="qwen3-vl-plus"
export JUDGE_MODEL="gpt-5.4-mini"

python run_pipeline.py \
  --video ../data/long1.mp4 \
  --output outputs/long1_eval \
  --vlm-api-base "$VLM_API_BASE" \
  --vlm-model "$VLM_MODEL" \
  --evaluate-vlm \
  --judge-model "$JUDGE_MODEL"
```

默认每项只评一次，适合快速实验。正式对比可添加 `--judge-repeats 3`，对每个片段和边界独立评测三次并逐项取中位数，但 API 调用量和耗时约变为三倍。裁判对片段沿用标注阶段相同的最多 16 帧协议（前文 2 + 片段内 12 + 后文 2），对相邻边界默认读取前后各 4 帧。

终端会直接打印四项 0–100 分：

- `SQ`（Segmentation Quality）：边界有效性、片段原子性和动作完整性的平均分；片段项按时长加权。
- `CF`（Caption Factuality）：`0.15×手别 + 0.35×动作 + 0.25×物体 + 0.25×方向/状态变化`。
- `TSC`（Temporal Semantic Consistency）：相邻片段在物体状态、动作方向和时间顺序上的连贯性。
- `EgoSegCap`：`0.4×SQ + 0.4×CF + 0.2×TSC`。

详细逐片段分数、逐边界分数、证据和失败调用写入 `vlm_evaluation.json`。这些是独立 VLM 裁判分，不等同于人工真值准确率；三种对比方法必须固定同一个裁判模型、prompt、帧采样、推理强度和重复次数。裁判调用失败不会删除已经完成的标注产物，而会记录 `N/A` 和错误原因。

## 输出

- `annotations.json`：面向训练/交付的紧凑细粒度标注。保留手部有效、VLM 成功且动作有意义的片段；`quality_status` 为 `accepted` 或 `review`，后者同时保留 `review_reasons`，不会再因需要复核而整段丢失。文件包含时间、视频路径、caption、动作、物体、左右手信息、质量分数及 `hand_landmarks.json` 采样范围。
- `annotations_diagnostics.json`：完整运行诊断，包括参数、候选与删除边界、全部有效/无效细片段、合并与重描述历史以及复核队列。
- `hand_landmarks.json`：每个采样时刻的左右手 21 点、有效性 mask、相机补偿质量和速度。
- `wrist_trajectories.csv`：每个采样时刻固定输出左右手各一行；缺失手的真实坐标为空，并另外记录边界专用掌心、`motion_source`、权重和插值 mask。
- `annotated_video.mp4`：保持原视频时间轴，根据 `annotations.json` 的 `[start_s, end_s)` 在对应区间显示中文caption；`review`片段使用“待复核”标记。当前不再生成骨架overlay视频。
- `valid_segments/`：按最终细粒度片段导出的有效操作视频。
- `vlm_evaluation.json`：仅在使用 `--evaluate-vlm` 时生成，包含 SQ、CF、TSC、EgoSegCap、各子项得分、逐片段/边界证据及评测覆盖率。

相邻片段使用半开时间区间 `[start_s, end_s)`，因此不会再因为最后一个采样时间而固定漏掉一段视频。
`annotations.json` 中的 `sample_range.end` 是包含式索引；使用 `--skip-video-outputs` 时不会生成片段视频，对应 `clip_path` 为 `null`。

## 坐标局限

MediaPipe 的 `x/y` 是图像归一化坐标，`z` 是相对深度；CSV 中 `world_x/y/z` 仍是 MediaPipe 的相对手部坐标，不是经过相机标定的米制世界坐标。当前速度使用背景仿射运动补偿后的二维掌心位移，适合本次轻量测试，但不等价于 VITRA 的完整世界坐标轨迹。

## 可选 Franka retarget（short1 / short2）

Franka retarget 是完全可选的第四阶段。默认不导入 MuJoCo，因此 long1 的安装方式、标注逻辑和输出保持不变。short1 使用整段右手腕轨迹；short2 使用左右手轨迹分别驱动两个独立的 Franka，并在同一个时间轴上生成并排回放。每只手都在自己的固定前向竖直平面中映射左右与抬放运动，再通过阻尼最小二乘 IK 和 Panda 位置执行器回放。双手模式保留时间同步，但不建模两手之间的米制距离、双臂碰撞或协同约束。

安装可选依赖：

```bash
pip install -r requirements-retarget.txt
```

`robot_descriptions` 会提供 MuJoCo Menagerie 的 Panda MJCF。也可以自行克隆 Menagerie 并设置：

```bash
export MUJOCO_MENAGERIE_PATH="$HOME/mujoco_menagerie"
```

无显示器的 Linux 服务器建议在启动 Python 前启用 EGL：

```bash
export MUJOCO_GL=egl
```

short1 完整运行：

```bash
python run_pipeline.py \
  --video ../data/short1.mp4 \
  --output outputs/short1_franka \
  --vlm-api-base "$VLM_API_BASE" \
  --vlm-model "$VLM_MODEL" \
  --retarget-franka \
  --retarget-hand right
```

short2 双手完整运行：

```bash
python run_pipeline.py \
  --video ../data/short2.mp4 \
  --output outputs/short2_franka \
  --vlm-api-base "$VLM_API_BASE" \
  --vlm-model "$VLM_MODEL" \
  --retarget-franka \
  --retarget-hand both
```

如果只想验证 IK 和 CSV，不生成任何 MP4，可额外添加 `--skip-video-outputs`。retarget 结果写入 `franka_retarget/`：

- `target_trajectory.csv`：由右手腕图像轨迹映射得到的末端目标。
- `joint_trajectory.csv`：仿真后的 Franka 七关节轨迹。
- `achieved_trajectory.csv`：实际末端轨迹和逐时刻误差。
- `retarget_metrics.json`：轨迹覆盖率、IK 与动力学跟踪误差。
- `retarget.mp4`：Franka 回放；蓝色小球是当前目标点。

双手模式在 `franka_retarget/left/` 和 `franka_retarget/right/` 下分别生成以上文件，并额外输出：

- `retarget_summary.json`：左右手覆盖率、IK/仿真误差和双手模式说明。
- `retarget_both.mp4`：左右两个独立 Franka 的同步并排回放。

long1 继续使用原命令，不添加 `--retarget-franka`。也不要根据文件名或时长自动启用 retarget；该开关由调用者显式控制。
