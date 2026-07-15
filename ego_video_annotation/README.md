# 第一视角具身操作视频自动标注

该项目完成多层级视频切分、语言标注、手部姿态与轨迹提取。默认使用 MediaPipe 同时检测最多两只手，为每只手输出 21 个关键点、左右手轨迹和逐帧有效性 mask；当前版本不生成 MANO mesh。

## 当前切分策略：先细后粗

管线不再用画面变化作为动作边界，也不再先生成粗粒度窗口。当前顺序是：

1. 默认以约 8 FPS 读取真实时间戳，检测左右手 21 点。
2. 使用背景 LK 光流、RANSAC 仿射变换估计相邻采样帧的相机运动，并从掌心位移中扣除相机运动。
3. 左右手分别计算并平滑掌心速度；在前后 0.75 秒参考窗口内寻找显著局部极小值。
4. 候选速度谷默认要求前后相对下降均不低于 40%，归一化 prominence 不低于 0.20，且前后确实存在运动。
5. 左右手在 0.4 秒内的候选融合为一个边界；单手候选也可保留。
6. 候选边界先产生较高召回率的临时细片段。每段最多均匀取 16 帧交给 VLM，获得固定 JSON caption。
7. 相邻片段若是同一操作手、同一动作和同一物体，则删除中间运动边界，合并为最终细粒度片段。连续切割、清洗、擦拭和搅拌不会按每次往返作为最终动作。
8. VLM最后只根据按时间排列的最终细粒度 JSON，将连续 fine id 归纳为粗粒度任务。

画面变化只保留为诊断字段和相机补偿的 fallback 判断，不参与动作边界打分。

## 手部缺失与有效视频

输出同时保留：

- `hand_present_raw`：采样帧确实检测到至少一只手；
- `hand_validity.left/right.observed`：对应手确实检测到；
- `smoothed_presence`：只桥接默认不超过 0.5 秒的短暂缺失；
- `interpolated_presence`：只代表活动状态被桥接，不会虚构 21 点；
- `valid_for_boundary`：该帧能否用于速度边界判断。

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
--velocity-drop-ratio 0.40
--velocity-prominence 0.20
--fine-frame-count 16
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

## 输出

- `annotations.json`：速度候选、被语义合并删除的边界、最终粗/细粒度片段、有效片段和复核队列。
- `hand_landmarks.json`：每个采样时刻的左右手 21 点、有效性 mask、相机补偿质量和速度。
- `wrist_trajectories.csv`：每个采样时刻固定输出左右手各一行；缺失手的 `valid_mask=false`、坐标为空，不会伪造轨迹点。
- `hand_overlay.mp4`：采样帧的手部骨架叠加。
- `valid_segments/`：按最终细粒度片段导出的有效操作视频。

相邻片段使用半开时间区间 `[start_s, end_s)`，因此不会再因为最后一个采样时间而固定漏掉一段视频。

## 坐标局限

MediaPipe 的 `x/y` 是图像归一化坐标，`z` 是相对深度；CSV 中 `world_x/y/z` 仍是 MediaPipe 的相对手部坐标，不是经过相机标定的米制世界坐标。当前速度使用背景仿射运动补偿后的二维掌心位移，适合本次轻量测试，但不等价于 VITRA 的完整世界坐标轨迹。
