# 第一视角具身操作视频自动标注

该项目对应考核的前三项：多层级视频切分、语言标注、手部姿态和轨迹提取。默认使用 MediaPipe 提取 21 个手部关键点与归一化三维坐标；手部 mesh 与物体 mesh 是后续可选模块。

## 切分策略

粗粒度不是简单按镜头变化切分。管线以 8 秒为上下文窗口、4 秒为中心锚点步长，每个窗口向 VLM 发送均匀抽取的 5 帧；标签只描述窗口中心时刻的稳定子任务和受限任务阶段（准备、取放、清洗、切配、烹饪、整理、移动、等待、其他）。相邻中心锚点的语义变化会给出一个 4 秒候选区间；该区间再以 1 秒级多帧 VLM 复核，要求新阶段连续出现后才确认边界，最后向附近的手部停顿或明显视觉变化微调。若 VLM 不可用或语义窗口响应成功率低于 60%，才退化为画面变化候选切分，并将该状态写入结果和复核队列。

细粒度切分仅在每个粗粒度子任务内部进行，依据局部手腕运动与画面变化生成动作级候选边界；每个细片段都带有 `parent_coarse_id`。这样可避免厨房第一视角视频中“场景没变、任务已变”或“相机晃动、任务未变”造成的错误层级。

## 服务器环境

建议在 Ubuntu 22.04 + NVIDIA RTX 4090（24 GB）上运行。当前基础管线不强依赖 PyTorch，便于先稳定产出；后续接入 HaMeR、SAM 2 时再在同一个环境安装 PyTorch/CUDA。

```bash
conda create -n egoanno python=3.10 -y
conda activate egoanno
pip install -r requirements.txt
```

## 下载考核数据

在仓库根目录执行；默认下载 PDF 中给出的 `ly985211/egodata` 数据集到 `data/egodata`。

```bash
python scripts/download_egodata.py
```

如下载速度不稳定，可在服务器上配置 ModelScope 镜像或重复运行该命令，下载器会复用已完成的文件。

## 运行

```bash
python run_pipeline.py --video /path/to/long1.mp4 --output outputs/long1
```

如需让视觉语言模型生成中文描述，设置 API 密钥并提供兼容 Chat Completions 的接口：

```bash
export VLM_API_KEY='...'
python run_pipeline.py --video /path/to/long1.mp4 --output outputs/long1 \
  --vlm-api-base https://your-api.example/v1 --vlm-model your-vlm-model
```

默认不调用 API；此时会输出固定模式的回退 JSON，方便验证其它结果。调用 API 时，每个细片段会向 VLM 提供按时间顺序排列的起始、中间、结束三帧，并要求只输出固定 JSON。低于 `--review-confidence`（默认 0.65）、包含未知动作/物体或 API 异常的片段，都会写入 `annotations.json` 的 `review_queue`，便于人工复核。

第一视角视频中手会因遮挡或移出视野而短暂消失。管线会保留每帧真实的 `hand_present_raw`，并仅对前后均检测到手的短缺口（默认不超过 1.5 秒）生成 `hand_present_smoothed`。手的出现/消失本身不会作为语义切分边界；有效片段以平滑后的手部覆盖率筛选，同时保留原始覆盖率与桥接比例，避免把短暂出画误当作操作结束。

## 输出

* `annotations.json`：粗/细粒度分段、有效片段、固定 JSON 语言标注、低置信度复核队列、质量和方法元数据。
* `hand_landmarks.json`：逐采样帧的 2D/归一化 3D 手部关键点与手腕轨迹。
* `wrist_trajectories.csv`：便于后续 retarget 的左右手腕轨迹。
* `hand_overlay.mp4`：手部骨架叠加可视化。
* `valid_segments/`：保留的有效操作片段。

## 坐标说明与局限

MediaPipe 的 `x, y` 是相对图像宽高归一化坐标；`z` 是相对深度而非相机标定后的绝对深度。它足以描述视频内手部姿态和轨迹。若需要精确米制轨迹或 MANO mesh，下一阶段会用 HaMeR 与相机/尺度对齐模块替换或补充本模块。
