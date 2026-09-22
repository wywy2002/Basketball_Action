# Basketball Pose Baseline

篮球比赛视频的二维人体姿态基线工程。当前主流程使用 SportsMOT 官方 YOLOX-X 球员检测器、RTMPose-X Body7 姿态模型和遮挡感知跟踪；三维重建留到二维标注稳定后再接入。

## 当前范围

- 统一 COCO-17 二维姿态 JSON 格式；
- 按 `video_id / shot_id / track_id` 组装连续轨迹；
- 对短缺帧做线性插值，保留低置信度信息；
- COCO-17 转换为 MotionBERT 常用的 H36M-17 顺序；
- 检查低置信度、关键点越框、轨迹跳变和重复帧；
- 提供 MMPose 官方统一推理接口的轻量封装。

当前版本不负责检测器训练、ReID、镜头切分或世界坐标恢复。

## 篮球视频中的两类干扰

实际推理入口不会直接保留画面中的所有人体：

1. 观众、替补和裁判：ROI 只作为空间先验；场外检测也参与跟踪，只有具有连续场内历史的边缘轨迹会恢复。单帧和无场内证据的外部轨迹保留在 `detections_audit.json`，不会静默删除。
2. 球员重叠：每个 SportsMOT 人体框单独运行 top-down RTMPose-X，并用人体框 IoU、关键点形状和球衣外观联合维持 `track_id`。短暂低置信检测会保留；无法确认的遮挡写入 `occlusion_events`，不会伪造骨架。
3. 阵营识别：从肩部和髋部围成的躯干区域提取 HSV 色度特征，按轨迹聚合，并在每个镜头内聚为两队；证据不足时输出 `unknown`。

球场 ROI 和颜色阈值是视频域配置，不是通用真值。`configs/video_1.json` 至 `configs/video_4.json` 分别对应四个样例视频，换比赛或镜头后必须重新检查。

## 环境

OpenMMLab 建议使用独立的 Python 3.10 或 3.11 环境。当前机器的系统 Python 3.13 仅用于运行本项目的无依赖单元测试，不建议直接安装 MMPose/MMCV。

```powershell
conda create -n basketball-pose python=3.11 -y
conda activate basketball-pose
cd D:\Basketball_Action
python -m pip install -e .
```

安装 MMPose、MMCV、MMDetection 和 PyTorch 时，应按显卡与 CUDA 版本使用 OpenMMLab 官方安装说明；这些依赖与预训练权重不包含在本仓库中。

当前 Windows/RTX 5060 实测采用轻量 ONNX 路线：

```powershell
python -m pip install numpy opencv-contrib-python tqdm "onnxruntime-gpu[cuda,cudnn]==1.23.2"
python -m pip install rtmlib==0.0.16 --no-deps
```

`rtmlib` 的默认依赖会额外安装 CPU 版 `onnxruntime`，它可能覆盖 GPU Provider，因此这里显式使用 `--no-deps`。

## 数据格式

输入 JSON 是记录数组，或包含 `records` 的对象。每条记录表示一帧中的一名球员：

```json
{
  "video_id": "video_1",
  "shot_id": "shot_0001",
  "frame_id": 12,
  "timestamp": 0.4,
  "track_id": "7",
  "image_size": [1920, 1080],
  "bbox_xyxy": [100.0, 200.0, 300.0, 700.0],
  "keypoints_2d": [[120.0, 230.0, 0.95]],
  "pose_schema": "coco17"
}
```

`keypoints_2d` 必须完整包含 17 个 `[x, y, confidence]`。坐标始终是原视频帧坐标，不是球员 crop 坐标。

## 命令

```powershell
# 验证并输出质量报告
$env:PYTHONPATH = "D:\Basketball_Action\src"
python -m basketball_pose.cli validate examples\sample_pose2d.json

# 分轨、短缺帧插值，并生成 H36M-17 输入
python -m basketball_pose.cli preprocess examples\sample_pose2d.json outputs\prepared.json --max-gap 2

# 查看真实模型入口（执行时才会要求 MMPose，并可能按其配置下载权重）
python -m basketball_pose.cli infer input_videos\video_1.mp4 outputs\video_1 --task 2d --device cuda:0
python -m basketball_pose.cli infer input_videos\video_1.mp4 outputs\video_1_3d --task 3d --device cuda:0

# 篮球感知 2D 管线：场地过滤、裁判候选过滤、遮挡跟踪
python -m basketball_pose.cli run-video input_videos\video_1.mp4 outputs\video_1_rtmo `
  --config configs\video_1.json --model rtmo --device cuda

# 推荐主流程：SportsMOT 球员检测 + RTMPose-X Body7
python -m basketball_pose.cli run-video input_videos\video_1.mp4 output\video_1 `
  --config configs\video_1.json --model sportsmot-rtmpose --device cuda
```

## 测试

```powershell
$env:PYTHONPATH = "D:\Basketball_Action\src"
python -m pytest tests -q
```

官方接口依据：MMPose `MMPoseInferencer('human')` 和 `MMPoseInferencer(pose3d='human3d')`。
