# Computer Vision Pipeline: Object Detection, Tracking & Lane Detection

A production-ready **Kedro + MLFlow** pipeline for implementing and evaluating state-of-the-art deep learning models for:
- **Object Detection** (RF-DETR, RT-DETR, YOLOv11/12)
- **Multi-Object Tracking** (ByteTrack, BoT-SORT, DeepSORT)
- **Lane Detection** (CLRNet, LaneATT)
- **Path/Trajectory Construction** (Extended module for path planning)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CV Pipeline Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │   Camera     │───>│   Data       │───>│   Object     │                 │
│   │   Input      │    │   Pipeline   │    │   Detection  │                 │
│   └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                   │                         │
│                       ┌───────────────────────────┼───────────────────────┐ │
│                       │                           ▼                       │ │
│                       │                    ┌──────────────┐               │ │
│                       │                    │   Multi-Obj  │               │ │
│   ┌──────────────┐    │                    │   Tracking   │               │ │
│   │   Lane       │<───┤                    └──────┬───────┘               │ │
│   │   Detection  │    │                           │                       │ │
│   └──────┬───────┘    │                           ▼                       │ │
│          │            │                    ┌──────────────┐               │ │
│          │            │                    │   Fusion &   │               │ │
│          ▼            │                    │   Path Plan  │               │ │
│   ┌──────────────┐    │                    └──────────────┘               │ │
│   │   Path/Lane  │    │                                                   │ │
│   │   Construct  │    └───────────────────────────────────────────────────┘ │
│   └──────────────┘                                                          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         MLFlow Tracking                              │   │
│   │  • Experiments  • Model Registry  • Metrics  • Artifacts            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
cv_pipeline/
├── conf/
│   ├── base/
│   │   ├── catalog.yml          # Data catalog definitions
│   │   ├── parameters/
│   │   │   ├── detection.yml    # Detection model configs
│   │   │   ├── tracking.yml     # Tracking configs
│   │   │   ├── lane.yml         # Lane detection configs
│   │   │   └── training.yml     # Training hyperparameters
│   │   └── logging.yml          # Logging configuration
│   └── local/
│       └── credentials.yml      # Local credentials
├── data/
│   ├── 01_raw/                  # Raw camera images/videos
│   ├── 02_intermediate/         # Preprocessed data
│   ├── 03_primary/              # Detection outputs
│   ├── 04_feature/              # Tracking features
│   ├── 05_model_input/          # Model-ready data
│   ├── 06_models/               # Trained models
│   ├── 07_model_output/         # Inference outputs
│   └── 08_reporting/            # Visualizations & reports
├── src/
│   └── cv_pipeline/
│       ├── __init__.py
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── data_processing/     # Data ingestion & preprocessing
│       │   ├── object_detection/    # Detection models
│       │   ├── tracking/            # MOT algorithms
│       │   ├── lane_detection/      # Lane detection models
│       │   └── path_construction/   # Path/trajectory planning
│       ├── models/
│       │   ├── __init__.py
│       │   ├── detectors/           # Detection model implementations
│       │   ├── trackers/            # Tracking implementations
│       │   └── lane/                # Lane detection implementations
│       └── utils/
│           ├── __init__.py
│           ├── visualization.py
│           ├── metrics.py
│           └── mlflow_utils.py
├── notebooks/                       # Jupyter notebooks for exploration
├── tests/                           # Unit tests
├── mlruns/                          # MLFlow tracking directory
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Supported Models

### Object Detection
| Model | mAP (COCO) | FPS (T4) | NMS-Free | Notes |
|-------|------------|----------|----------|-------|
| **RF-DETR-M** | 54.7% | 220 | ✓ | SOTA 2025, best accuracy/speed |
| **RT-DETR-R50** | 53.1% | 108 | ✓ | Production-ready transformer |
| **YOLOv12-M** | 52.5% | 180 | Partial | Area Attention, FlashAttention |
| **YOLOv11-M** | 51.5% | 200 | ✗ | Well-tested, community support |

### Multi-Object Tracking
| Tracker | MOTA | IDF1 | FPS | Re-ID | Notes |
|---------|------|------|-----|-------|-------|
| **BoT-SORT** | 80.5 | 80.2 | 30 | ✓ | Best accuracy, CMC support |
| **ByteTrack** | 80.3 | 77.3 | 60 | ✗ | Fast, low confidence matching |
| **OC-SORT** | 78.0 | 77.5 | 55 | ✗ | Motion-based, robust |
| **DeepSORT** | 75.4 | 77.2 | 25 | ✓ | Classic, good for long-term |

### Lane Detection
| Model | F1 (CULane) | FPS | Notes |
|-------|-------------|-----|-------|
| **CLRerNet** | 81.5% | 90 | SOTA with LaneIoU |
| **CLRNet** | 80.5% | 100 | Cross-layer refinement |
| **LaneATT** | 77.0% | 250 | Fastest, attention-based |
| **UFLD** | 72.3% | 300 | Ultra-fast, row-based |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/cv-pipeline.git
cd cv-pipeline

# Create conda environment
conda create -n cv_pipeline python=3.10
conda activate cv_pipeline

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

## Quick Start

### 1. Run the Full Pipeline

```bash
# Run all pipelines
kedro run

# Run specific pipeline
kedro run --pipeline=object_detection
kedro run --pipeline=tracking
kedro run --pipeline=lane_detection
kedro run --pipeline=path_construction
```

### 2. Start MLFlow UI

```bash
mlflow ui --port 5000
# Visit http://localhost:5000
```

### 3. Run with Specific Model

```bash
# Use RT-DETR for detection
kedro run --pipeline=object_detection --params="detection.model=rt_detr"

# Use ByteTrack for tracking
kedro run --pipeline=tracking --params="tracking.tracker=bytetrack"
```

### 4. Visualize Results

```bash
kedro viz  # Interactive pipeline visualization
```

## Configuration

### Detection Configuration (`conf/base/parameters/detection.yml`)

```yaml
detection:
  model: "rf_detr"  # rf_detr, rt_detr, yolov12, yolov11
  variant: "medium"  # nano, small, medium, large, xlarge
  confidence_threshold: 0.25
  iou_threshold: 0.45
  classes: null  # null for all classes, or [0, 1, 2] for specific
  input_size: [640, 640]
  device: "cuda:0"
  half_precision: true
```

### Tracking Configuration (`conf/base/parameters/tracking.yml`)

```yaml
tracking:
  tracker: "botsort"  # botsort, bytetrack, ocsort, deepsort
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  track_buffer: 30
  match_thresh: 0.8
  # BoT-SORT specific
  cmc_method: "sparseOptFlow"  # Camera Motion Compensation
  proximity_thresh: 0.5
  appearance_thresh: 0.25
  # Re-ID settings
  with_reid: true
  reid_model: "osnet_x0_25"
```

### Lane Detection Configuration (`conf/base/parameters/lane.yml`)

```yaml
lane:
  model: "clrernet"  # clrernet, clrnet, laneatt, ufld
  backbone: "resnet34"  # resnet18, resnet34, resnet50, dla34
  num_lanes: 4
  conf_threshold: 0.4
  nms_threshold: 50
  img_h: 320
  img_w: 800
  # Path construction
  path_construction:
    enable: true
    smoothing_method: "bezier"  # bezier, spline, polynomial
    lookahead_distance: 30  # meters
```

## Pipeline Details

### Data Processing Pipeline

```python
# Handles video/image ingestion, augmentation, and preprocessing
pipelines:
  - node: load_camera_data
  - node: preprocess_frames
  - node: apply_augmentations
  - node: prepare_batches
```

### Object Detection Pipeline

```python
# Multi-model detection with automatic model selection
pipelines:
  - node: load_detection_model
  - node: run_inference
  - node: post_process_detections
  - node: log_detection_metrics
```

### Tracking Pipeline

```python
# Integrates detections with MOT algorithms
pipelines:
  - node: initialize_tracker
  - node: associate_detections
  - node: update_tracks
  - node: handle_lost_tracks
  - node: extract_trajectories
```

### Lane Detection Pipeline

```python
# Lane detection and path construction
pipelines:
  - node: load_lane_model
  - node: detect_lanes
  - node: fit_lane_curves
  - node: construct_drivable_path
  - node: validate_lane_consistency
```

## MLFlow Integration

All experiments are tracked with MLFlow:

```python
import mlflow

# Automatic logging with Kedro-MLFlow
mlflow.autolog()

# Custom metrics
with mlflow.start_run(run_name="detection_experiment"):
    mlflow.log_param("model", "rf_detr")
    mlflow.log_metric("mAP@0.5", 0.547)
    mlflow.log_metric("fps", 220)
    mlflow.log_artifact("model.pt")
```

### Tracked Metrics

- **Detection**: mAP@0.5, mAP@0.5:0.95, precision, recall, FPS
- **Tracking**: MOTA, IDF1, HOTA, ID switches, track fragmentation
- **Lane Detection**: F1 score, precision, recall, lane accuracy
- **System**: GPU memory, inference latency, throughput

## Extended: Path Construction Module

The path construction module fuses detection, tracking, and lane information:

```python
# Path construction integrates multiple sources
path_constructor = PathConstructor(
    lane_weight=0.6,
    object_weight=0.3,
    trajectory_weight=0.1,
    smoothing="bezier",
    safety_margin=1.5  # meters
)

# Generate drivable path
path = path_constructor.construct(
    lanes=lane_detections,
    objects=tracked_objects,
    ego_position=current_pose
)
```

## Testing & Evaluation

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=cv_pipeline tests/

# Benchmark models
python -m cv_pipeline.benchmark --models all --dataset coco
```

## Datasets

The pipeline supports standard autonomous driving datasets:

- **COCO**: Object detection training/evaluation
- **MOT17/MOT20**: Multi-object tracking benchmarks
- **CULane**: Lane detection (98K+ images)
- **TuSimple**: Lane detection with trajectories
- **BDD100K**: Multi-task driving dataset
- **nuScenes**: 3D detection and tracking

## License

Apache 2.0

## References

- RF-DETR: [Roboflow Blog](https://blog.roboflow.com/rf-detr/)
- RT-DETR: [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- ByteTrack: [ECCV 2022](https://github.com/ifzhang/ByteTrack)
- BoT-SORT: [arXiv:2206.14651](https://arxiv.org/abs/2206.14651)
- CLRNet: [CVPR 2022](https://github.com/Turoad/CLRNet)
- CLRerNet: [WACV 2024](https://github.com/hirotomusiker/CLRerNet)
