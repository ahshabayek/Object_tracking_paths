# CV Pipeline Implementation Explanations

This document provides detailed explanations of the design decisions and implementation choices made in this project.

---

## Custom Datasets

The 5 custom Kedro datasets were designed to handle the specific I/O needs of a CV pipeline for autonomous driving.

### 1. VideoDataSet

**Purpose**: Load video files as a list of frames for batch processing

```yaml
raw_video:
  type: cv_pipeline.datasets.VideoDataSet
  filepath: data/01_raw/video.mp4
  load_args:
    target_fps: 30      # Downsample 60fps video to 30fps
    resize: [1920, 1080]
    max_frames: 1000
    start_frame: 100
```

**Why needed**: Standard Kedro doesn't have a video loader. This uses `cv2.VideoCapture` and provides:
- FPS control (skip frames to match target FPS)
- Frame range selection (start/end frame)
- Resize on load (reduce memory for large videos)
- Grayscale conversion option

**Key implementation details**:
- Returns `List[np.ndarray]` where each frame is (H, W, C) in BGR format
- Frame skipping calculated as `video_fps / target_fps` to maintain temporal consistency
- Memory-efficient: frames loaded sequentially, not all at once into memory during read

---

### 2. TensorDataSet

**Purpose**: Save/load PyTorch tensors (batched frames, feature maps, embeddings)

```yaml
detection_batch:
  type: cv_pipeline.datasets.TensorDataSet
  filepath: data/05_model_input/detection_batch.pt
  save_args:
    compress: true
```

**Why needed**: Kedro's PickleDataSet works but `torch.save/load` is optimized for tensors:
- Handles GPU tensors properly
- `map_location` for device mapping (load GPU tensor to CPU)
- Optional compression for large tensors
- `weights_only=True` for security (prevents arbitrary code execution)

**Use cases**:
- Preprocessed frame batches ready for model inference
- Extracted Re-ID feature embeddings
- Intermediate feature maps for debugging

---

### 3. PyTorchModelDataSet

**Purpose**: Serialize detection/tracking/lane models with versioning

```yaml
detection_model:
  type: cv_pipeline.datasets.PyTorchModelDataSet
  filepath: data/06_models/detection_model.pt
  versioned: true
  save_args:
    save_state_dict_only: true
    map_location: cpu
```

**Why needed**: Models need special handling:
- **Full model save**: `torch.save(model)` - includes architecture, useful for inference
- **State dict only**: Just weights - smaller, requires model class to reload
- Supports checkpoint format (`{"state_dict": ..., "epoch": ...}`)
- Works with Kedro versioning for experiment tracking

**Loading modes**:
1. Full model: Direct load, no class needed
2. State dict with class: Requires `model_class` and `model_args` in `load_args`
3. Checkpoint dict: Automatically detects `state_dict` or `model_state_dict` keys

---

### 4. CameraStreamDataSet

**Purpose**: Real-time inference from webcam or RTSP streams

```yaml
raw_camera_stream:
  type: cv_pipeline.datasets.CameraStreamDataSet
  load_args:
    source: "rtsp://192.168.1.100:554/stream"  # Or 0 for webcam
    fps: 30
    duration: 10.0      # Capture 10 seconds
    buffer_size: 10     # Async frame buffer
    threaded: true
```

**Why needed**: Live camera input for real-time autonomous driving:
- Supports camera index (0, 1) or RTSP URLs
- Credential injection for authenticated streams
- Threaded buffering to prevent frame drops
- Duration/max_frames limits for controlled capture

**Threading model**:
- Background thread continuously captures frames into a queue
- Main thread consumes frames at its own pace
- Buffer overflow handled by dropping oldest frames (real-time priority)

---

### 5. VideoWriterDataSet

**Purpose**: Write annotated output videos with detection/tracking overlays

```yaml
annotated_video:
  type: cv_pipeline.datasets.VideoWriterDataSet
  filepath: data/08_reporting/annotated_output.mp4
  save_args:
    fps: 30
    codec: mp4v
    resize: [1920, 1080]
```

**Why needed**: Export visualized results:
- Configurable codec (mp4v, h264, xvid, mjpg)
- FPS control independent of source
- Resize output frames
- Handles grayscale to BGR conversion automatically

**Supported codecs**:
| Codec | Container | Quality | Speed |
|-------|-----------|---------|-------|
| mp4v | .mp4 | Good | Fast |
| avc1/h264 | .mp4 | Best | Slow |
| xvid | .avi | Good | Fast |
| mjpg | .avi | Large | Fastest |

---

### Data Flow Diagram

```
VideoDataSet ──► frames ──► detection ──► TensorDataSet (features)
     │                          │
     │                          ▼
     │               PyTorchModelDataSet (model weights)
     │                          │
     ▼                          ▼
CameraStreamDataSet ──► live frames ──► tracking ──► VideoWriterDataSet
                                                         (annotated output)
```

---

## Visualization Utilities

### Design Philosophy

The visualization module follows these principles:
1. **In-place modification**: Functions modify images directly for efficiency
2. **Consistent interface**: All functions accept dicts or dataclass objects
3. **Configurable appearance**: Colors, thickness, transparency all adjustable
4. **Stateful option**: `Visualizer` class maintains trajectory history across frames

### Color Scheme

| Object Type | Color (BGR) | Rationale |
|-------------|-------------|-----------|
| Person | Red (0,0,255) | High visibility, safety-critical |
| Car | Blue (255,0,0) | Common, distinct from person |
| Truck | Purple (128,0,128) | Similar to car but distinguishable |
| Ego lanes | Green (0,255,0) | Safe driving zone |
| Adjacent lanes | Orange (0,165,255) | Caution, potential lane change |
| Path | Green (0,255,0) | Planned trajectory |

---

## Re-ID Feature Extractors

### Why Multiple Extractors?

| Extractor | Parameters | Speed | Accuracy | Use Case |
|-----------|------------|-------|----------|----------|
| OSNet x1.0 | 2.2M | Medium | Best | Production tracking |
| OSNet x0.25 | 0.2M | Fast | Good | Edge devices |
| FastReID | 25M+ | Slow | Best | Offline analysis |
| SimpleReID | 0.5M | Fastest | Basic | Development/testing |

### Feature Extraction Pipeline

```
Image ──► Crop bbox ──► Resize (256x128) ──► Normalize ──► CNN ──► L2 Normalize ──► Feature [512-dim]
                              │
                              ▼
                    ImageNet mean/std normalization
                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
```

### Distance Metrics

- **Cosine distance**: `1 - dot(a, b)` - Preferred for normalized features
- **Euclidean distance**: `||a - b||` - Better for non-normalized features

---

## Camera Motion Compensation (CMC)

### Algorithm Comparison

| Method | Accuracy | Speed | Best For |
|--------|----------|-------|----------|
| ECC | Highest | Slowest | Offline processing |
| SIFT | High | Slow | Textured scenes |
| ORB | Medium | Fast | Real-time |
| OpticalFlow | Medium | Fast | Smooth motion |
| SparseFlow | Medium | Fastest | General use |

### When to Use CMC

CMC is critical when:
- Camera is mounted on a moving vehicle
- Tracking objects across frames with camera pan/tilt
- Kalman filter predictions need to account for ego-motion

CMC is unnecessary when:
- Camera is stationary (surveillance)
- Very short tracking windows
- Objects move much faster than camera

### Transformation Pipeline

```
Frame N-1 ──► Compute warp matrix ──► Apply to predicted boxes ──► Match with detections
    │                                        │
    ▼                                        ▼
Frame N ─────────────────────────────► Detections
```

---

## Evaluation Metrics

### Detection Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| mAP@50 | Mean AP at IoU=0.5 | 0-1 | Standard PASCAL VOC metric |
| mAP@75 | Mean AP at IoU=0.75 | 0-1 | Stricter localization |
| mAP@50:95 | Mean over IoU 0.5:0.95 | 0-1 | COCO primary metric |
| Precision | TP/(TP+FP) | 0-1 | False positive rate |
| Recall | TP/(TP+FN) | 0-1 | Miss rate |

### Tracking Metrics (MOT Challenge)

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| MOTA | 1 - (FN+FP+IDS)/GT | -∞ to 1 | Overall tracking accuracy |
| MOTP | Σ IoU / TP | 0-1 | Localization precision |
| IDF1 | 2·IDTP/(2·IDTP+IDFP+IDFN) | 0-1 | Identity preservation |
| ID Switches | Count | 0-∞ | Track ID changes |

### Lane Detection Metrics

- **Accuracy**: TP / (TP + FP + FN)
- **F1**: Harmonic mean of precision and recall
- **IoU-based**: Lanes rendered as thick lines, then mask IoU computed

---

## Object Detection Models

### Model Selection Philosophy

We support multiple detection architectures to balance different trade-offs:

| Priority | Model | Use Case |
|----------|-------|----------|
| Accuracy | D-FINE | When accuracy matters most, offline processing |
| Speed | YOLOv12, RT-DETR | Real-time applications |
| Flexibility | YOLO-World, GroundingDINO | Unknown classes, open-vocabulary |
| Commercial | D-FINE, RT-DETR | Apache 2.0 license projects |

### Model Categories

#### 1. Closed-Vocabulary Detectors (Fixed Classes)

These models detect a fixed set of classes (typically COCO 80 classes).

**YOLO Family (YOLOv10, YOLOv11, YOLOv12)**
- CNN-based, single-stage detectors
- YOLOv12 adds "Area Attention" for better accuracy
- Best for: Real-time applications, edge deployment
- License: AGPL-3.0 (restrictive for commercial)

**RT-DETR / RT-DETRv2 (CVPR 2024)**
- First transformer to beat YOLO in real-time
- Hybrid encoder: CNN + Transformer
- Flexible speed tuning via decoder layers
- License: Apache 2.0 (commercial-friendly)

**D-FINE (ICLR 2025 Spotlight)**
- Fine-grained Distribution Refinement
- Best accuracy: 57.4% mAP on COCO
- Better localization than YOLO
- License: Apache 2.0 (commercial-friendly)

**RF-DETR (Roboflow)**
- Uses DINOv2 vision foundation model as backbone
- First to break 60% mAP in real-time
- Best accuracy/speed trade-off

#### 2. Open-Vocabulary Detectors (Any Class)

These models can detect objects not seen during training.

**YOLO-World (Tencent AI Lab, 2024)**
- Zero-shot detection via text prompts
- Real-time performance
- Use case: Detect custom objects without retraining

```python
# Example: Detect custom objects
params = {
    "model": "yolo_world",
    "text_prompts": ["red sports car", "pedestrian with umbrella", "delivery truck"]
}
```

**GroundingDINO (IDEA Research)**
- Language-grounded detection
- Natural language prompts
- Higher accuracy but slower
- Use case: Complex scene understanding

```python
# Example: Natural language detection
params = {
    "model": "grounding_dino",
    "text_prompt": "person wearing a red jacket . car parked on the left . traffic light ."
}
```

### Why We Chose These Models

| Model | Why Included | Research Basis |
|-------|--------------|----------------|
| RT-DETR | CVPR 2024 best paper candidate, proves transformers beat CNNs | [Paper](https://arxiv.org/abs/2304.08069) |
| D-FINE | ICLR 2025 Spotlight, SOTA accuracy, Apache license | [Paper](https://arxiv.org/abs/2410.13842) |
| YOLO-World | Zero-shot capability, no retraining needed | [Paper](https://arxiv.org/abs/2401.17270) |
| GroundingDINO | Natural language understanding, 63% AP fine-tuned | [Paper](https://arxiv.org/abs/2303.05499) |
| YOLOv12 | Latest YOLO, Area Attention innovation | Ultralytics 2025 |

### Detection Output Format

All detectors output standardized `Detection` objects:

```python
@dataclass
class Detection:
    bbox: np.ndarray      # [x1, y1, x2, y2] in pixels
    confidence: float     # 0.0 to 1.0
    class_id: int         # COCO class ID, or -1 for open-vocab
    class_name: str       # "person", "car", or custom prompt
    phrase: str = None    # For GroundingDINO text matches
    features: np.ndarray = None  # For Re-ID embeddings
```

---

## 2D Tracking vs 3D BEV Perception

### Understanding the Difference

This is a critical architectural decision for autonomous driving systems.

#### Current Implementation: 2D Detection + Tracking

```
Single Camera → 2D Bounding Boxes → Track in Pixel Space
                      ↓
              [x1, y1, x2, y2] in image coordinates
```

**What we get:**
- Object locations in the image (pixels)
- Track IDs across frames
- Trajectories in image space

**What we DON'T get:**
- Real-world distance (meters)
- 3D position and orientation
- 360° awareness

#### Alternative: BEV (Bird's Eye View) Perception

```
Multiple Cameras (6) → 3D Understanding → Top-Down World View
         ↓                                       ↓
   Surround view                    [x, y, z] in meters from vehicle
```

**What we get:**
- 3D object positions in world coordinates
- Exact distances in meters
- Object dimensions (width, height, length)
- Heading/orientation angles
- 360° coverage

### Detailed Comparison

| Aspect | 2D Tracking | BEV Perception |
|--------|-------------|----------------|
| **Output coordinates** | Pixels [x1, y1, x2, y2] | Meters [x, y, z, w, h, l, yaw] |
| **Depth information** | None | Yes - exact distance |
| **Camera requirement** | Single camera | Multiple cameras (typically 6) |
| **Coverage** | Camera FOV only | 360° surround |
| **Occlusion handling** | Limited | Multi-view fusion helps |
| **Path planning ready** | No - needs depth estimation | Yes - directly usable |
| **Computational cost** | Lower | Higher |
| **Model complexity** | Simpler | Complex (transformers + attention) |

### Practical Example

**Scenario**: A car is detected ahead

**2D Tracking Output:**
```
Frame 1: bbox=[500, 300, 600, 400], track_id=5
Frame 2: bbox=[510, 305, 610, 405], track_id=5
Observation: Car moved 10 pixels right
Question: Is it 5 meters away or 50 meters? UNKNOWN
```

**BEV Perception Output:**
```
Frame 1: position=(15.2m, 3.5m, 0m), heading=5°, track_id=5
Frame 2: position=(14.8m, 3.3m, 0m), heading=4°, track_id=5
Observation: Car is 14.8m ahead, 3.3m to the left, approaching at 0.4m/frame
Action: It's moving into our lane, prepare to brake
```

### When to Use Each Approach

| Use Case | Recommended Approach | Reasoning |
|----------|---------------------|-----------|
| Video surveillance | 2D Tracking | Fixed camera, no depth needed |
| Sports analytics | 2D Tracking | Known field dimensions, calibration possible |
| People counting | 2D Tracking | Simple, fast, sufficient |
| Traffic monitoring | 2D Tracking | Fixed infrastructure cameras |
| **Autonomous driving** | **BEV Perception** | Need 3D for path planning |
| **Robot navigation** | **BEV Perception** | Need depth for obstacle avoidance |
| **Drone flight** | **BEV Perception** | 3D awareness critical |
| **ADAS features** | **BEV Perception** | Collision warnings need distance |

### BEV Models Overview

| Model | Input | Key Innovation | Performance |
|-------|-------|----------------|-------------|
| **BEVFormer** | Cameras only | Spatiotemporal transformers | 56.9% NDS on nuScenes |
| **BEVFusion** | Cameras + LiDAR | Multi-modal fusion | Best accuracy |
| **PETR** | Cameras only | 3D position embeddings | Good accuracy |
| **BEVDet** | Cameras only | Explicit depth estimation | Fast |

### Current Pipeline Scope

Our current implementation focuses on **2D perception** because:

1. **Simpler deployment** - Single camera, less hardware
2. **Broader applicability** - Works for non-AV use cases
3. **Foundation first** - 2D tracking is a building block for 3D
4. **Research flexibility** - Can add BEV later as separate module

**Future enhancement**: BEVFormer integration for full 3D perception when multi-camera input is available.

---

## Model Weights Management

### Weight Registry Design

We maintain a centralized registry of pretrained weights:

```python
WEIGHT_REGISTRY = {
    "model-variant": {
        "url": "download_url",
        "filename": "local_filename",
        "size_mb": expected_size,
        "framework": "ultralytics|dfine|groundingdino",
        "description": "Human-readable description",
    }
}
```

### Supported Model Weights

| Category | Models | Variants |
|----------|--------|----------|
| RT-DETR | rtdetr | l, x |
| D-FINE | dfine | n, s, m, l, x |
| YOLO-World | yoloworld | s, m, l, x |
| GroundingDINO | groundingdino | tiny, base |
| YOLOv11 | yolo11 | n, s, m, l, x |
| YOLOv12 | yolo12 | n, s, m, l, x |
| RF-DETR | rfdetr | b, l |
| OSNet (Re-ID) | osnet | x1.0, x0.75, x0.5, x0.25 |

### Automatic Download

```python
from cv_pipeline.utils.weights import WeightsManager

manager = WeightsManager()
weights_path = manager.get_weights("dfine-l")  # Downloads if not cached
```

---

## Next Steps

### Recommended Priority Order

1. **BEV Perception** (if multi-camera available)
   - Implement BEVFormer for 3D detection
   - Add camera calibration utilities
   - Create 3D tracking pipeline

2. **Additional Trackers**
   - OC-SORT (observation-centric)
   - DeepSORT (deep association)
   - StrongSORT (enhanced features)

3. **Model Optimization**
   - TensorRT export for faster inference
   - ONNX export for cross-platform
   - INT8 quantization for edge devices

4. **Segmentation Integration**
   - SAM (Segment Anything Model)
   - Mask2Former for panoptic segmentation

---

## File Structure Summary

```
src/cv_pipeline/
├── datasets/
│   └── __init__.py          # VideoDataSet, TensorDataSet, etc.
├── models/
│   ├── __init__.py
│   └── reid/
│       └── __init__.py      # Re-ID extractors
├── pipelines/
│   ├── data_processing/     # Frame loading, preprocessing
│   ├── object_detection/    # RF-DETR, RT-DETR, D-FINE, YOLO, YOLO-World, GroundingDINO
│   ├── tracking/            # ByteTrack, BoT-SORT
│   ├── lane_detection/      # CLRNet, LaneATT, UFLD
│   └── path_construction/   # Path planning, trajectory generation
└── utils/
    ├── __init__.py
    ├── visualization.py     # Drawing functions
    ├── cmc.py               # Camera motion compensation
    ├── metrics.py           # Evaluation metrics (mAP, MOTA, IDF1)
    ├── mlflow_utils.py      # Experiment tracking
    ├── weights.py           # Model weights management
    └── profiling.py         # Performance profiling

tests/
├── conftest.py              # Pytest fixtures
├── test_datasets.py
├── test_visualization.py
├── test_metrics.py
├── test_cmc.py
├── test_reid.py
├── test_mlflow_utils.py
├── test_weights.py
├── test_profiling.py
├── test_integration.py
├── test_pipeline_registry.py
└── test_object_detection.py  # New: Tests for all detection models
```

---

## 3D Perception Pipeline

### Overview

For full autonomous driving capability, we need 3D perception - understanding object positions in real-world coordinates (meters), not just image pixels. Our 3D perception pipeline consists of 5 key modules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         3D PERCEPTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │ Monocular Depth │    │  BEV Perception │    │   3D Tracking   │          │
│  │   (ZoeDepth)    │    │  (SparseBEV)    │    │   (AB3DMOT)     │          │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘          │
│           │                      │                      │                    │
│           ▼                      ▼                      ▼                    │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              Unified End-to-End (UniAD / Sparse4D v3)           │        │
│  │     Detection → Tracking → Prediction → Planning (all tasks)    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Monocular Depth Estimation

**Purpose**: Estimate per-pixel depth from a single RGB image.

**Location**: `src/cv_pipeline/pipelines/depth_estimation/`

#### Supported Models

| Model | Accuracy (Abs Rel↓) | Speed | Best For |
|-------|---------------------|-------|----------|
| ZoeDepth | 0.075 | Medium | Production, mixed indoor/outdoor |
| Metric3D | 0.082 | Medium | When camera intrinsics available |
| DepthAnything | 0.090 | Fast | Real-time applications |
| MiDaS | 0.110 | Fast | Robust baseline, any domain |

#### ZoeDepth Architecture

```
Input Image [H, W, 3]
      ↓
┌─────────────────────────────────────┐
│     MiDaS Backbone (DPT-Hybrid)     │  ← Relative depth features
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│    Metric Bins Module               │  ← Learn depth distribution
│    (64 bins for indoor/outdoor)     │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│    Attractor Module                 │  ← Fine-grained depth refinement
└─────────────────────────────────────┘
      ↓
Dense Depth Map [H, W] in meters
```

**Key Insight**: ZoeDepth combines **relative depth** (good at ordering) with **metric heads** (good at scale) using learned bin centers.

#### Lifting 2D to 3D

With depth, we can lift 2D detections to 3D positions:

```python
# Pinhole camera model
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
# Z comes from depth map at detection center
```

**Limitations**:
- Monocular depth is approximate (±10-20% error)
- No multi-camera fusion
- Single viewpoint only

---

### 2. BEV Perception (SparseBEV, BEVFormer)

**Purpose**: Project multi-camera images to Bird's Eye View for 3D object detection.

**Location**: `src/cv_pipeline/pipelines/bev_perception/`

#### Why BEV?

```
2D Detection                          BEV Perception
─────────────                          ───────────────
bbox: [450, 320, 550, 420]            position: [15.2m, 3.5m, 0.5m]
(pixels, no depth)                     (meters from ego vehicle)

Can answer: "Is object in image?"     Can answer: "Is object 15m away?"
Cannot: "How far is object?"          Can: "Will we collide in 3 seconds?"
```

#### Supported Models

| Model | NDS (nuScenes) | FPS | Key Innovation |
|-------|---------------|-----|----------------|
| SparseBEV | 67.5% | 23.5 | Fully sparse queries, SOTA |
| BEVFormer | 56.9% | 5-10 | Dense BEV grid, first transformer BEV |
| Sparse4D | 65.0% | 20+ | 4D anchors (space + time) |
| StreamPETR | 62.0% | 30+ | Streaming perception, long memory |

#### SparseBEV Architecture

```
6 Cameras (360° coverage)
         ↓
┌─────────────────────────────────────┐
│     Shared Image Backbone           │  ← ResNet-50/101, VoVNet
│     (extract 2D features)           │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│   Sparse BEV Queries (N=900)        │  ← Learnable 3D reference points
│   (not dense grid like BEVFormer)   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│   Scale-Adaptive Self-Attention     │  ← Handle objects of different sizes
│   (adaptive receptive field)        │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│   Temporal Self-Attention           │  ← Fuse with 4 history frames
│   (motion modeling)                 │
└─────────────────────────────────────┘
         ↓
3D Bounding Boxes [x, y, z, w, l, h, yaw, vx, vy]
```

**Why SparseBEV is SOTA**:
1. **Sparse queries**: Only 900 queries vs 40,000 BEV cells (100x fewer)
2. **Scale-adaptive attention**: Small objects get fine detail, large objects get context
3. **Adaptive sampling**: Sample image features where objects are, not everywhere

#### Detection3D Output Format

```python
@dataclass
class Detection3D:
    center: np.ndarray    # [x, y, z] in meters (ego frame)
    size: np.ndarray      # [w, l, h] in meters
    rotation: float       # yaw in radians
    velocity: np.ndarray  # [vx, vy] in m/s
    confidence: float     # 0-1
    class_id: int         # nuScenes class (0-9)
    class_name: str       # "car", "pedestrian", etc.
```

---

### 3. 3D Object Tracking (AB3DMOT, SimpleTrack)

**Purpose**: Associate 3D detections across frames to maintain object identities.

**Location**: `src/cv_pipeline/pipelines/tracking_3d/`

#### Why 3D Tracking?

2D tracking loses information when objects overlap in image space:

```
2D Tracking:                      3D Tracking:
─────────────                     ────────────
Car A: bbox [100, 200, 200, 300]  Car A: [15m, 3m, 0m]
Car B: bbox [110, 210, 210, 310]  Car B: [35m, 5m, 0m]

Overlap in image! Hard to track.   20m apart in 3D! Easy to track.
```

#### Supported Trackers

| Tracker | AMOTA (nuScenes) | Key Features |
|---------|------------------|--------------|
| AB3DMOT | ~65% | Kalman filter, 3D IoU, simple baseline |
| SimpleTrack | ~70% | Two-stage association, velocity prediction |
| CenterPoint | ~66% | Part of CenterPoint detector |
| OC-SORT 3D | ~68% | Observation-centric, handles occlusion |

#### AB3DMOT Algorithm

```python
def update(detections):
    # 1. Predict all existing tracks using Kalman filter
    for track in tracks:
        predicted_state = kalman_predict(track)
    
    # 2. Compute 3D IoU cost matrix
    cost_matrix = compute_3d_iou(tracks, detections)
    
    # 3. Hungarian matching
    matched, unmatched_dets, unmatched_tracks = hungarian_match(cost_matrix)
    
    # 4. Update matched tracks
    for track_idx, det_idx in matched:
        tracks[track_idx].update(detections[det_idx])
    
    # 5. Create new tracks for unmatched detections
    for det_idx in unmatched_dets:
        new_track = Track3D(detections[det_idx])
        tracks.append(new_track)
    
    # 6. Delete dead tracks (no detection for N frames)
    tracks = [t for t in tracks if not t.is_dead]
```

#### Kalman Filter State

The 3D Kalman filter tracks a 10-dimensional state:

```
State Vector: [x, y, z, yaw, l, w, h, vx, vy, vz]
                ↑ position  ↑ orientation  ↑ size  ↑ velocity

Motion Model: Constant Velocity
  x' = x + vx * dt
  y' = y + vy * dt
  z' = z + vz * dt
  (yaw, l, w, h assumed constant)
```

#### 3D IoU Computation

```python
def compute_3d_iou(box1, box2):
    # 1. Compute axis-aligned bounding box (AABB) intersection
    inter_min = np.maximum(box1.min, box2.min)
    inter_max = np.minimum(box1.max, box2.max)
    
    # 2. Intersection volume
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)
    
    # 3. Union volume
    vol1 = box1.length * box1.width * box1.height
    vol2 = box2.length * box2.width * box2.height
    union_vol = vol1 + vol2 - inter_vol
    
    return inter_vol / union_vol
```

---

### 4. End-to-End Autonomous Driving (UniAD)

**Purpose**: Unify all perception + prediction + planning tasks in one model.

**Location**: `src/cv_pipeline/pipelines/end_to_end/`

#### Why End-to-End?

Traditional Pipeline (Modular):
```
Detect → Track → Map → Predict → Plan
   ↓        ↓      ↓       ↓        ↓
 Loss    Loss   Loss    Loss     Loss

Problem: Each module optimized independently
         Information loss at each handoff
         Errors compound through pipeline
```

End-to-End (UniAD):
```
┌────────────────────────────────────────────┐
│            Single Neural Network           │
│  ┌──────┬───────┬─────┬─────────┬──────┐  │
│  │Detect│ Track │ Map │ Predict │ Plan │  │
│  └──────┴───────┴─────┴─────────┴──────┘  │
│                     ↓                      │
│              Joint Loss Function           │
│         (optimize for final planning)      │
└────────────────────────────────────────────┘

Advantage: All tasks share features
           Gradients flow end-to-end
           Optimize for what matters: safe planning
```

#### UniAD Architecture (CVPR 2023 Best Paper)

```
Multi-Camera Images
        ↓
┌──────────────────────────────────────────────┐
│              BEV Encoder                     │
│   (ResNet-50 → BEV features via attention)   │
└──────────────────────────────────────────────┘
        ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Track Query    │    Map Query    │  Motion Query   │
│ (detect+track)  │   (lanes/etc)   │   (forecast)    │
└─────────────────┴─────────────────┴─────────────────┘
        ↓                 ↓                  ↓
┌──────────────────────────────────────────────────────┐
│            Unified Transformer Decoder               │
│         (cross-attention between all queries)        │
└──────────────────────────────────────────────────────┘
        ↓
┌─────────┬─────────┬─────────┬───────────┬─────────┐
│ 3D Det  │ Tracks  │   Map   │ Forecasts │  Plan   │
│ Output  │ Output  │ Output  │  Output   │ Output  │
└─────────┴─────────┴─────────┴───────────┴─────────┘
```

#### Tasks in UniAD

| Task | Query Type | Output | Purpose |
|------|------------|--------|---------|
| Detection | Track queries | 3D boxes | Find objects |
| Tracking | Track queries (propagated) | Track IDs | Maintain identity |
| Mapping | Map queries | Lane polylines | Understand road |
| Motion Forecast | Motion queries | Future trajectories | Predict others |
| Occupancy | Occupancy queries | Future occupancy grid | Scene-level prediction |
| Planning | Ego query | Ego trajectory | What should ego do? |

#### Motion Forecasting Output

```python
@dataclass
class MotionForecast:
    track_id: int                # Which object
    modes: List[np.ndarray]      # K possible futures, each [T, 2]
    probabilities: List[float]   # Probability of each mode
    timestamps: np.ndarray       # [0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s]

# Example: Predicting a car might go straight OR turn left
forecast.modes[0] = [[1, 0], [2, 0], [3, 0], ...]  # Go straight (prob: 0.7)
forecast.modes[1] = [[1, 1], [2, 3], [3, 5], ...]  # Turn left (prob: 0.3)
```

#### Planning Output

```python
@dataclass
class PlanningOutput:
    trajectory: np.ndarray       # [T, 2] planned ego positions
    timestamps: np.ndarray       # When to be at each position
    velocities: np.ndarray       # Planned speeds
    confidence: float            # Planning confidence

# Example: Plan to drive forward and slightly left
plan.trajectory = [
    [0, 0],      # t=0: current position
    [2, 0.2],    # t=0.5s: 2m forward, 0.2m left
    [4, 0.4],    # t=1.0s: 4m forward, 0.4m left
    [6, 0.5],    # t=1.5s: 6m forward, 0.5m left
]
```

---

### 5. Sparse4D v3: Unified Detection + Tracking

**Purpose**: Combine detection and tracking in a single model via anchor propagation.

**Location**: `src/cv_pipeline/pipelines/bev_perception/sparse4d_v3.py`

#### Key Innovation: 4D Anchors

Traditional tracking: Detect objects → Match across frames (Hungarian)

Sparse4D v3: Anchors ARE tracks (no matching needed)

```
Frame t-1: Anchor_5 at [15m, 3m] ─── propagate ───→ Anchor_5 at [14.5m, 3m]
                                    (warp by ego motion)

Frame t:   Update Anchor_5 with detection at [14.3m, 2.9m]
           → Track ID preserved automatically!
```

#### Architecture

```
Previous 4D Anchors (N=900)
         ↓ (propagate by ego motion + velocity)
Predicted Anchor Positions
         ↓
┌────────────────────────────────────┐
│    Feature Sampling from Images    │
│   (sample at predicted locations)  │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│    Sparse Temporal Attention       │
│   (fuse current + history frames)  │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│    Iterative Box Refinement        │
│   (6 refinement stages)            │
└────────────────────────────────────┘
         ↓
Updated 4D Anchors with Detections + Track IDs
```

#### Why Sparse4D v3 is Elegant

| Traditional Approach | Sparse4D v3 |
|---------------------|-------------|
| Detect → Track (two models) | Single unified model |
| Hungarian matching O(N³) | Anchor propagation O(N) |
| Tracking is post-hoc | Tracking is learned end-to-end |
| Separate velocity estimation | Velocity in anchor state |

#### Anchor4D Data Structure

```python
@dataclass
class Anchor4D:
    position: np.ndarray    # [x, y, z] current position
    size: np.ndarray        # [w, l, h] box dimensions
    rotation: float         # yaw angle
    velocity: np.ndarray    # [vx, vy] for propagation
    track_id: int           # persistent ID
    confidence: float       # detection score
    age: int               # frames since creation
    features: np.ndarray    # learned anchor features
    
    def propagate(self, ego_motion, dt):
        """Propagate anchor to next frame."""
        # Apply velocity
        new_pos = self.position + self.velocity * dt
        
        # Compensate for ego motion
        new_pos = ego_motion @ new_pos
        
        # Decay confidence (require re-detection)
        new_conf = self.confidence * 0.9
        
        return Anchor4D(new_pos, ..., track_id=self.track_id)
```

---

### Performance Summary

| Module | Model | Metric | Value | Hardware |
|--------|-------|--------|-------|----------|
| Depth | ZoeDepth | Abs Rel | 0.075 | Single GPU |
| BEV | SparseBEV | NDS | 67.5% | V100 |
| 3D Track | AB3DMOT | AMOTA | ~65% | CPU |
| End-to-End | UniAD | NDS + Plan L2 | 60.5% + 1.03m | 8x A100 |
| Unified | Sparse4D v3 | NDS | 71.9% | V100 |

---

### When to Use Each Module

| Scenario | Recommended Module |
|----------|-------------------|
| Single camera, basic 3D | Depth Estimation + Lift |
| Multi-camera, 3D detection | SparseBEV / BEVFormer |
| 3D detection + tracking | Sparse4D v3 |
| Full autonomous stack | UniAD |
| Real-time, edge device | Depth + 2D Tracking |
| Maximum accuracy | UniAD + Sparse4D fusion |

---

## Research References

### Object Detection
- RT-DETR: "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024)
- D-FINE: "Redefine Regression Task in DETRs as Fine-grained Distribution Refinement" (ICLR 2025)
- YOLO-World: "Real-Time Open-Vocabulary Object Detection" (CVPR 2024)
- GroundingDINO: "Marrying DINO with Grounded Pre-Training" (ECCV 2024)

### Depth Estimation
- ZoeDepth: "Zero-shot Transfer by Combining Relative and Metric Depth" (arXiv 2023)
- Metric3D: "Towards Zero-shot Metric 3D Prediction from A Single Image" (ICCV 2023)
- DepthAnything: "Unleashing the Power of Large-Scale Unlabeled Data" (CVPR 2024)

### BEV Perception
- BEVFormer: "Learning Bird's-Eye-View Representation from Multi-Camera Images" (ECCV 2022)
- SparseBEV: "High-Performance Sparse 3D Object Detection from Multi-Camera Videos" (ICCV 2023)
- BEVFusion: "Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View" (ICRA 2023)

### 3D Tracking
- AB3DMOT: "3D Multi-Object Tracking: A Baseline and New Evaluation Metrics" (IROS 2020)
- SimpleTrack: "Understanding and Rethinking 3D Multi-object Tracking" (ICCV 2021)
- CenterPoint: "Center-based 3D Object Detection and Tracking" (CVPR 2021)

### End-to-End Autonomous Driving
- UniAD: "Planning-oriented Autonomous Driving" (CVPR 2023 Best Paper)
- VAD: "Vectorized Scene Representation for Efficient Autonomous Driving" (ICCV 2023)
- Sparse4D v3: "Unified Detection and Tracking with Sparse Spatial-Temporal Fusion" (ICLR 2024)

### Tracking (2D)
- ByteTrack: "Simple, Effective and Efficient Multi-Object Tracking" (ECCV 2022)
- BoT-SORT: "Robust Associations Multi-Pedestrian Tracking" (arXiv 2022)
