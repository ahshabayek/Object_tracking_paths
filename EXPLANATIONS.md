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

## Next Steps

### Recommended Priority Order

1. **Integration Testing**
   - Test full pipeline end-to-end with sample video
   - Verify data flows correctly between nodes
   - Test MLFlow logging integration

2. **Model Weights Setup**
   - Download pretrained weights for RT-DETR/YOLO
   - Download lane detection model weights (CLRNet)
   - Configure weight paths in `conf/base/parameters.yml`

3. **Performance Optimization**
   - Add TensorRT/ONNX export for inference speedup
   - Implement batch inference for detection
   - Profile and optimize bottlenecks

4. **Documentation**
   - API documentation with Sphinx
   - Usage examples and tutorials
   - Docker container for deployment

5. **Additional Features**
   - Add OC-SORT and DeepSORT trackers
   - Implement 3D object detection integration
   - Add sensor fusion with LiDAR/radar

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
│   ├── object_detection/    # RF-DETR, RT-DETR, YOLO
│   ├── tracking/            # ByteTrack, BoT-SORT
│   ├── lane_detection/      # CLRNet, LaneATT
│   └── path_construction/   # Path planning
└── utils/
    ├── __init__.py
    ├── visualization.py     # Drawing functions
    ├── cmc.py              # Camera motion compensation
    └── metrics.py          # Evaluation metrics

tests/
├── conftest.py             # Pytest fixtures
├── test_datasets.py
├── test_visualization.py
├── test_metrics.py
├── test_cmc.py
└── test_reid.py
```
