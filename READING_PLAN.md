# CV Pipeline Research Reading Plan

A curated reading plan for the latest leading papers in object detection, multi-object tracking, depth estimation, BEV perception, and end-to-end autonomous driving.

---

## Reading Plan Overview

| Week | Topic | Papers | Priority |
|------|-------|--------|----------|
| 1 | Object Detection Foundations | D-FINE, RF-DETR, RT-DETR | High |
| 2 | Open-Vocabulary Detection | YOLO-World, GroundingDINO | High |
| 3 | Multi-Object Tracking | MOTIP, ByteTrack, BoT-SORT | High |
| 4 | Monocular Depth Estimation | Depth Anything 3, ZoeDepth | Medium |
| 5 | BEV Perception | SparseBEV, BEVFormer | High |
| 6 | End-to-End Autonomous Driving | UniAD, DriveTransformer | High |
| 7 | Unified Detection + Tracking | Sparse4D v3 | Medium |
| 8 | Lane Detection & Planning | CLRerNet, VAD | Medium |

---

## Week 1: Object Detection Foundations

### 1.1 D-FINE (ICLR 2025 Spotlight) - **MUST READ**

**Paper**: "D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement"

**Why Read**: Current SOTA real-time detector, 59.3% AP on COCO with Objects365 pretraining. Apache 2.0 license makes it commercially viable.

**Key Innovations**:
- Fine-grained Distribution Refinement (FDR) for bounding box regression
- Global Optimal Localization Self-Distillation (GO-LSD)
- Up to 5.3% AP improvement on existing DETR models

**Performance**: 54.0% / 55.8% AP at 124 / 78 FPS (T4 GPU)

**Links**:
- Paper: https://arxiv.org/abs/2410.13842
- Code: https://github.com/Peterande/D-FINE
- OpenReview: https://openreview.net/forum?id=MFZjrTFE7h

**Reading Time**: 2-3 hours

---

### 1.2 RF-DETR (ICLR 2026) - **MUST READ**

**Paper**: "RF-DETR: Real-time Object Detection and Segmentation"

**Why Read**: First detector to break 60% mAP in real-time using DINOv2 backbone. Designed for fine-tuning on custom datasets.

**Key Innovations**:
- DINOv2 vision transformer backbone
- SOTA accuracy/latency trade-offs
- Excellent for domain-specific fine-tuning

**Performance**: 54.7% mAP50:95, 73.6% mAP50 (real-time)

**Links**:
- Paper: Roboflow technical report
- Code: https://github.com/roboflow/rf-detr

**Reading Time**: 1-2 hours

---

### 1.3 RT-DETR / RT-DETRv2 (CVPR 2024)

**Paper**: "DETRs Beat YOLOs on Real-time Object Detection"

**Why Read**: Landmark paper proving transformers can beat CNNs in real-time detection. Foundation for D-FINE and RF-DETR.

**Key Innovations**:
- Hybrid encoder (CNN + Transformer)
- Flexible decoder layer tuning for speed/accuracy trade-off
- NMS-free detection

**Performance**: 53.1%+ AP at 108 FPS

**Links**:
- Paper: https://arxiv.org/abs/2304.08069
- Code: https://github.com/lyuwenyu/RT-DETR

**Reading Time**: 2-3 hours

---

## Week 2: Open-Vocabulary Detection

### 2.1 YOLO-World (CVPR 2024)

**Paper**: "Real-Time Open-Vocabulary Object Detection"

**Why Read**: Enables zero-shot detection via text prompts - detect any object without retraining.

**Key Innovations**:
- Vision-language path aggregation
- Re-parameterizable weights for efficiency
- Real-time open-vocabulary detection

**Use Case**: Detect custom objects ("red sports car", "delivery truck") without fine-tuning

**Links**:
- Paper: https://arxiv.org/abs/2401.17270
- Code: https://github.com/AILab-CVC/YOLO-World

**Reading Time**: 2 hours

---

### 2.2 GroundingDINO (ECCV 2024)

**Paper**: "Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"

**Why Read**: Natural language prompts for detection. Foundation for many vision-language applications.

**Key Innovations**:
- Text-grounded object detection
- Feature enhancer for cross-modality fusion
- Language-guided query selection

**Performance**: 52.5% AP zero-shot, 63.0% AP fine-tuned on COCO

**Links**:
- Paper: https://arxiv.org/abs/2303.05499
- Code: https://github.com/IDEA-Research/GroundingDINO

**Reading Time**: 2-3 hours

---

### 2.3 FSOD-VFM (ICLR 2026 Submission)

**Paper**: "Few-Shot Object Detection with Vision Foundation Models"

**Why Read**: Leverages vision foundation models (SAM2, DINOv2) for few-shot detection.

**Key Innovations**:
- Universal Proposal Network (UPN) for category-agnostic detection
- SAM2 integration for mask extraction
- DINOv2 features for rapid adaptation

**Links**:
- OpenReview: https://openreview.net/forum?id=jHlAq2rYUw

**Reading Time**: 2 hours

---

## Week 3: Multi-Object Tracking

### 3.1 MOTIP (CVPR 2025) - **MUST READ**

**Paper**: "Multiple Object Tracking as ID Prediction"

**Why Read**: Novel paradigm treating tracking as ID prediction rather than detection + association.

**Key Innovations**:
- Learnable ID embeddings for trajectory representation
- Deformable DETR integration for object-level embeddings
- Unified detection and tracking

**Links**:
- Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf
- Code: https://github.com/MCG-NJU/MOTIP

**Reading Time**: 2-3 hours

---

### 3.2 TrackTrack (CVPR 2025)

**Paper**: "Track-Focused Online Multi-Object Tracker"

**Why Read**: Addresses MOT from track perspective rather than detection perspective.

**Key Innovations**:
- Track-Perspective-Based Association (TPA)
- Track-Aware Initialization (TAI)
- Online processing without future frame dependency

**Links**:
- CVPR 2025: https://cvpr.thecvf.com/virtual/2025/poster/35174

**Reading Time**: 2 hours

---

### 3.3 ByteTrack (ECCV 2022) - **Foundation Paper**

**Paper**: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

**Why Read**: Essential foundation - simple yet effective. Used in your current pipeline.

**Key Innovations**:
- Associate low-confidence detections (the "byte" that was thrown away)
- Two-stage association: high-score first, then low-score
- Simple Kalman filter + IoU matching

**Performance**: 80.3% MOTA, 77.3% IDF1 on MOT17

**Links**:
- Paper: https://arxiv.org/abs/2110.06864
- Code: https://github.com/ifzhang/ByteTrack

**Reading Time**: 1-2 hours

---

### 3.4 BoT-SORT (2022) - **Foundation Paper**

**Paper**: "BoT-SORT: Robust Associations Multi-Pedestrian Tracking"

**Why Read**: Best accuracy with Re-ID and camera motion compensation. Used in your pipeline.

**Key Innovations**:
- Camera motion compensation (CMC)
- Appearance-based matching with Re-ID
- Kalman filter improvements

**Performance**: 80.5% MOTA, 80.2% IDF1 on MOT17

**Links**:
- Paper: https://arxiv.org/abs/2206.14651
- Code: https://github.com/NirAharon/BoT-SORT

**Reading Time**: 2 hours

---

### 3.5 VOVTrack (ICLR 2025)

**Paper**: "Exploring the Potentiality in Videos for Open-Vocabulary Object Tracking"

**Why Read**: Combines open-vocabulary detection with multi-object tracking.

**Key Innovations**:
- Open-vocabulary MOT (OVMOT)
- Video-centric training for temporal consistency
- Track both seen and unseen categories

**Links**:
- OpenReview: https://openreview.net/forum?id=3vxfFFP3q5

**Reading Time**: 2 hours

---

## Week 4: Monocular Depth Estimation

### 4.1 Depth Anything 3 (November 2025) - **LATEST SOTA**

**Paper**: "Depth Anything 3: Recovering the Visual Space from Any Views"

**Why Read**: Latest SOTA, 35.7% improvement over prior methods. Handles multi-view inputs.

**Key Innovations**:
- Single plain transformer (vanilla DINOv2) without architectural specialization
- Unified depth-ray prediction target
- Handles arbitrary number of visual inputs with or without camera poses

**Performance**: Outperforms DA2 and VGGT across all metrics

**Links**:
- Paper: https://arxiv.org/abs/2511.10647
- Project: https://depth-anything-3.github.io/
- Code: https://github.com/ByteDance-Seed/Depth-Anything-3

**Reading Time**: 2-3 hours

---

### 4.2 Depth Anything V2 (NeurIPS 2024)

**Paper**: "Depth Anything V2: A More Capable Foundation Model for Monocular Depth Estimation"

**Why Read**: Strong foundation model, best in wildlife/outdoor benchmarks.

**Key Innovations**:
- Larger scale training with synthetic data
- Better generalization to diverse domains
- Video-consistent depth (Video Depth Anything, Jan 2025)

**Performance**: MAE 0.454m, correlation 0.962 (wildlife benchmark)

**Links**:
- Code: https://github.com/DepthAnything/Depth-Anything-V2

**Reading Time**: 2 hours

---

### 4.3 ZoeDepth (2023) - **Foundation Paper**

**Paper**: "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth"

**Why Read**: Clever combination of relative and metric depth. Used in your pipeline.

**Key Innovations**:
- Metric Bins Module for depth distribution learning
- Attractor Module for fine-grained refinement
- Domain-agnostic relative depth + domain-specific metric heads

**Performance**: 0.075 Abs Rel (indoor/outdoor)

**Links**:
- Paper: https://arxiv.org/abs/2302.12288
- Code: https://github.com/isl-org/ZoeDepth

**Reading Time**: 2 hours

---

### 4.4 Depth Pro (Apple, 2024)

**Paper**: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"

**Why Read**: High-resolution depth with sharp boundaries. Apple's production-ready solution.

**Key Innovations**:
- Sharp object boundaries
- Metric depth without camera intrinsics
- Fast inference (~second per image)

**Links**:
- LearnOpenCV: https://learnopencv.com/depth-pro-monocular-metric-depth/

**Reading Time**: 1-2 hours

---

## Week 5: BEV Perception

### 5.1 SparseBEV (ICCV 2023) - **MUST READ**

**Paper**: "SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos"

**Why Read**: SOTA BEV detection with sparse queries (100x fewer than dense methods).

**Key Innovations**:
- Sparse BEV queries (900 vs 40,000 dense cells)
- Scale-adaptive self-attention
- Adaptive sampling from images

**Performance**: 67.5% NDS on nuScenes at 23.5 FPS

**Links**:
- Paper: https://arxiv.org/abs/2308.09244

**Reading Time**: 2-3 hours

---

### 5.2 BEVFormer (ECCV 2022) - **Foundation Paper**

**Paper**: "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"

**Why Read**: First transformer-based BEV perception. Foundation for later work.

**Key Innovations**:
- Spatial cross-attention for multi-camera fusion
- Temporal self-attention for temporal modeling
- Dense BEV grid representation

**Performance**: 56.9% NDS on nuScenes

**Links**:
- Paper: https://arxiv.org/abs/2203.17270
- Code: https://github.com/fundamentalvision/BEVFormer

**Reading Time**: 3-4 hours

---

### 5.3 DriveTransformer (ICLR 2025)

**Paper**: "DriveTransformer: Unified Transformer for Scalable End-to-End Autonomous Driving"

**Why Read**: Addresses cumulative errors in sequential perception-prediction-planning paradigm.

**Key Innovations**:
- Non-sequential task execution
- Efficient long-range perception
- Long-term temporal fusion

**Links**:
- OpenReview: https://openreview.net/forum?id=M42KR4W9P5

**Reading Time**: 2-3 hours

---

### 5.4 BEVWorld (2025)

**Paper**: "BEVWorld: A Multimodal World Model for Autonomous Driving via Unified BEV Latent Space"

**Why Read**: World model approach for autonomous driving in BEV space.

**Key Innovations**:
- Multimodal tokenizer to unified BEV latent space
- Latent BEV sequence diffusion model
- Environment modeling for prediction

**Links**:
- OpenReview: https://openreview.net/forum?id=MFrqTfubEB

**Reading Time**: 2 hours

---

### 5.5 BEV Perception Survey (2024)

**Paper**: "BEV perception for autonomous driving: State of the art and future perspectives"

**Why Read**: Comprehensive survey covering the entire BEV perception landscape.

**Links**:
- ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0957417424019705

**Reading Time**: 3-4 hours (skim recommended)

---

## Week 6: End-to-End Autonomous Driving

### 6.1 UniAD (CVPR 2023 Best Paper) - **MUST READ**

**Paper**: "Planning-oriented Autonomous Driving"

**Why Read**: CVPR 2023 Best Paper. Unified framework for all AD tasks.

**Key Innovations**:
- Unified detection, tracking, mapping, prediction, planning
- Query-based multi-task learning
- Planning-oriented optimization

**Performance**: 60.5% NDS, 1.03m planning L2 error

**Links**:
- Paper: https://arxiv.org/abs/2212.10156
- Code: https://github.com/OpenDriveLab/UniAD

**Reading Time**: 3-4 hours

---

### 6.2 BridgeAD (CVPR 2025)

**Paper**: "Bridging Past and Future: End-to-End Autonomous Driving with Historical Prediction"

**Why Read**: Latest CVPR 2025 work enhancing E2E driving with historical context.

**Key Innovations**:
- Historical prediction for current frame perception
- Future frame prediction for planning
- Bidirectional temporal reasoning

**Links**:
- Paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Bridging_Past_and_Future_End-to-End_Autonomous_Driving_with_Historical_Prediction_CVPR_2025_paper.pdf

**Reading Time**: 2-3 hours

---

### 6.3 VAD (ICCV 2023)

**Paper**: "VAD: Vectorized Scene Representation for Efficient Autonomous Driving"

**Why Read**: Efficient vectorized representation for autonomous driving.

**Key Innovations**:
- Vectorized scene representation (vs rasterized)
- Efficient planning with vector outputs
- End-to-end trainable

**Links**:
- Paper: https://arxiv.org/abs/2303.12077

**Reading Time**: 2-3 hours

---

### 6.4 Vision-Language-Action (VLA) Models (2025 Trend)

**Why Read**: Emerging paradigm combining vision, language, and action. Projected to dominate 50%+ of L3/L4 markets by 2030.

**Key Concepts**:
- Chain-of-thought reasoning for driving decisions
- LLM-augmented perception and planning
- Reinforcement learning innovations (DeepSeek-R1 influence)

**Industry Examples**: Li Auto VLA, NVIDIA NDAS (Alpamayo)

**Reading Time**: 2-3 hours (multiple sources)

---

## Week 7: Unified Detection + Tracking

### 7.1 Sparse4D v3 (ICLR 2024)

**Paper**: "Sparse4D v3: Advancing End-to-End 3D Detection and Tracking with Focused Supervision"

**Why Read**: Unified detection + tracking via anchor propagation. No Hungarian matching needed.

**Key Innovations**:
- 4D anchors (space + time)
- Anchor propagation for implicit tracking
- Single model for detection + tracking

**Performance**: 71.9% NDS on nuScenes

**Links**:
- Paper: https://arxiv.org/abs/2311.11722

**Reading Time**: 2-3 hours

---

### 7.2 StreamPETR (2023)

**Paper**: "StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection"

**Why Read**: Streaming perception with long-term memory.

**Key Innovations**:
- Object-centric temporal modeling
- Streaming perception paradigm
- Memory-based long-term tracking

**Performance**: 62.0% NDS at 30+ FPS

**Links**:
- Paper: https://arxiv.org/abs/2303.11926

**Reading Time**: 2 hours

---

## Week 8: Lane Detection & Planning

### 8.1 CLRerNet (WACV 2024)

**Paper**: "CLRerNet: Improving Confidence of Lane Detection with LaneIoU"

**Why Read**: SOTA lane detection building on CLRNet.

**Key Innovations**:
- LaneIoU metric for better evaluation
- Confidence improvement techniques
- Real-time performance

**Performance**: 81.5% F1 on CULane at 90 FPS

**Links**:
- Code: https://github.com/hirotomusiker/CLRerNet

**Reading Time**: 2 hours

---

### 8.2 Separable Hierarchical Lane Detection Transformer (2025)

**Paper**: "Aggregate global features into separable hierarchical lane detection transformer"

**Why Read**: Latest transformer-based lane detection.

**Key Innovations**:
- Separable lane multi-head attention
- Window self-attention for efficiency
- Pure transformer architecture

**Links**:
- Nature Scientific Reports: https://www.nature.com/articles/s41598-025-86894-z

**Reading Time**: 2 hours

---

### 8.3 Lane Detection Survey (2025)

**Paper**: "Monocular Lane Detection Based on Deep Learning: A Survey"

**Why Read**: Comprehensive survey of lane detection methods.

**Key Topics**:
- Task paradigms for lane discrimination
- Lane modeling approaches
- 3D lane detection
- HD map construction

**Links**:
- arXiv: https://arxiv.org/abs/2411.16316

**Reading Time**: 3-4 hours (skim recommended)

---

## Quick Reference: Paper Priority Tiers

### Tier 1: Must Read (Foundation + SOTA)
1. **D-FINE** - SOTA real-time detection (ICLR 2025)
2. **UniAD** - End-to-end AD framework (CVPR 2023 Best Paper)
3. **MOTIP** - Novel tracking paradigm (CVPR 2025)
4. **SparseBEV** - SOTA BEV perception (ICCV 2023)
5. **Depth Anything 3** - Latest depth estimation (2025)

### Tier 2: High Priority (Recent Advances)
6. **RF-DETR** - DINOv2-based detection (ICLR 2026)
7. **BridgeAD** - E2E driving with history (CVPR 2025)
8. **DriveTransformer** - Unified E2E transformer (ICLR 2025)
9. **TrackTrack** - Track-focused MOT (CVPR 2025)
10. **YOLO-World** - Open-vocab detection (CVPR 2024)

### Tier 3: Important Context (Foundation Papers)
11. **RT-DETR** - First real-time DETR (CVPR 2024)
12. **ByteTrack** - Simple effective tracking (ECCV 2022)
13. **BoT-SORT** - Tracking with Re-ID (2022)
14. **BEVFormer** - Foundation BEV (ECCV 2022)
15. **ZoeDepth** - Metric depth (2023)

### Tier 4: Domain Specific (As Needed)
16. **GroundingDINO** - Text-prompted detection
17. **Sparse4D v3** - Unified detect+track
18. **CLRerNet** - Lane detection
19. **VAD** - Vectorized driving
20. **BEV Survey** - Comprehensive overview

---

## Reading Schedule Suggestions

### Intensive (4 weeks)
- Week 1: Tier 1 papers (5 papers)
- Week 2: Tier 2 papers (5 papers)
- Week 3: Tier 3 papers (5 papers)
- Week 4: Tier 4 papers (selective)

### Standard (8 weeks)
- Follow the week-by-week plan above
- 2-3 papers per week
- Include time for implementation experiments

### Light (12 weeks)
- 1-2 papers per week
- Focus on Tier 1-2 first
- Add implementation practice between papers

---

## Implementation Practice Suggestions

After reading each section, practice with:

1. **Object Detection**: Fine-tune D-FINE on custom dataset
2. **Tracking**: Implement MOTIP-style ID prediction
3. **Depth**: Compare ZoeDepth vs Depth Anything V2
4. **BEV**: Run SparseBEV on nuScenes samples
5. **E2E**: Experiment with UniAD visualization

---

## Sources

- [D-FINE GitHub](https://github.com/Peterande/D-FINE)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [MOTIP CVPR 2025](https://github.com/MCG-NJU/MOTIP)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [UniAD GitHub](https://github.com/OpenDriveLab/UniAD)
- [Multi-object tracking review (2025)](https://link.springer.com/article/10.1007/s10462-025-11212-y)
- [BEV perception survey](https://www.sciencedirect.com/science/article/abs/pii/S0957417424019705)
- [Lane detection survey](https://arxiv.org/abs/2411.16316)
