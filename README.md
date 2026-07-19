# Structure-Driven Feature Modeling for Robust Fish Detection in Complex Underwater Environments

Official PyTorch implementation of the paper:

> **Structure-Driven Feature Modeling for Robust Fish Detection in Complex Underwater Environments**

This repository provides the implementation of the proposed Structure-Driven Feature Modeling method for underwater fish detection.

## Overview

Underwater fish detection is affected by low contrast, uneven illumination, blur, partial occlusion, weak target boundaries, and high similarity between fish and background regions.

To address these challenges, we propose a structure-driven feature modeling mechanism that explicitly enhances boundary-, contour-, and morphology-related information in shallow feature maps.

The proposed method mainly includes:

- Structure Channel Attention;
- Multi-Scale Structure Difference;
- Structure Normalization Module;
- Structure-Guided Spatial Attention;
- Structure-Aware Feature Modeling Units;
- Structure-Driven Feature Modeling;
- Structure-Driven Fusion.

The proposed mechanism is inserted into the high-resolution P3 branch of the detector. It extracts multi-scale residual responses, normalizes their distributions, generates spatial attention weights, and fuses the enhanced representation with the original appearance features.

## Main Features

- Explicit modeling of boundary, contour, and morphological cues;
- Multi-scale residual extraction for suppressing slowly varying background responses;
- Lightweight channel and spatial attention mechanisms;
- Feature enhancement at the high-resolution P3 level;
- Compatibility with multiple YOLO architectures;
- Controlled paired comparisons between original and enhanced detectors.

The method has been evaluated on:

- YOLOv5n;
- YOLOv7-tiny;
- YOLOv8n;
- YOLOv10n;
- YOLOv11n;
- YOLOv12n.

## Method Overview

The proposed framework consists of two main components.

### 1. Structure-Driven Feature Modeling

The Structure-Driven Feature Modeling component progressively processes the P3 feature through a cascade of lightweight Structure-Aware Feature Modeling Units.

Each unit performs:

1. feature re-encoding;
2. channel-response recalibration;
3. multi-scale residual extraction;
4. response normalization;
5. spatial attention;
6. residual refinement.

### 2. Structure-Driven Fusion

The Structure-Driven Fusion component combines the enhanced boundary- and contour-aware representation with the original RGB appearance feature.

This design preserves useful color and texture information while increasing the contribution of geometric and morphological cues.

## Experimental Environment

The experiments reported in the paper were conducted using the following environment:

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 |
| Python | 3.10.0 |
| PyTorch | 2.1.2 |
| CUDA | 12.1 |
| Input resolution | 640 × 640 |
| Batch size | 16 |
| Training epochs | 300 |
| Optimizer | AdamW |
| Initial learning rate | 2 × 10⁻³ |
| Momentum | 0.9 |
| Weight decay | 5 × 10⁻⁴ |
| Warm-up epochs | 3 |
| Pre-trained weights | Not used |

The original comparative and ablation experiments used a fixed random seed of 0. Additional repeated experiments were conducted using random seeds 0, 42, and 1020.

## Installation

Clone this repository:

```bash
git clone https://github.com/Kirito-97/Structure-Driven-Feature-Modeling-for-Robust-Fish-Detection-in-Complex-Underwater-Environments.git
cd Structure-Driven-Feature-Modeling-for-Robust-Fish-Detection-in-Complex-Underwater-Environments
