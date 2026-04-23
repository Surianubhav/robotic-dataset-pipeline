# Egocentric Robotics Data Pipeline

A complete pipeline for converting egocentric (first-person) videos into structured datasets for robotics learning.

## Problem
Robots cannot learn effectively from simulation alone.

They need real-world, first-person human interaction data — but:
- No structured pipelines exist (especially in India)
- Data annotation is expensive
- Scaling is difficult

## Solution
This project builds an end-to-end pipeline that:

1. Takes egocentric video input
2. Segments actions using motion + heuristics
3. Detects objects using YOLO
4. Tracks hand gestures using MediaPipe
5. Infers activities using hybrid logic (flow + objects)
6. Generates structured outputs for training robots

## What makes us different?

This pipeline adopts a heuristic approach instead of training custom deep learning or LLM models. By leveraging pre-trained components (YOLO, MediaPipe) combined with rule based reasoning and motion analysis, it eliminates the need for additional training. This significantly reduces cost, enables real-time performance on edge devices, and allows rapid adaptation to domain specific environments with minimal tuning.

## Pipeline Overview
```
Video Input
↓
Optical Flow (Motion Analysis)
↓
Object Detection (YOLO)
↓
Hand Tracking (MediaPipe)
↓
Heuristic Fusion
↓
Activity Classification
↓
Structured Dataset Output (JSON)
```
## Usage

```bash
python -m src.main --mode video --input sample.mp4
```
## Installation
```bash
pip install -r requirements.txt
```
## Contributors

Anubhav Suri

Manjot Kaur:- https://github.com/

Gorisha Soni:- https://github.com/Gorisha3
