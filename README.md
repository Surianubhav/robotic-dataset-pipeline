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

Unlike traditional approaches that rely on trained deep learning models, this system uses a hybrid heuristic pipeline combining optical flow, object detection, and rule-based reasoning. This significantly reduces training overhead, lowers deployment cost, and enables real-time execution on edge hardware with minimal domain-specific tuning.
