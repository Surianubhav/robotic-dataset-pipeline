# Pipeline Architecture

## Overview

This system converts egocentric video/realtime cam into structured robotics datasets using a modular pipeline.

## Steps

1. Video Input
2. Optical Flow (motion detection)
3. Object Detection (YOLO)
4. Hand Tracking (MediaPipe)
5. Heuristic-based fusion
6. Activity classification
7. Dataset generation

## Design Philosophy

- No additional model training required
- Real-time performance
- Edge-device compatibility
- Modular and scalable
