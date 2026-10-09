# Third-Party Software and Attribution

This repository contains research code developed for UAV-based RGB and multispectral YOLO instance segmentation, weed detection, TensorRT deployment, and georeferenced weed prescription mapping.

## 1. Ultralytics YOLO

- Project: [Ultralytics](https://github.com/ultralytics/ultralytics)
- License: GNU Affero General Public License v3.0 (AGPL-3.0), subject to the terms applicable to the incorporated version.
- Source: Ultralytics YOLO framework.

This repository incorporates modified Ultralytics source code to support the RGB and five-channel multispectral configurations used in this research, including YOLO11, YOLO12, and YOLO26.

Original copyright and license notices are retained in the applicable source files.

## 2. YOLO-Multispectral

- Project: [YOLO-Multispectral](https://github.com/aesparon/YOLO-Multispectral)
- Author: Andrew Esparon
- License identified for the author's code: MIT License
- Copyright: Copyright (c) 2025 Andrew Esparon

The YOLO-Multispectral project provided a reference for multispectral adaptation of YOLO architectures. This research includes additional implementation work to support YOLO26 and the five-band input configuration used in the sugarbeet experiments.

Where source code from YOLO-Multispectral has been incorporated or adapted, the applicable MIT copyright and permission notice must be preserved.

## 3. Research-Specific Implementations

The repository includes custom scripts for:

- RGB and RGBRN dataset preparation.
- YOLO instance-segmentation training and evaluation.
- TensorRT model export and benchmarking.
- NVIDIA Jetson Orin Nano and AGX Orin deployment.
- Orthomosaic processing and georeferenced weed prescription-map generation.

These scripts were developed for the associated research study.

## 4. License Information

The repository contains Ultralytics-derived code subject to applicable AGPL-3.0 requirements. MIT-licensed components retain their respective copyright and permission notices.

The repository-level license does not remove or replace the notices and conditions applicable to third-party components.
