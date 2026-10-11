# UAV Multispectral YOLO: Automated Agricultural Imagery Mapping on Edge Devices

A research framework for **RGB and multispectral UAV image instance segmentation, automated geospatial mapping, and deployment on NVIDIA Jetson edge devices**. The framework adapts Ultralytics YOLO architectures for three-channel RGB and five-channel **RGBRN (Red, Green, Blue, Red Edge, Near-Infrared)** inputs, with **sugarbeet crop and weed mapping** as the case-study application.

This repository brings together dataset preparation, model training and evaluation, TensorRT export and benchmarking, and orthomosaic-to-prescription-map processing. The included mapping implementations are tailored to the sugarbeet application; adapting them to other crops or agricultural targets requires appropriate datasets, classes, and mapping rules.

## Key features

- **RGB and five-band multispectral input:** R, G, B, Red Edge, and NIR, in that order.
- **YOLO instance segmentation:** research adaptations for YOLO architectures.
- **Model development:** scripts for dataset arrangement, RGB/RGBRN training, and evaluation.
- **Edge optimization:** TensorRT FP32, FP16, and INT8 export and evaluation workflows.
- **NVIDIA Jetson deployment:** dedicated processing scripts for Jetson Orin Nano and Jetson AGX Orin.
- **Geospatial mapping:** orthomosaic inference, crop/weed segmentation outputs, weed-density mapping, and spatially referenced prescription-map generation.

## Workflow

```text
UAV imagery / georeferenced orthomosaics
                  |
                  v
        RGB or RGBRN preparation
                  |
                  v
    Multispectral YOLO instance segmentation
                  |
                  v
       PyTorch or TensorRT deployment
          (NVIDIA Jetson edge)
                  |
                  v
   orthomosaic inference and stitching
                  |
                  v
        Georeferenced weed maps
```

**Scope:** Model support and mapping scripts are included in the repository, but this is research software rather than a turnkey package. Users must configure their paths, environments, and inputs before execution.

## Repository structure

```text
uav-multispectral-yolo/
├── scripts/                 # Research scripts (kept together)
├── uav_MS_yolo/
│   └── code/
│       └── ultralytics/     # Modified Ultralytics implementation
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── CITATION.cff
├── .gitignore
└── PRE_RELEASE_CHECKLIST.md
```

The directory tree summarizes the main project files. Additional directories may be present in the bundled source.

### Research scripts

All research scripts are maintained together in the **`scripts/`** folder; they are not separated into additional subfolders.

| Script | Role |
|---|---|
| `data_arrangement.py` | Arrange and preprocess research imagery and labels |
| `train_RGB.py` | RGB model training |
| `train_RGBRN.py` | Five-channel RGBRN model training |
| `test_RGB.py` | RGB model evaluation |
| `test_RGBRN.py` | RGBRN model evaluation |
| `export_tensorrt_RGB.py` | RGB TensorRT export |
| `export_tensorrt_RGBRN.py` | RGBRN TensorRT export |
| `test_RGB_TensorRT.py` | RGB TensorRT evaluation |
| `test_RGBRN_TensorRT.py` | RGBRN TensorRT evaluation |
| `Mapping_pipeline_RGB_Orinnano.py` | RGB orthomosaic mapping on Orin Nano |
| `Mapping_pipeline_RGBRN_OrinNano.py` | RGBRN orthomosaic mapping on Orin Nano |
| `Mapping_pipeline_RGB_AGXorin.py` | RGB orthomosaic mapping on AGX Orin |
| `Mapping_pipeline_RGBRN_AGXorin.py` | RGBRN orthomosaic mapping on AGX Orin |
| `RGBRN_orthomosaic_for_inference.py` | Prepare a five-band orthomosaic for inference |

The script names are listed as included in the prepared research package; confirm exact capitalization against the GitHub tree if you have subsequently renamed files.

## Case study: sugarbeet and weed mapping

The framework was developed and evaluated using UAV imagery of sugarbeet fields. The research task uses two instance-segmentation categories: **Sugarbeet** (crop) and **Weed**. RGB and RGBRN imagery are used to train and evaluate YOLO instance-segmentation models, while the orthomosaic processing scripts convert model predictions into spatially referenced weed-mapping products.

The sugarbeet pipeline is an **application example**, not a claim that the provided models or prescription rules generalize without retraining to other crops, weed species, sensors, or field conditions.

### Mapping outputs

Depending on the selected mapping script and configuration, the processing workflow is intended to produce:

- Tiled and stitched instance-segmentation predictions for the orthomosaic.
- Georeferenced crop and weed visualization layers.
- Grid-based weed-count or density representations.
- Weed prescription maps derived from the configured density-to-rate rules.

Inspect the selected script for the exact output filenames, grid size, thresholds, and output formats. Validate the coordinate reference system and agronomic prescription settings before any operational use.

## Input data and preprocessing

### RGB

RGB models use images directly obtained from the sensor. No preprocessing steps like RGBRN. The relevant training, evaluation, and mapping scripts should be configured to use the corresponding RGB imagery and trained weights.

### RGBRN

The intended multispectral channel order is:

1. Red (R)
2. Green (G)
3. Blue (B)
4. Red Edge (RE)
5. Near-Infrared (NIR)

Maintain this order throughout band alignment, stacking, tile creation, training, and inference. Multispectral images must have the spatial dimensions, band registration, and numeric representation expected by the modified model loader.

## Installation and environment

The repository includes a modified Ultralytics implementation, not only scripts that use the stock PyPI package. Ensure Python imports the local modified source when running RGBRN models.

The precise dependency versions depend on the host computer or Jetson device. In particular, **PyTorch, CUDA, TensorRT, and JetPack must be compatible with the target platform**. The uploaded research package did not include a fully pinned, verified cross-platform dependency specification, so a single universal installation command is not claimed here.

A basic import-location check, when running from the repository root, is:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "uav_MS_yolo" / "code"))

import ultralytics
from ultralytics import YOLO

print("Version:", ultralytics.__version__)
print("Loaded from:", ultralytics.__file__)
```

The loaded path should point to `uav_MS_yolo/code/ultralytics/`, rather than an unrelated `site-packages` installation. The modified package in the supplied code identifies itself as Ultralytics version **8.4.112**.
For environment setup, dependencies, training, and NVIDIA Jetson deployment instructions, see the Installation Guide.

### Typical usage sequence

1. Prepare RGB or registered five-band RGBRN imagery and annotations.
2. Configure dataset paths and model options in the appropriate script under `scripts/`.
3. Train and evaluate a model using the modified Ultralytics source.
4. Export to TensorRT on a compatible target platform if edge acceleration is required.
5. Configure the matching orthomosaic mapping script for the input imagery, model, grid settings, and output directory.
6. Review segmentation, geospatial alignment, weed-density, and prescription outputs.

**Execution note:** The current scripts contain machine-specific or placeholder paths (for example, `/processed_RGBRN`, `/Orthomosaic_RGB.tif`, and `/Orthomosaic_RGBRN.tif`). Set these to your actual data and model locations before running. Script-specific dependencies and configuration parameters are not yet consolidated into a unified command-line interface.

## Edge deployment

The repository includes research implementations for **NVIDIA Jetson Orin Nano** and **Jetson AGX Orin**. RGB and RGBRN models can be evaluated in PyTorch and TensorRT configurations, including FP32, FP16, and INT8 where supported by the model and device environment.

TensorRT engine files are generally specific to the build environment and hardware/software configuration. Export or rebuild engines on the intended deployment platform, and document the relevant JetPack, CUDA, TensorRT, and PyTorch versions when reporting benchmarks.

For scientific comparison, distinguish **model inference throughput** from **total orthomosaic-to-map processing time**, which can include tiling, inference, stitching, geospatial processing, and writing outputs.

## Data, weights, and reproducibility

Full UAV datasets, orthomosaics, trained `.pt` weights, TensorRT `.engine` files, and generated maps are not included by default. Their redistribution requires separate checks for permissions, size, and compatibility.

For reproducibility, document the dataset splits, band preprocessing, training hyperparameters, software versions, inference thresholds, and geospatial mapping settings used for each reported experiment. See [`PRE_RELEASE_CHECKLIST.md`](PRE_RELEASE_CHECKLIST.md) for remaining release checks.

## Attribution and licensing

This repository incorporates modified **Ultralytics YOLO** source code and contains research-specific scripts for multispectral instance segmentation and agricultural mapping. The Ultralytics-derived portions are subject to the applicable **GNU Affero General Public License v3.0 (AGPL-3.0)** terms.

The multispectral approach was developed with reference to [Andrew Esparon's YOLO-Multispectral](https://github.com/aesparon/YOLO-Multispectral), whose applicable original contributions were provided under the **MIT License**. Where code from that project has been copied or adapted, preserve its full MIT copyright and permission notice.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for license terms and attribution. Separate third-party components, datasets, and weights may carry additional terms.

## Citation

If you use this repository, please cite the associated research publication and/or the archived software release when available. See [`CITATION.cff`](CITATION.cff) for citation metadata; it must be completed with the final author, publication, and release information before archival.

## Research status and disclaimer

This repository is shared to support academic research, transparency, and reproducibility. It is not a validated autonomous spraying control system. Prescription-map outputs require independent spatial and agronomic verification before use in field operations.
