# Sugarbeet Weed Detection and Prescription Mapping with RGB and RGBRN Imagery

Research code for RGB and five-band RGBRN (Red, Green, Blue, Red Edge, NIR) YOLO instance-segmentation experiments, TensorRT export/evaluation, and UAV orthomosaic-to-weed-prescription-map processing on NVIDIA Jetson devices.

> **Reproducibility and release status:** This is a cleaned research-code package, not yet a fully portable, one-command installation. Before public release or use for a new experiment, complete the checks in [`PRE_RELEASE_CHECKLIST.md`](PRE_RELEASE_CHECKLIST.md). In particular, verify the applicable license for the bundled modified Ultralytics source, set all local paths, and make the RGBRN preprocessing used for inference match the preprocessing used to create the training data.

## What's included

### Research scripts in the scripts/ folder

| Script | Purpose (based on filename and code organization) |
|---|---|
| `data_arrangement.py` | Arrange/preprocess RGB and RGBRN training data |
| `train_RGB.py` | Train an RGB model |
| `train_RGBRN.py` | Train a five-channel RGBRN model |
| `test_RGB.py`, `test_RGBRN.py` | Evaluate RGB and RGBRN models |
| `export_tensorrt_RGB.py`, `export_tensorrt_RGBRN.py` | Export models to TensorRT |
| `test_RGB_TensorRT.py`, `test_RGBRN_TensorRT.py` | Evaluate TensorRT models |
| `Mapping_pipeline_RGB_Orinnano.py`, `Mapping_pipeline_RGBRN_OrinNano.py` | RGB/RGBRN prescription-mapping pipeline variants for Jetson Orin Nano |
| `Mapping_pipeline_RGB_AGXorin.py`, `Mapping_pipeline_RGBRN_AGXorin.py` | RGB/RGBRN pipeline variants for Jetson AGX Orin |
| `RGBRN_orthomosaic_for_inference.py` | Prepare RGBRN orthomosaic input for inference |

The filenames and scripts reflect the supplied research workflow. Their configuration, dependencies, model paths, and device-specific assumptions have not all been generalized or tested on a clean machine.

### Modified YOLO source

`uav_MS_yolo/code/ultralytics_uav_MS_yolo/ultralytics/` contains the bundled Ultralytics-derived source used by the scripts. The source package identifies itself as version `8.4.112` and contains AGPL-3.0 license headers. The nested `.git` metadata from the supplied copy has been removed to avoid creating an accidental nested Git repository. See the license/release blocker in `PRE_RELEASE_CHECKLIST.md` before making this repository public.

## Data and model files

The original archive does not provide a portable, complete sample dataset or trained model weights. Do not commit field orthomosaics, annotated datasets, `.pt` weights, TensorRT `.engine` files, or generated prescription outputs unless you have checked size, ownership, privacy, and redistribution permissions. Configure paths to your own data and model artifacts.

The scripts currently contain example or machine-specific absolute paths (for example `/processed_RGBRN`, `/Orthomosaic_RGB.tif`, and `/Orthomosaic_RGBRN.tif`). Update the relevant configuration before running. A path that begins with `/` is an absolute path on Unix-like systems, not a path relative to this repository.

## RGBRN channel order and preprocessing

The intended five-band order is **Red, Green, Blue, Red Edge, NIR**. Preserve this order consistently when preparing tiles, training, exporting, and running inference.

Preprocessing must match between training and inference. The supplied `data_arrangement.py` contains per-band `cv2.normalize(..., 0, 255, cv2.NORM_MINMAX)` processing, while some mapping scripts configure a `divide_256` mode. These operations are not equivalent. Do not assume that the deployment defaults are training-compatible; confirm the exact data-generation path and select matching preprocessing before interpreting model results.

## Environment and execution

The supplied code depends on a compatible combination of Python, PyTorch, the bundled modified Ultralytics source, OpenCV, geospatial/image libraries, and (for TensorRT/Jetson workflows) NVIDIA JetPack, CUDA, TensorRT, and device-specific packages. A fully pinned, cross-platform dependency specification was not included in the supplied archive, so this repository intentionally does not claim that a generic `pip install -r requirements.txt` is sufficient.

Recommended initial workflow:

1. Create an isolated Python environment appropriate for your target device.
2. Install a compatible PyTorch/CUDA/TensorRT stack for that device, where applicable.
3. Review `PRE_RELEASE_CHECKLIST.md` and resolve the license and preprocessing items.
4. Edit dataset, pretrained-weight, orthomosaic, output, and model paths in the relevant script.
5. Confirm the channel count/order and dataset YAML before training or inference.
6. Run a small controlled test and verify outputs before processing a full orthomosaic.

Do not treat the scripts as validated deployment instructions until the environment and end-to-end outputs have been tested on the intended hardware.

## Prescription maps

The mapping scripts are intended to connect orthomosaic inference with geospatial prescription-map outputs. Exact thresholds, grid resolution, output fields, and spraying-rate settings are configured in the scripts and should be reviewed for each experiment and field operation. Validate the resulting maps and coordinate reference system before using them to guide machinery or chemical application.

## Citation

Citation metadata is a template in [`CITATION.cff`](CITATION.cff). Replace the TODO author and release details and archive the exact public release (for example, with Zenodo) before citing a version in a manuscript.

## License and Attribution

This repository includes modified Ultralytics YOLO source code and research-specific implementations for RGB and five-channel multispectral instance segmentation.

The Ultralytics-derived software is subject to the applicable GNU Affero General Public License v3.0 (AGPL-3.0) terms.

The multispectral implementation was developed with reference to the YOLO-Multispectral project by Andrew Esparon. Applicable MIT copyright and permission notices are preserved for any incorporated or adapted code.

For further information, see LICENSE and THIRD_PARTY_NOTICES.md.

## License

No repository-level `LICENSE` file has been added because the supplied archive did not establish the applicable license for the complete collection of custom scripts and bundled upstream-derived source. See [`PRE_RELEASE_CHECKLIST.md`](PRE_RELEASE_CHECKLIST.md). Do not assume that all components can be redistributed under a new license until their provenance and license terms have been checked.
