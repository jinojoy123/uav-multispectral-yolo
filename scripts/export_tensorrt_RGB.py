# RGB YOLO SEGMENTATION MODEL
# TENSORRT EXPORT FOR EDGE DEPLOYMENT
#
# Deployment configurations:
#
#   1. PyTorch FP32       -> original best.pt (unchanged)
#   2. TensorRT FP32      -> .engine
#   3. TensorRT FP16      -> .engine
#   4. TensorRT INT8 PTQ  -> .engine
#
# INT8:
#   Static Post-Training Quantization (PTQ)
#   Calibration data are taken from the validation split.
#
# IMPORTANT:
#   The independent TEST dataset is NOT used for INT8 calibration.

import os
import shutil
import torch

from pathlib import Path
from ultralytics import YOLO


# Trained PyTorch model

MODEL_PATH = r"/weights/best.pt"


# Dataset YAML
# INT8 calibration will use the VALIDATION split.


DATA_YAML = r"model_quantization/dataset.yaml"

# Output directory

OUTPUT_DIR = Path(
    "deployment_RGB"
)

# Input image size

IMG_SIZE = 640

# TensorRT maximum batch size
# For your edge-device experiment where latency per individual image is important, batch=1 is appropriate.

BATCH_SIZE = 1

# TensorRT workspace
# None allows TensorRT to automatically determine the workspace.
# This is safer across Jetson devices with different memory capacities.

WORKSPACE = None

# Calibration fraction
# 1.0 = use the complete validation dataset for INT8 calibration.
# If the validation set is very large, this can later be changed.

CALIBRATION_FRACTION = 1.0

# CREATE OUTPUT DIRECTORY

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# CHECK CUDA

print("\n")
print("=" * 80)
print("DEVICE INFORMATION")
print("=" * 80)


print(
    f"PyTorch version : "
    f"{torch.__version__}"
)


print(
    f"CUDA available  : "
    f"{torch.cuda.is_available()}"
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "\nCUDA GPU was not detected.\n"
        "TensorRT export must be performed on an NVIDIA CUDA device."
    )


print(
    f"CUDA device     : "
    f"{torch.cuda.get_device_name(0)}"
)


print(
    f"CUDA version    : "
    f"{torch.version.cuda}"
)

# VERIFY FILES

if not os.path.isfile(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not os.path.isfile(
    DATA_YAML
):

    raise FileNotFoundError(
        f"\nDataset YAML not found:\n{DATA_YAML}"
    )

# PYTORCH FP32 BASELINE

print("\n")
print("=" * 80)
print("PYTORCH FP32 BASELINE")
print("=" * 80)


print(
    "\nOriginal PyTorch checkpoint:"
)

print(
    MODEL_PATH
)


print(
    "\nThe original .pt checkpoint is retained unchanged."
)


print(
    "\nNo conversion is performed for the PyTorch FP32 baseline."
)

# OPTIONAL COPY OF ORIGINAL PYTORCH MODEL

pytorch_output = (
    OUTPUT_DIR /
    "RGB_PyTorch_FP32.pt"
)


shutil.copy2(
    MODEL_PATH,
    pytorch_output
)


print(
    "\nPyTorch FP32 copy saved to:"
)

print(
    pytorch_output
)

# FUNCTION TO MOVE EXPORTED ENGINE

def save_engine(
    exported_path,
    destination
):

    exported_path = Path(
        exported_path
    )


    destination = Path(
        destination
    )


    if exported_path.resolve() == destination.resolve():

        return destination


    if destination.exists():

        destination.unlink()


    shutil.move(
        str(exported_path),
        str(destination)
    )


    return destination


# TENSORRT FP32

print("\n")
print("=" * 80)
print("EXPORTING TENSORRT FP32")
print("=" * 80)


model = YOLO(
    MODEL_PATH
)


fp32_export = model.export(

    format="engine",

    imgsz=IMG_SIZE,

    batch=BATCH_SIZE,

    dynamic=False,

    workspace=WORKSPACE,

    device=0,

    verbose=True
)


fp32_engine = (
    OUTPUT_DIR /
    "RGB_TensorRT_FP32.engine"
)


fp32_engine = save_engine(
    fp32_export,
    fp32_engine
)


print(
    "\nTensorRT FP32 engine saved:"
)

print(
    fp32_engine
)



 # TENSORRT FP16

print("\n")
print("=" * 80)
print("EXPORTING TENSORRT FP16")
print("=" * 80)


model = YOLO(
    MODEL_PATH
)


fp16_export = model.export(

    format="engine",

    imgsz=IMG_SIZE,

    batch=BATCH_SIZE,

    half=True,

    dynamic=False,

    workspace=WORKSPACE,

    device=0,

    verbose=True
)


fp16_engine = (
    OUTPUT_DIR /
    "RGB_TensorRT_FP16.engine"
)


fp16_engine = save_engine(
    fp16_export,
    fp16_engine
)


print(
    "\nTensorRT FP16 engine saved:"
)

print(
    fp16_engine
)

# TENSORRT INT8
# STATIC POST-TRAINING QUANTIZATION
# Representative calibration images are obtained from DATA_YAML.
# split="val":
#   use validation images for calibration.
# fraction=1.0:
#   use the complete validation split.
# TEST DATA ARE NOT USED.

print("\n")
print("=" * 80)
print("EXPORTING TENSORRT INT8")
print("STATIC POST-TRAINING QUANTIZATION")
print("=" * 80)


print(
    "\nCalibration dataset YAML:"
)

print(
    DATA_YAML
)


print(
    "\nCalibration split: validation"
)


print(
    f"Calibration fraction: "
    f"{CALIBRATION_FRACTION}"
)


model = YOLO(
    MODEL_PATH
)


int8_export = model.export(

    format="engine",

    imgsz=IMG_SIZE,

    batch=BATCH_SIZE,

    int8=True,

    data=DATA_YAML,

    split="val",

    fraction=CALIBRATION_FRACTION,

    dynamic=False,

    workspace=WORKSPACE,

    device=0,

    verbose=True
)


int8_engine = (
    OUTPUT_DIR /
    "RGB_TensorRT_INT8.engine"
)


int8_engine = save_engine(
    int8_export,
    int8_engine
)


print(
    "\nTensorRT INT8 engine saved:"
)

print(
    int8_engine
)

# FILE SIZE FUNCTION

def get_file_size_mb(
    file_path
):

    return (
        os.path.getsize(
            file_path
        )
        /
        (1024 * 1024)
    )

# SUMMARY

print("\n")
print("=" * 80)
print("RGB DEPLOYMENT MODEL EXPORT COMPLETED")
print("=" * 80)


print(
    "\n1. PyTorch FP32:"
)

print(
    pytorch_output
)

print(
    f"Size: "
    f"{get_file_size_mb(pytorch_output):.2f} MB"
)


print(
    "\n2. TensorRT FP32:"
)

print(
    fp32_engine
)

print(
    f"Size: "
    f"{get_file_size_mb(fp32_engine):.2f} MB"
)


print(
    "\n3. TensorRT FP16:"
)

print(
    fp16_engine
)

print(
    f"Size: "
    f"{get_file_size_mb(fp16_engine):.2f} MB"
)


print(
    "\n4. TensorRT INT8:"
)

print(
    int8_engine
)

print(
    f"Size: "
    f"{get_file_size_mb(int8_engine):.2f} MB"
)


print("\n")
print("=" * 80)
print("IMPORTANT")
print("=" * 80)


print(
    "\nINT8 calibration used the validation dataset."
)


print(
    "The independent test dataset was not used for calibration."
)


print(
    "\nUse the SAME independent test dataset to evaluate all four "
    "deployment configurations."
)


print("\n")