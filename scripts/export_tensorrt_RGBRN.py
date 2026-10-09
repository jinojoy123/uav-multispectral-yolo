# RGBRN YOLO SEGMENTATION MODEL
# TENSORRT EXPORT FOR EDGE DEPLOYMENT
# RGBRN input:
#   R
#   G
#   B
#   Red Edge
#   NIR
# Number of input channels = 5
# Deployment configurations:
#
#   1. PyTorch FP32       -> original best.pt (unchanged)
#   2. TensorRT FP32      -> .engine
#   3. TensorRT FP16      -> .engine
#   4. TensorRT INT8 PTQ  -> .engine
# INT8 uses STATIC POST-TRAINING QUANTIZATION.
# Calibration uses representative 5-channel RGBRN validation imagery.
# Independent TEST data are NOT used for calibration.

import os
import sys
import torch
import tifffile
import shutil

print(
    "\nPython executable:"
)

print(
    sys.executable
)


from pathlib import Path

# SET PATHS

# Path to cloned YOLO-Multispectral repository

clone_dir = r"uav_MS_yolo"


# Add modified ultralytics source to Python path

yolo_source_path = os.path.abspath(

    os.path.join(
        clone_dir,
        "code",
        "ultralytics_MS"
    )

)


# Helper scripts location

code_path = os.path.abspath(

    os.path.join(
        clone_dir,
        "code"
    )

)


sys.path.insert(
    0,
    yolo_source_path
)


sys.path.insert(
    0,
    code_path
)

# IMPORT YOLO

from ultralytics import YOLO

# DATASET SETTINGS

dataset_path = r"/processed_RGBRN"

os.makedirs(
    dataset_path,
    exist_ok=True
)


images_train_path_RGBRN = os.path.join(

    dataset_path,
    "train"

)


images_val_path_RGBRN = os.path.join(

    dataset_path,
    "val"

)


images_test_path_RGBRN = os.path.join(

    dataset_path,
    "test"

)
number_of_channels = 5

number_of_classes = 2

class_names = [

    'Crop',

    'Weed'

]

# CREATE DATA YAML

output_yaml_RGBRN = os.path.join(

    dataset_path,

    "data_RGBRN.yaml"

)


yaml_path = output_yaml_RGBRN


print(
    "\nDataset YAML created:"
)

print(
    yaml_path
)


# USER SETTINGS
# IMPORTANT:
#
# Change this to the RGBRN best.pt checkpoint that you want to export.

MODEL_PATH = os.path.join(

    dataset_path,

    "runs",

    "yolo",

    "weights",

    "best.pt"

)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = Path(

    os.path.join(

        dataset_path,

        "deployment_RGBRN"

    )

)

# EXPORT SETTINGS

IMG_SIZE = 640


BATCH_SIZE = 1


WORKSPACE = None


# Use complete validation dataset for INT8 calibration

CALIBRATION_FRACTION = 1.0

# CREATE OUTPUT DIRECTORY

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)

# DEVICE

device = 0 if torch.cuda.is_available() else 'cpu'


print(
    f"\nUsing device: {device}"
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

        f"\nRGBRN model not found:\n"

        f"{MODEL_PATH}"

    )


if not os.path.isfile(
    yaml_path
):

    raise FileNotFoundError(

        f"\nRGBRN dataset YAML not found:\n"

        f"{yaml_path}"

    )

# PYTORCH FP32 BASELINE

print("\n")
print("=" * 80)
print("PYTORCH FP32 BASELINE")
print("=" * 80)


print(
    "\nOriginal RGBRN PyTorch checkpoint:"
)


print(
    MODEL_PATH
)


print(
    "\nThe original .pt checkpoint is retained unchanged."
)

# COPY PYTORCH FP32 CHECKPOINT

pytorch_output = (

    OUTPUT_DIR /

    "RGBRN_PyTorch_FP32.pt"

)


shutil.copy2(

    MODEL_PATH,

    pytorch_output

)


print(
    "\nPyTorch FP32 copy saved:"
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


    if (

        exported_path.resolve()

        ==

        destination.resolve()

    ):

        return destination


    if destination.exists():

        destination.unlink()


    shutil.move(

        str(exported_path),

        str(destination)

    )


    return destination

# LOAD MODEL AND VERIFY INPUT CHANNELS

print("\n")
print("=" * 80)
print("LOADING RGBRN MODEL")
print("=" * 80)


model = YOLO(

    MODEL_PATH

)


print(
    "\nRGBRN model loaded successfully."
)

# ATTEMPT TO VERIFY FIRST CONVOLUTION INPUT CHANNELS

first_conv_channels = None


for module in model.model.modules():

    if isinstance(
        module,
        torch.nn.Conv2d
    ):

        first_conv_channels = (
            module.in_channels
        )

        break


print(
    f"\nDetected first convolution input channels: "
    f"{first_conv_channels}"
)


if first_conv_channels != 5:

    raise RuntimeError(

        "\nERROR:\n"

        f"The loaded model reports "
        f"{first_conv_channels} input channels.\n"

        "Expected 5 channels for RGBRN.\n\n"

        "Do not continue TensorRT export until the "
        "5-channel model is loaded correctly."

    )


print(
    "\n5-channel RGBRN model verified."
)

# TENSORRT FP32

print("\n")
print("=" * 80)
print("EXPORTING RGBRN TENSORRT FP32")
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

    "RGBRN_TensorRT_FP32.engine"

)


fp32_engine = save_engine(

    fp32_export,

    fp32_engine

)


print(
    "\nRGBRN TensorRT FP32 engine saved:"
)


print(
    fp32_engine
)


# TENSORRT FP16

print("\n")
print("=" * 80)
print("EXPORTING RGBRN TENSORRT FP16")
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

    "RGBRN_TensorRT_FP16.engine"

)


fp16_engine = save_engine(

    fp16_export,

    fp16_engine

)


print(
    "\nRGBRN TensorRT FP16 engine saved:"
)


print(
    fp16_engine
)

# TENSORRT INT8
# STATIC POST-TRAINING QUANTIZATION
# The custom RGBRN data loader must provide:
#       B x 5 x 640 x 640
# tensors during calibration.
# The validation split is used as representative calibration data.
# The independent test dataset is NOT used.

print("\n")
print("=" * 80)
print("EXPORTING RGBRN TENSORRT INT8")
print("STATIC POST-TRAINING QUANTIZATION")
print("=" * 80)


print(
    "\nCalibration YAML:"
)


print(
    yaml_path
)


print(
    "\nExpected calibration input channels: 5"
)


print(
    "\nCalibration split: validation"
)


print(
    f"\nCalibration fraction: "
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

    data=yaml_path,

    split="val",

    fraction=CALIBRATION_FRACTION,

    dynamic=False,

    workspace=WORKSPACE,

    device=0,

    verbose=True

)


int8_engine = (

    OUTPUT_DIR /

    "RGBRN_TensorRT_INT8.engine"

)


int8_engine = save_engine(

    int8_export,

    int8_engine

)


print(
    "\nRGBRN TensorRT INT8 engine saved:"
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

# FINAL SUMMARY

print("\n")
print("=" * 80)
print("RGBRN DEPLOYMENT MODEL EXPORT COMPLETED")
print("=" * 80)


print(
    "\nInput channels: 5"
)


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
print("INT8 CALIBRATION")
print("=" * 80)


print(
    "\nMethod:"
)


print(
    "Static Post-Training Quantization (PTQ)"
)


print(
    "\nCalibration dataset:"
)


print(
    "RGBRN validation dataset"
)


print(
    "\nCalibration input:"
)


print(
    "R + G + B + Red Edge + NIR"
)


print(
    "\nIndependent test data were NOT used for calibration."
)


print("\n")
print("=" * 80)