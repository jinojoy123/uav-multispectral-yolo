# uav_MS_yolo Training Setup

import os
import sys
import torch
print("\nPython executable:")
print(sys.executable)

from pathlib import Path

# SET PATHS

# Path to cloned uav_MS_yolo repository
clone_dir = r"uav_MS_yolo"

# Add modified ultralytics source to Python path
yolo_source_path = os.path.abspath(
    os.path.join(clone_dir, "code", "ultralytics_uav_MS_yolo")
)
# helper scripts location
code_path = os.path.abspath(
    os.path.join(clone_dir, "code")
)
sys.path.insert(0, yolo_source_path)
sys.path.insert(0, code_path)

# IMPORT YOLO

from ultralytics import YOLO

# TRAINING PARAMETERS

# Number of epochs
num_epochs = 100

# Batch size
batch_size = 16

# Early stopping patience
#patience = 50
pretrained_model = '/yolomodel.pt'

# DATASET SETTINGS

# RGBRN dataset path
dataset_path = r"/processed_RGBRN"
os.makedirs(dataset_path, exist_ok=True)
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
# Number of multispectral channels
number_of_channels = 5

# Number of classes
number_of_classes = 2

# Class names
class_names = ['Crop', 'Weed']

# CREATE DATA YAML

output_yaml_RGBRN = os.path.join(
    dataset_path,
    "data_RGBRN.yaml"
)

yaml_path = output_yaml_RGBRN
print(f"\nDataset YAML created:")

# LOAD MODEL

# model = YOLO(patched_model_path)
model = YOLO(pretrained_model)

# DEVICE

device = 0 if torch.cuda.is_available() else 'cpu'

print(f"\nUsing device: {device}")

# START TRAINING

results = model.train(
    data=yaml_path,
    epochs=num_epochs,
    batch=batch_size,
    imgsz=640,
    #patience=patience,
    device=device,
    workers=4,
    deterministic=False,
    plots=True,
    seed = 0,

    # Output
    project=os.path.join(dataset_path, "runs"),
    name="Crop_Weed_RGBRN",

    # Segmentation
    task='segment',

    # Save
    save=True,
    save_period=10,

    # Validation
    val=True,

    # Mixed precision
    amp=True
)

# =============================================================================
# VALIDATION
# =============================================================================

metrics = model.val()

print("\nValidation metrics:")
print(metrics)


