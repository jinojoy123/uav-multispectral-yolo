# Prepare RGBREN Dataset for YOLO Training

import os
import cv2
import shutil
import tifffile
import numpy as np

from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split


# Your dataset root
dataset_root = "data"

# Input folders
images_dir = os.path.join(dataset_root, "images")
labels_dir = os.path.join(dataset_root, "labels")

# Output folders
output_rgbren_dir = os.path.join(dataset_root, "processed_RGBREN")

# Classes
classes_list = ['Crop', 'Weed']
number_of_classes = len(classes_list)

# Image size
# clip_size = 640

# uint16 -> uint8 conversion
uint16_to_uint8_method = 'normalize'

# CREATE OUTPUT FOLDERS

for base_dir in [output_rgbren_dir]:

    for split in ['train', 'val', 'test']:

        os.makedirs(
            os.path.join(base_dir, split, 'images'),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(base_dir, split, 'labels'),
            exist_ok=True
        )

# FIND IMAGE PREFIXES

red_files = sorted(glob(os.path.join(images_dir, '*_RED.tif')))

prefixes = []

for red_path in red_files:

    prefix = os.path.basename(red_path).replace('_RED.tif', '')

    prefixes.append(prefix)

print(f"Found {len(prefixes)} multispectral image groups")

# TRAIN / VAL / TEST SPLIT

train_files, temp_files = train_test_split(
    prefixes,
    test_size=0.2,
    random_state=42
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    random_state=42
)

print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Test: {len(test_files)}")

def load_single_band_tiff(path):

    img = tifffile.imread(path)

    img = np.asarray(img)

    print(f"Loaded: {os.path.basename(path)} -> {img.shape}")

    # If TIFF has multiple channels, keep first channel only
    if img.ndim == 3:

        img = img[:, :, 0]

        print(f"Converted to single band -> {img.shape}")

    return img

# PROCESS EACH IMAGE GROUP

def convert_uint16_to_uint8(img):

    # Ensure image is numpy array
    img = np.asarray(img)

    # If grayscale image
    if img.ndim == 2:

        img = cv2.normalize(
            img,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        return img

    # If multi-channel image
    elif img.ndim == 3:

        img_uint8 = np.zeros(img.shape, dtype=np.uint8)

        for c in range(img.shape[2]):

            band = img[:, :, c]

            band_norm = cv2.normalize(
                band,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            img_uint8[:, :, c] = band_norm.astype(np.uint8)

        return img_uint8

    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

# SAVE RGBREN

def process_and_save(prefix, split_name):

    # Load spectral bands
    
    red = load_single_band_tiff(
        os.path.join(images_dir, f'{prefix}_RED.tif')
    )

    green = load_single_band_tiff(
        os.path.join(images_dir, f'{prefix}_GREEN.tif')
    )

    blue = load_single_band_tiff(
        os.path.join(images_dir, f'{prefix}_BLUE.tif')
    )

    rededge = load_single_band_tiff(
        os.path.join(images_dir, f'{prefix}_REDEDGE.tif')
    )

    nir = load_single_band_tiff(
        os.path.join(images_dir, f'{prefix}_NIR.tif')
    )
    print("RED:", red.shape)
    print("GREEN:", green.shape)
    print("BLUE:", blue.shape)
    print("REDEDGE:", rededge.shape)
    print("NIR:", nir.shape)

    # RGBREN STACK

    rgbren = np.stack([
        red,
        green,
        blue,
        rededge,
        nir
    ], axis=-1)

    rgbren = convert_uint16_to_uint8(rgbren)
    print("RGBREN stacked shape:", rgbren.shape)

    # resized_channels = []

    # for c in range(rgbren.shape[2]):

    #     resized_band = cv2.resize(
    #         rgbren[:, :, c],
    #         (clip_size, clip_size)
    #     )

    #     resized_channels.append(resized_band)

    # rgbren = np.stack(resized_channels, axis=-1)

    # Save RGBREN
    rgbren_out_path = os.path.join(
        output_rgbren_dir,
        split_name,
        'images',
        f'{prefix}.tif'
    )

    tifffile.imwrite(rgbren_out_path, rgbren)

    # COPY LABEL

    label_src = os.path.join(
        labels_dir,
        f'{prefix}.txt'
    )

    if os.path.exists(label_src):

        shutil.copy(
            label_src,
            os.path.join(
                output_rgbren_dir,
                split_name,
                'labels',
                f'{prefix}.txt'
            )
        )

# PROCESS DATASETS

print("\nProcessing TRAIN set...")

for prefix in train_files:
    process_and_save(prefix, 'train')

print("\nProcessing VAL set...")

for prefix in val_files:
    process_and_save(prefix, 'val')

print("\nProcessing TEST set...")

for prefix in test_files:
    process_and_save(prefix, 'test')

# FINAL OUTPUT

print("\nDataset preparation completed.")


print("\nRGBREN Dataset:")
print(output_rgbren_dir)