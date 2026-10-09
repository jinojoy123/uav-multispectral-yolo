# YOLO INSTANCE SEGMENTATION - INDEPENDENT TEST EVALUATION
#
# Outputs:
#   1. Prediction visualization
#       Crop      = GREEN
#       Weed      = RED
#
#   2. Ground-truth visualization
#       Crop      = GREEN
#       Weed      = RED
#
#   3. Predicted total instances
#   4. Ground-truth total instances
#   5. Per-class IoU
#   6. Overall IoU
#   7. Mask mAP50
#   8. Per-image CSV
#   9. Overall metrics CSV
# =============================================================================

from ultralytics import YOLO

import os
import cv2
import yaml
import numpy as np
import pandas as pd

from pathlib import Path

# USER SETTINGS

# Trained model
MODEL_PATH = "/weights/best.pt"

# Dataset YAML
DATA_YAML = "dataset.yaml"

# Output directory
OUTPUT_DIR = "test_evaluation_RGB"

# Inference settings
IMG_SIZE = 640

# IMPORTANT:
# Use the same confidence threshold for ALL models being compared.
CONF_THRESHOLD = 0.5

IOU_NMS_THRESHOLD = 0.70

DEVICE = 0

# CLASS DEFINITIONS

# IMPORTANT:
# These must correspond to the class IDs in dataset.yaml.
#
# Assumption:
# 0 = Crop
# 1 = Weed

CLASS_NAMES = {
    0: "Crop",
    1: "Weed"
}


# OpenCV uses BGR colors
COLORS = {
    0: (0, 255, 0),       # Crop = GREEN
    1: (0, 0, 255)        # Weed = RED
}


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

output_dir = Path(OUTPUT_DIR)

prediction_dir = output_dir / "predictions"
groundtruth_dir = output_dir / "ground_truth"
comparison_dir = output_dir / "comparison"

prediction_dir.mkdir(
    parents=True,
    exist_ok=True
)

groundtruth_dir.mkdir(
    parents=True,
    exist_ok=True
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# LOAD DATASET YAML

with open(DATA_YAML, "r") as f:
    dataset_yaml = yaml.safe_load(f)


if "test" not in dataset_yaml:
    raise ValueError(
        "\nERROR: dataset.yaml does not contain a 'test:' entry.\n"
        "Please add the independent test image path to dataset.yaml."
    )

# RESOLVE TEST IMAGE PATH

yaml_directory = Path(DATA_YAML).resolve().parent


if "path" in dataset_yaml and dataset_yaml["path"] is not None:

    dataset_root = Path(dataset_yaml["path"])

    if not dataset_root.is_absolute():
        dataset_root = (
            yaml_directory /
            dataset_root
        ).resolve()

else:

    dataset_root = yaml_directory


test_entry = Path(dataset_yaml["test"])


if test_entry.is_absolute():

    test_image_dir = test_entry

else:

    test_image_dir = (
        dataset_root /
        test_entry
    ).resolve()


print("\nTest image directory:")
print(test_image_dir)


# FIND TEST LABEL DIRECTORY

# Standard YOLO structure:
#
# dataset/
# ├── images/
# │   └── test/
# └── labels/
#     └── test/
#

test_path_string = str(test_image_dir)

if (
    f"{os.sep}images{os.sep}" in test_path_string
):

    test_label_dir = Path(
        test_path_string.replace(
            f"{os.sep}images{os.sep}",
            f"{os.sep}labels{os.sep}"
        )
    )

else:

    raise ValueError(
        "\nCould not automatically determine test label folder.\n"
        "Expected standard YOLO structure:\n"
        "images/test and labels/test"
    )


print("\nTest label directory:")
print(test_label_dir)


# FIND TEST IMAGES

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.tif",
    "*.tiff",
    "*.JPG"
]


image_files = []

for extension in image_extensions:

    image_files.extend(
        test_image_dir.glob(extension)
    )


image_files = sorted(image_files)


print(
    f"\nNumber of test images: "
    f"{len(image_files)}"
)


if len(image_files) == 0:

    raise RuntimeError(
        "No test images were found."
    )

# LOAD MODEL

print("\nLoading model...")

model = YOLO(
    MODEL_PATH
)

# GLOBAL COUNTERS

predicted_counts = {
    0: 0,
    1: 0
}

groundtruth_counts = {
    0: 0,
    1: 0
}


# Intersection / union accumulated across dataset
intersection_total = {
    0: 0,
    1: 0
}

union_total = {
    0: 0,
    1: 0
}


# Store per-image results
per_image_results = []


# FUNCTION:
# YOLO SEGMENTATION POLYGON -> BINARY MASK

def polygon_to_mask(
    polygon_points,
    width,
    height
):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    polygon_points = np.array(
        polygon_points,
        dtype=np.float32
    )

    polygon_points[:, 0] *= width
    polygon_points[:, 1] *= height

    polygon_points = polygon_points.astype(
        np.int32
    )

    cv2.fillPoly(
        mask,
        [polygon_points],
        1
    )

    return mask

# FUNCTION:
# DRAW MASK

def overlay_mask(
    image,
    mask,
    color,
    alpha=0.40
):

    overlay = image.copy()

    overlay[mask > 0] = color

    blended = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0
    )

    image[mask > 0] = blended[mask > 0]

    return image

# PROCESS EACH TEST IMAGE

print("\nRunning test inference...\n")


for image_index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"[{image_index}/{len(image_files)}] "
        f"{image_path.name}"
    )


    # READ ORIGINAL IMAGE

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "WARNING: Could not read image:",
            image_path
        )

        continue


    height, width = image.shape[:2]


    prediction_visualization = (
        image.copy()
    )

    groundtruth_visualization = (
        image.copy()
    )


    # CREATE CLASS MASKS

    gt_masks = {
        0: np.zeros(
            (height, width),
            dtype=np.uint8
        ),

        1: np.zeros(
            (height, width),
            dtype=np.uint8
        )
    }


    pred_masks = {
        0: np.zeros(
            (height, width),
            dtype=np.uint8
        ),

        1: np.zeros(
            (height, width),
            dtype=np.uint8
        )
    }


    # PER-IMAGE COUNTS

    image_gt_counts = {
        0: 0,
        1: 0
    }

    image_pred_counts = {
        0: 0,
        1: 0
    }

    # LOAD GROUND TRUTH LABEL

    label_path = (
        test_label_dir /
        f"{image_path.stem}.txt"
    )


    if label_path.exists():

        with open(
            label_path,
            "r"
        ) as f:

            lines = f.readlines()


        for line in lines:

            parts = line.strip().split()

            if len(parts) < 7:
                continue


            class_id = int(
                float(parts[0])
            )


            if class_id not in CLASS_NAMES:
                continue


            coordinates = np.array(
                list(
                    map(
                        float,
                        parts[1:]
                    )
                )
            )


            coordinates = coordinates.reshape(
                -1,
                2
            )


            gt_mask_instance = polygon_to_mask(
                coordinates,
                width,
                height
            )


            # Combine all GT instances
            # belonging to this class
            gt_masks[class_id] = np.maximum(
                gt_masks[class_id],
                gt_mask_instance
            )


            # Count instance
            groundtruth_counts[class_id] += 1

            image_gt_counts[class_id] += 1

            # DRAW GT MASK

            groundtruth_visualization = (
                overlay_mask(
                    groundtruth_visualization,
                    gt_mask_instance,
                    COLORS[class_id]
                )
            )


            # Polygon boundary
            polygon_pixels = coordinates.copy()

            polygon_pixels[:, 0] *= width
            polygon_pixels[:, 1] *= height

            polygon_pixels = polygon_pixels.astype(
                np.int32
            )


            cv2.polylines(
                groundtruth_visualization,
                [polygon_pixels],
                True,
                COLORS[class_id],
                2
            )

    # MODEL INFERENCE

    results = model.predict(

        source=str(image_path),

        imgsz=IMG_SIZE,

        conf=CONF_THRESHOLD,

        iou=IOU_NMS_THRESHOLD,

        device=DEVICE,

        verbose=False
    )


    result = results[0]

    # PROCESS PREDICTIONS

    if (
        result.masks is not None
        and
        result.boxes is not None
    ):

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )


        # Polygon masks in original image coordinates
        polygons = result.masks.xy


        for class_id, polygon in zip(
            classes,
            polygons
        ):

            if class_id not in CLASS_NAMES:
                continue


            if polygon is None:
                continue


            if len(polygon) < 3:
                continue


            # COUNT PREDICTED INSTANCE

            predicted_counts[class_id] += 1

            image_pred_counts[class_id] += 1


            # CREATE PREDICTED MASK

            polygon_pixels = np.array(
                polygon,
                dtype=np.int32
            )


            instance_mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )


            cv2.fillPoly(
                instance_mask,
                [polygon_pixels],
                1
            )


            pred_masks[class_id] = np.maximum(
                pred_masks[class_id],
                instance_mask
            )

            # DRAW PREDICTION

            prediction_visualization = (
                overlay_mask(
                    prediction_visualization,
                    instance_mask,
                    COLORS[class_id]
                )
            )


            cv2.polylines(
                prediction_visualization,
                [polygon_pixels],
                True,
                COLORS[class_id],
                2
            )

    # CALCULATE IoU FOR THIS IMAGE

    image_iou = {}


    for class_id in CLASS_NAMES:

        gt_mask = gt_masks[class_id].astype(
            bool
        )

        pred_mask = pred_masks[class_id].astype(
            bool
        )


        intersection = np.logical_and(
            gt_mask,
            pred_mask
        ).sum()


        union = np.logical_or(
            gt_mask,
            pred_mask
        ).sum()


        intersection_total[class_id] += (
            intersection
        )

        union_total[class_id] += union


        if union > 0:

            iou = intersection / union

        else:

            iou = np.nan


        image_iou[class_id] = iou

    # SAVE PREDICTION VISUALIZATION

    cv2.imwrite(

        str(
            prediction_dir /
            image_path.name
        ),

        prediction_visualization
    )

    # SAVE GROUND-TRUTH VISUALIZATION

    cv2.imwrite(

        str(
            groundtruth_dir /
            image_path.name
        ),

        groundtruth_visualization
    )

    # CREATE SIDE-BY-SIDE COMPARISON

    comparison = np.hstack(
        (
            groundtruth_visualization,
            prediction_visualization
        )
    )


    cv2.imwrite(

        str(
            comparison_dir /
            image_path.name
        ),

        comparison
    )

    # STORE PER-IMAGE RESULTS

    per_image_results.append(
        {

            "image":
                image_path.name,

            "GT_Crop":
                image_gt_counts[0],

            "Pred_Crop":
                image_pred_counts[0],

            "GT_Weed":
                image_gt_counts[1],

            "Pred_Weed":
                image_pred_counts[1],

            "IoU_Crop":
                image_iou[0],

            "IoU_Weed":
                image_iou[1]

        }
    )

# DATASET-LEVEL IoU

dataset_iou = {}


for class_id in CLASS_NAMES:

    if union_total[class_id] > 0:

        dataset_iou[class_id] = (

            intersection_total[class_id]
            /
            union_total[class_id]

        )

    else:

        dataset_iou[class_id] = np.nan

# Overall combined IoU

total_intersection = sum(
    intersection_total.values()
)

total_union = sum(
    union_total.values()
)


if total_union > 0:

    overall_iou = (
        total_intersection /
        total_union
    )

else:

    overall_iou = np.nan

# STANDARD ULTRALYTICS TEST METRICS

print(
    "\nRunning Ultralytics test-set evaluation..."
)


metrics = model.val(

    data=DATA_YAML,

    split="test",

    imgsz=IMG_SIZE,

    device=DEVICE,

    plots=True,

    verbose=True
)

# EXTRACT SEGMENTATION METRICS

mask_map50 = metrics.seg.map50
mask_map5095 = metrics.seg.map

mask_precision = metrics.seg.mp
mask_recall = metrics.seg.mr

# SAVE PER-IMAGE RESULTS

per_image_df = pd.DataFrame(
    per_image_results
)


per_image_csv = (
    output_dir /
    "per_image_metrics.csv"
)


per_image_df.to_csv(
    per_image_csv,
    index=False
)

# CREATE OVERALL METRICS TABLE

overall_results = pd.DataFrame(
    [
        {

            "Mask_mAP50":
                mask_map50,

            "Mask_mAP50_95":
                mask_map5095,

            "Mask_Precision":
                mask_precision,

            "Mask_Recall":
                mask_recall,

            "IoU_Crop":
                dataset_iou[0],

            "IoU_Weed":
                dataset_iou[1],

            "Overall_IoU":
                overall_iou,

            "GT_Crop":
                groundtruth_counts[0],

            "Pred_Crop":
                predicted_counts[0],

            "GT_Weed":
                groundtruth_counts[1],

            "Pred_Weed":
                predicted_counts[1],

            "Total_GT_Instances":
                (
                    groundtruth_counts[0]
                    +
                    groundtruth_counts[1]
                ),

            "Total_Predicted_Instances":
                (
                    predicted_counts[0]
                    +
                    predicted_counts[1]
                )

        }
    ]
)


overall_csv = (
    output_dir /
    "overall_metrics.csv"
)


overall_results.to_csv(
    overall_csv,
    index=False
)

# PRINT FINAL RESULTS

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)


print("\nSEGMENTATION METRICS")

print(
    f"Mask mAP50      : "
    f"{mask_map50:.4f}"
)

print(
    f"Mask mAP50-95   : "
    f"{mask_map5095:.4f}"
)

print(
    f"Mask Precision  : "
    f"{mask_precision:.4f}"
)

print(
    f"Mask Recall     : "
    f"{mask_recall:.4f}"
)


print("\nIoU")

print(
    f"Crop IoU   : "
    f"{dataset_iou[0]:.4f}"
)

print(
    f"Weed IoU        : "
    f"{dataset_iou[1]:.4f}"
)

print(
    f"Overall IoU     : "
    f"{overall_iou:.4f}"
)


print("\nGROUND-TRUTH INSTANCE COUNTS")

print(
    f"Crop       : "
    f"{groundtruth_counts[0]}"
)

print(
    f"Weed            : "
    f"{groundtruth_counts[1]}"
)

print(
    f"Total           : "
    f"{sum(groundtruth_counts.values())}"
)


print("\nPREDICTED INSTANCE COUNTS")

print(
    f"Crop       : "
    f"{predicted_counts[0]}"
)

print(
    f"Weed            : "
    f"{predicted_counts[1]}"
)

print(
    f"Total           : "
    f"{sum(predicted_counts.values())}"
)


print("\nOUTPUT FILES")

print(
    "Prediction visualizations :",
    prediction_dir
)

print(
    "Ground-truth visualizations:",
    groundtruth_dir
)

print(
    "Side-by-side comparisons  :",
    comparison_dir
)

print(
    "Per-image metrics         :",
    per_image_csv
)

print(
    "Overall metrics           :",
    overall_csv
)


print("\n")
print("=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)