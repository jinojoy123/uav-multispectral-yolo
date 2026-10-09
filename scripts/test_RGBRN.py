# uav_MS_yolo RGBRN - INDEPENDENT TEST EVALUATION
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
#   3. Side-by-side GT vs Prediction images
#   4. Predicted total instances
#   5. Ground-truth total instances
#   6. Dataset-level IoU
#       - Crop IoU
#       - Weed IoU
#       - Overall IoU
#   7. Ultralytics segmentation metrics
#       - Mask mAP50
#       - Mask mAP50-95
#       - Precision
#       - Recall
#   8. Per-image CSV
#   9. Overall metrics CSV

# uav_MS_yolo Setup

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
    os.path.join(clone_dir, "code", "uav_MS_yolo")
)

# helper scripts location
code_path = os.path.abspath(
    os.path.join(clone_dir, "code")
)

sys.path.insert(0, yolo_source_path)
sys.path.insert(0, code_path)

# IMPORT YOLO

from ultralytics import YOLO

# ADDITIONAL IMPORTS FOR TEST EVALUATION

import cv2
import yaml
import numpy as np
import pandas as pd
import tifffile

# DATASET SETTINGS

# RGBRN dataset path
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


# Number of multispectral channels
number_of_channels = 5

# Number of classes
number_of_classes = 2

# Class names
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

print("\nDataset YAML created:")
print(yaml_path)

# TEST SETTINGS

# IMPORTANT:
# Change this to the best.pt file that you want to test.

MODEL_PATH = os.path.join(
    dataset_path,
    "runs",
    "yolo",
    "weights",
    "best.pt"
)


# Output folder
OUTPUT_DIR = os.path.join(
    dataset_path,
    "test_evaluation_RGBRN"
)


# Inference settings
IMG_SIZE = 640

# Use EXACTLY the same confidence threshold
# for every RGB and RGBRN model.
CONF_THRESHOLD = 0.5

IOU_NMS_THRESHOLD = 0.70

# DEVICE

device = 0 if torch.cuda.is_available() else 'cpu'

print(f"\nUsing device: {device}")

# CLASS DEFINITIONS

# Must correspond exactly to data_RGBRN.yaml
#
# 0 = Crop
# 1 = Weed

CLASS_NAMES = {
    0: "Crop",
    1: "Weed"
}


# OpenCV uses BGR
COLORS = {

    # Crop = GREEN
    0: (0, 255, 0),

    # Weed = RED
    1: (0, 0, 255)
}

# CREATE OUTPUT DIRECTORIES

output_dir = Path(
    OUTPUT_DIR
)

prediction_dir = (
    output_dir /
    "predictions"
)

groundtruth_dir = (
    output_dir /
    "ground_truth"
)

comparison_dir = (
    output_dir /
    "comparison"
)


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

# DEFINE TEST IMAGE AND LABEL FOLDERS

# Based directly on your RGBRN dataset structure:
#
# processed_RGBRN/
# └── test/
#     ├── images/
#     └── labels/
#

test_image_dir = Path(
    images_test_path_RGBRN
) / "images"

test_label_dir = Path(
    images_test_path_RGBRN
) / "labels"


print("\nTest image directory:")
print(test_image_dir)

print("\nTest label directory:")
print(test_label_dir)


# VERIFY DIRECTORIES


if not test_image_dir.exists():

    raise FileNotFoundError(
        f"\nTest image directory does not exist:\n"
        f"{test_image_dir}"
    )


if not test_label_dir.exists():

    raise FileNotFoundError(
        f"\nTest label directory does not exist:\n"
        f"{test_label_dir}"
    )

# FIND TEST IMAGES

image_extensions = [

    "*.tif",
    "*.tiff",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.JPG"

]


image_files = []


for extension in image_extensions:

    image_files.extend(
        test_image_dir.glob(extension)
    )


image_files = sorted(
    image_files
)


print(
    f"\nNumber of test images: "
    f"{len(image_files)}"
)


if len(image_files) == 0:

    raise RuntimeError(
        "\nNo test images were found."
    )

# LOAD TRAINED RGBRN MODEL

print("\nLoading trained RGBRN model:")

print(MODEL_PATH)


model = YOLO(
    MODEL_PATH
)


print("\nModel loaded successfully.")

# GLOBAL INSTANCE COUNTERS

predicted_counts = {

    0: 0,
    1: 0

}


groundtruth_counts = {

    0: 0,
    1: 0

}


# =============================================================================
# GLOBAL IoU ACCUMULATORS
# =============================================================================

intersection_total = {

    0: 0,
    1: 0

}


union_total = {

    0: 0,
    1: 0

}

# PER-IMAGE RESULTS

per_image_results = []


# FUNCTION:
# YOLO NORMALIZED SEGMENTATION POLYGON -> MASK

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
# DRAW COLORED MASK

def overlay_mask(
    image,
    mask,
    color,
    alpha=0.40
):

    overlay = image.copy()


    overlay[
        mask > 0
    ] = color


    blended = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0
    )


    image[
        mask > 0
    ] = blended[
        mask > 0
    ]


    return image


# FUNCTION:
# CONVERT MULTISPECTRAL IMAGE TO RGB FOR VISUALIZATION

def load_rgb_visualization(
    image_path
):

    """
    Loads the 5-band RGBRN image and extracts
    the first three channels for visualization.

    Assumed channel order:
        R, G, B, RedEdge, NIR

    Model inference still uses the complete
    5-channel image.
    """


    suffix = (
        image_path
        .suffix
        .lower()
    )

    # TIFF / TIF

    if suffix in [
        ".tif",
        ".tiff"
    ]:

        image_ms = tifffile.imread(
            str(image_path)
        )

        # Expected:
        # H x W x 5

        if (
            image_ms.ndim == 3
            and
            image_ms.shape[-1] >= 5
        ):

            rgb = image_ms[
                :, :, :3
            ]

        # Alternative:
        # 5 x H x W

        elif (
            image_ms.ndim == 3
            and
            image_ms.shape[0] >= 5
        ):

            rgb = np.transpose(
                image_ms[:3, :, :],
                (1, 2, 0)
            )


        else:

            raise ValueError(
                f"\nUnexpected multispectral shape "
                f"for {image_path.name}: "
                f"{image_ms.shape}"
            )

        # Convert to uint8 ONLY for visualization

        if rgb.dtype != np.uint8:

            rgb_float = (
                rgb.astype(
                    np.float32
                )
            )


            # Percentile stretching
            # visualization only
            rgb_vis = np.zeros_like(
                rgb_float,
                dtype=np.float32
            )


            for channel in range(3):

                band = rgb_float[
                    :, :, channel
                ]


                low = np.percentile(
                    band,
                    2
                )

                high = np.percentile(
                    band,
                    98
                )


                if high > low:

                    band = (
                        band - low
                    ) / (
                        high - low
                    )

                else:

                    band = np.zeros_like(
                        band
                    )


                band = np.clip(
                    band,
                    0,
                    1
                )


                rgb_vis[
                    :, :, channel
                ] = band


            rgb = (
                rgb_vis * 255
            ).astype(
                np.uint8
            )


        # RGB -> BGR for OpenCV
        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )


        return rgb

    # Standard image

    else:

        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            raise ValueError(
                f"Could not read "
                f"{image_path}"
            )


        return image

# PROCESS EACH TEST IMAGE

print("\nRunning RGBRN test inference...\n")


for image_index, image_path in enumerate(
    image_files,
    start=1
):


    print(
        f"[{image_index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    # LOAD RGB VERSION FOR VISUALIZATION ONLY

    image = load_rgb_visualization(
        image_path
    )


    height, width = (
        image.shape[:2]
    )


    prediction_visualization = (
        image.copy()
    )


    groundtruth_visualization = (
        image.copy()
    )

    # CREATE EMPTY CLASS MASKS

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

    # PER-IMAGE INSTANCE COUNTERS

    image_gt_counts = {

        0: 0,
        1: 0

    }


    image_pred_counts = {

        0: 0,
        1: 0

    }

    # GROUND-TRUTH LABEL FILE

    label_path = (
        test_label_dir /
        f"{image_path.stem}.txt"
    )


    if label_path.exists():


        with open(
            label_path,
            "r"
        ) as f:

            lines = (
                f.readlines()
            )


        for line in lines:


            parts = (
                line
                .strip()
                .split()
            )


            # Segmentation polygon requires:
            #
            # class x1 y1 x2 y2 x3 y3 ...
            #
            if len(parts) < 7:

                continue


            class_id = int(
                float(
                    parts[0]
                )
            )


            if (
                class_id
                not in CLASS_NAMES
            ):

                continue


            coordinates = np.array(

                list(
                    map(
                        float,
                        parts[1:]
                    )
                )

            )


            coordinates = (
                coordinates.reshape(
                    -1,
                    2
                )
            )


            gt_instance_mask = (
                polygon_to_mask(
                    coordinates,
                    width,
                    height
                )
            )

            # Add to class-level GT mask

            gt_masks[
                class_id
            ] = np.maximum(

                gt_masks[
                    class_id
                ],

                gt_instance_mask

            )

            # Count GT instance

            groundtruth_counts[
                class_id
            ] += 1


            image_gt_counts[
                class_id
            ] += 1

            # Draw GT mask

            groundtruth_visualization = (
                overlay_mask(

                    groundtruth_visualization,

                    gt_instance_mask,

                    COLORS[
                        class_id
                    ]

                )
            )

            # Draw polygon boundary

            polygon_pixels = (
                coordinates.copy()
            )


            polygon_pixels[
                :, 0
            ] *= width


            polygon_pixels[
                :, 1
            ] *= height


            polygon_pixels = (
                polygon_pixels.astype(
                    np.int32
                )
            )


            cv2.polylines(

                groundtruth_visualization,

                [
                    polygon_pixels
                ],

                True,

                COLORS[
                    class_id
                ],

                2

            )


    else:

        print(
            "WARNING: Missing label:",
            label_path
        )

    # RGBRN MODEL INFERENCE
    #
    # IMPORTANT:
    #
    # image_path points to the ORIGINAL 5-channel image.
    #
    # We DO NOT send the 3-channel visualization image
    # to the model.
    #
    # Your modified uav_MS_ultralytics loader receives the
    # original multispectral image.

    results = model.predict(

        source=str(
            image_path
        ),

        imgsz=IMG_SIZE,

        conf=CONF_THRESHOLD,

        iou=IOU_NMS_THRESHOLD,

        device=device,

        verbose=False

    )


    result = results[0]

    # PROCESS PREDICTED INSTANCE MASKS

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


        # result.masks.xy returns polygon coordinates
        # in ORIGINAL image coordinates

        polygons = (
            result.masks.xy
        )


        for class_id, polygon in zip(
            classes,
            polygons
        ):


            if (
                class_id
                not in CLASS_NAMES
            ):

                continue


            if polygon is None:

                continue


            if len(polygon) < 3:

                continue


            # Count predicted instance

            predicted_counts[
                class_id
            ] += 1


            image_pred_counts[
                class_id
            ] += 1

            # Create predicted binary mask

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
                [
                    polygon_pixels
                ],
                1
            )

            # Combine all predicted masks belonging
            # to the same class

            pred_masks[
                class_id
            ] = np.maximum(

                pred_masks[
                    class_id
                ],

                instance_mask

            )

            # Draw prediction

            prediction_visualization = (
                overlay_mask(

                    prediction_visualization,

                    instance_mask,

                    COLORS[
                        class_id
                    ]

                )
            )


            cv2.polylines(

                prediction_visualization,

                [
                    polygon_pixels
                ],

                True,

                COLORS[
                    class_id
                ],

                2

            )

    # CALCULATE PER-IMAGE IoU

    image_iou = {}


    for class_id in CLASS_NAMES:


        gt_mask = (
            gt_masks[
                class_id
            ].astype(bool)
        )


        pred_mask = (
            pred_masks[
                class_id
            ].astype(bool)
        )


        intersection = np.logical_and(
            gt_mask,
            pred_mask
        ).sum()


        union = np.logical_or(
            gt_mask,
            pred_mask
        ).sum()


        intersection_total[
            class_id
        ] += intersection


        union_total[
            class_id
        ] += union


        if union > 0:

            iou = (
                intersection
                /
                union
            )

        else:

            iou = np.nan


        image_iou[
            class_id
        ] = iou

    # SAVE PREDICTION VISUALIZATION

    prediction_output_path = (

        prediction_dir /
        f"{image_path.stem}.jpg"

    )


    cv2.imwrite(

        str(
            prediction_output_path
        ),

        prediction_visualization

    )

    # SAVE GROUND-TRUTH VISUALIZATION

    groundtruth_output_path = (

        groundtruth_dir /
        f"{image_path.stem}.jpg"

    )


    cv2.imwrite(

        str(
            groundtruth_output_path
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


    comparison_output_path = (

        comparison_dir /
        f"{image_path.stem}.jpg"

    )


    cv2.imwrite(

        str(
            comparison_output_path
        ),

        comparison

    )

    # SAVE PER-IMAGE RESULTS

    per_image_results.append(
        {

            "Image":
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


    if (
        union_total[
            class_id
        ] > 0
    ):

        dataset_iou[
            class_id
        ] = (

            intersection_total[
                class_id
            ]

            /

            union_total[
                class_id
            ]

        )

    else:

        dataset_iou[
            class_id
        ] = np.nan

# OVERALL COMBINED IoU

total_intersection = sum(
    intersection_total.values()
)


total_union = sum(
    union_total.values()
)


if total_union > 0:

    overall_iou = (
        total_intersection
        /
        total_union
    )

else:

    overall_iou = np.nan

# STANDARD ULTRALYTICS TEST METRICS

print(
    "\nRunning Ultralytics RGBRN test-set evaluation..."
)


metrics = model.val(

    data=yaml_path,

    split="test",

    imgsz=IMG_SIZE,

    device=device,

    plots=True,

    verbose=True

)

# EXTRACT SEGMENTATION METRICS

mask_map50 = (
    metrics.seg.map50
)

mask_map5095 = (
    metrics.seg.map
)

mask_precision = (
    metrics.seg.mp
)

mask_recall = (
    metrics.seg.mr
)

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
print("=" * 75)
print("FINAL RGBRN TEST RESULTS")
print("=" * 75)


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
    "Prediction visualizations  :",
    prediction_dir
)


print(
    "Ground-truth visualizations:",
    groundtruth_dir
)


print(
    "Side-by-side comparisons   :",
    comparison_dir
)


print(
    "Per-image metrics          :",
    per_image_csv
)


print(
    "Overall metrics            :",
    overall_csv
)


print("\n")
print("=" * 75)
print("RGBRN TEST EVALUATION COMPLETED")
print("=" * 75)