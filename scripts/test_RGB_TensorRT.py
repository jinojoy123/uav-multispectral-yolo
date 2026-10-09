# YOLO INSTANCE SEGMENTATION - MULTI-FORMAT INDEPENDENT TEST EVALUATION
#
# Supports:
#   - PyTorch (.pt)
#   - TensorRT (.engine), including FP32 / FP16 / INT8 engines
#
# For every model, outputs:
#   1. Prediction visualization
#       Crop     = GREEN
#       Weed      = RED
#   2. Ground-truth visualization
#   3. Side-by-side comparison
#   4. Predicted / ground-truth instance counts
#   5. Per-class and overall semantic-mask IoU
#   6. Ultralytics segmentation metrics:
#       mAP50, mAP50-95, Precision, Recall
#   7. Timing:
#       preprocess, inference, postprocess, total pipeline time
#   8. FPS based on inference-only time and total pipeline time
#   9. Model file size
#  10. Per-image CSV for every model
#  11. Per-model overall CSV
#  12. Combined CSV comparing all model formats
#
# IMPORTANT:
#   - Use the SAME test dataset and thresholds for all models.
#   - TensorRT engines should normally be built on the target deployment device.
#   - INT8 engines should be calibrated/exported correctly before evaluation.
#   - IMG_SIZE should match the TensorRT export size for fixed-shape engines.

from ultralytics import YOLO

import os
import cv2
import yaml
import time
import numpy as np
import pandas as pd

from pathlib import Path

# MODELS TO TEST
# Add/remove models here.
#
# "name" is the label that will appear in output folders and CSV files.
# "path" can be either a .pt model or a TensorRT .engine model.
#
# Example:
# MODELS = [
#     {"name": "PyTorch_FP32",   "path": "models/best.pt"},
#     {"name": "TensorRT_FP32",  "path": "models/best_fp32.engine"},
#     {"name": "TensorRT_FP16",  "path": "models/best_fp16.engine"},
#     {"name": "TensorRT_INT8",  "path": "models/best_int8.engine"},
# ]
#
# If you only want to test one TensorRT engine, simply keep one entry.

MODELS = [
    # {
    #     "name": "PyTorch_FP32",
    #     "path": "/RGB_PyTorch_FP32.pt"
    # },

    # Uncomment and update these paths after creating the engines.
    #
    # {
    #     "name": "TensorRT_INT8",
    #     "path": "/RGB_TensorRT_INT8.engine"
    # },
    #
    # {
    #     "name": "TensorRT_FP16",
    #     "path": "/weights/best_fp16.engine"
    # },
    #
    # {
    #     "name": "TensorRT_INT8",
    #     "path": "/weights/best_int8.engine"
    # },
]


# Dataset YAML
DATA_YAML = "/dataset.yaml"

# Main output directory
OUTPUT_DIR = "test_evaluation_RGB"

# Inference settings
IMG_SIZE = 640

# IMPORTANT:
# Use exactly the same confidence threshold for all models being compared.
CONF_THRESHOLD = 0.50

IOU_NMS_THRESHOLD = 0.70

# CUDA device
DEVICE = 0

# Number of warm-up predictions before timing the test set.
# Useful for CUDA / TensorRT initialization.
WARMUP_RUNS = 5

# Run Ultralytics validation to obtain mAP, precision and recall.
RUN_ULTRALYTICS_VAL = True

# Save Ultralytics validation plots.
SAVE_VAL_PLOTS = True

# CLASS DEFINITIONS

# Must correspond to class IDs in dataset.yaml.
# Assumption:
# 0 = Crop
# 1 = Weed

CLASS_NAMES = {
    0: "Crop",
    1: "Weed"
}

# OpenCV uses BGR colors.
COLORS = {
    0: (0, 255, 0),       # Crop = GREEN
    1: (0, 0, 255)        # Weed = RED
}

# HELPER FUNCTIONS

def safe_name(name):
    """Create a filesystem-safe model name."""
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(name)
    )


def synchronize_cuda():
    """
    Synchronize CUDA when available.

    This makes wall-clock timing more meaningful because CUDA operations
    can otherwise execute asynchronously.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    except Exception:
        pass


def get_file_size_mb(file_path):
    """Return model file size in MB."""
    file_path = Path(file_path)

    if not file_path.exists():
        return np.nan

    return file_path.stat().st_size / (1024 ** 2)


def polygon_to_mask(
    polygon_points,
    width,
    height
):
    """
    Convert normalized YOLO segmentation polygon coordinates
    to a binary mask.
    """

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


def overlay_mask(
    image,
    mask,
    color,
    alpha=0.40
):
    """Overlay a binary mask on an image."""

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

# LOAD DATASET YAML

with open(DATA_YAML, "r") as file:
    dataset_yaml = yaml.safe_load(file)


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

test_path_string = str(test_image_dir)

if f"{os.sep}images{os.sep}" in test_path_string:

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
    "*.tiff"
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

# CREATE MAIN OUTPUT DIRECTORY

output_dir = Path(OUTPUT_DIR)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# STORAGE FOR COMBINED MODEL COMPARISON

all_models_results = []

# EVALUATE EACH MODEL

for model_number, model_info in enumerate(
    MODELS,
    start=1
):

    model_name = model_info["name"]
    model_path = Path(model_info["path"])

    model_name_safe = safe_name(
        model_name
    )


    print("\n")
    print("=" * 80)
    print(
        f"MODEL {model_number}/{len(MODELS)}: "
        f"{model_name}"
    )
    print("=" * 80)

    print("\nModel path:")
    print(model_path)


    if not model_path.exists():
        print(
            f"\nWARNING: Model does not exist. "
            f"Skipping:\n{model_path}"
        )
        continue

    # IDENTIFY MODEL FORMAT

    extension = model_path.suffix.lower()

    if extension == ".pt":
        model_format = "PyTorch"

    elif extension == ".engine":
        model_format = "TensorRT"

    else:
        model_format = extension.replace(
            ".",
            ""
        ).upper()


    model_size_mb = get_file_size_mb(
        model_path
    )


    print(
        f"\nModel format : {model_format}"
    )

    print(
        f"Model size   : {model_size_mb:.2f} MB"
    )

    # CREATE MODEL-SPECIFIC OUTPUT DIRECTORIES

    model_output_dir = (
        output_dir /
        model_name_safe
    )

    prediction_dir = (
        model_output_dir /
        "predictions"
    )

    groundtruth_dir = (
        model_output_dir /
        "ground_truth"
    )

    comparison_dir = (
        model_output_dir /
        "comparison"
    )

    validation_dir = (
        model_output_dir /
        "validation"
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

    validation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # LOAD MODEL

    print("\nLoading model...")

    model = YOLO(
        str(model_path),
        task="segment"
    )

    # WARM-UP

    # Warm-up is NOT included in reported timing.
    # TensorRT/CUDA initialization can make the first prediction unusually slow.

    if WARMUP_RUNS > 0:

        print(
            f"\nRunning {WARMUP_RUNS} warm-up inference(s)..."
        )

        warmup_image = image_files[0]

        for warmup_index in range(
            WARMUP_RUNS
        ):

            _ = model.predict(
                source=str(warmup_image),
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_NMS_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

        synchronize_cuda()

    # GLOBAL COUNTERS FOR THIS MODEL

    predicted_counts = {
        0: 0,
        1: 0
    }

    groundtruth_counts = {
        0: 0,
        1: 0
    }


    intersection_total = {
        0: 0,
        1: 0
    }

    union_total = {
        0: 0,
        1: 0
    }


    per_image_results = []


    # Timing storage
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    ultralytics_total_times = []
    pipeline_times = []

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
            ) as file:

                lines = file.readlines()


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


                # Invalid polygons are ignored safely.
                if (
                    len(coordinates) < 6
                    or
                    len(coordinates) % 2 != 0
                ):
                    continue


                coordinates = coordinates.reshape(
                    -1,
                    2
                )


                gt_mask_instance = polygon_to_mask(
                    coordinates,
                    width,
                    height
                )


                gt_masks[class_id] = np.maximum(
                    gt_masks[class_id],
                    gt_mask_instance
                )


                groundtruth_counts[class_id] += 1

                image_gt_counts[class_id] += 1


                groundtruth_visualization = (
                    overlay_mask(
                        groundtruth_visualization,
                        gt_mask_instance,
                        COLORS[class_id]
                    )
                )


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


        # =====================================================================
        # MODEL INFERENCE + TIMING
        # =====================================================================

        synchronize_cuda()

        pipeline_start = time.perf_counter()


        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_NMS_THRESHOLD,
            device=DEVICE,
            verbose=False
        )


        synchronize_cuda()

        pipeline_end = time.perf_counter()


        pipeline_time_ms = (
            pipeline_end -
            pipeline_start
        ) * 1000.0


        result = results[0]

        # ULTRALYTICS TIMING

        speed = getattr(
            result,
            "speed",
            {}
        ) or {}


        preprocess_ms = float(
            speed.get(
                "preprocess",
                np.nan
            )
        )

        inference_ms = float(
            speed.get(
                "inference",
                np.nan
            )
        )

        postprocess_ms = float(
            speed.get(
                "postprocess",
                np.nan
            )
        )


        if (
            np.isfinite(preprocess_ms)
            and
            np.isfinite(inference_ms)
            and
            np.isfinite(postprocess_ms)
        ):

            ultralytics_total_ms = (
                preprocess_ms
                +
                inference_ms
                +
                postprocess_ms
            )

        else:
            ultralytics_total_ms = np.nan


        preprocess_times.append(
            preprocess_ms
        )

        inference_times.append(
            inference_ms
        )

        postprocess_times.append(
            postprocess_ms
        )

        ultralytics_total_times.append(
            ultralytics_total_ms
        )

        pipeline_times.append(
            pipeline_time_ms
        )

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


            # Polygon masks in original image coordinates.
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

            union_total[class_id] += (
                union
            )


            if union > 0:
                iou = intersection / union
            else:
                iou = np.nan


            image_iou[class_id] = iou

        # SAVE VISUALIZATIONS

        cv2.imwrite(
            str(
                prediction_dir /
                image_path.name
            ),
            prediction_visualization
        )


        cv2.imwrite(
            str(
                groundtruth_dir /
                image_path.name
            ),
            groundtruth_visualization
        )


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

        inference_fps = (
            1000.0 / inference_ms
            if np.isfinite(inference_ms) and inference_ms > 0
            else np.nan
        )

        pipeline_fps = (
            1000.0 / pipeline_time_ms
            if pipeline_time_ms > 0
            else np.nan
        )


        per_image_results.append(
            {
                "Model":
                    model_name,

                "Format":
                    model_format,

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
                    image_iou[1],

                "Preprocess_ms":
                    preprocess_ms,

                "Inference_ms":
                    inference_ms,

                "Postprocess_ms":
                    postprocess_ms,

                "Ultralytics_Total_ms":
                    ultralytics_total_ms,

                "Pipeline_ms":
                    pipeline_time_ms,

                "Inference_FPS":
                    inference_fps,

                "Pipeline_FPS":
                    pipeline_fps
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

    # TIMING SUMMARY

    mean_preprocess_ms = float(
        np.nanmean(preprocess_times)
    )

    mean_inference_ms = float(
        np.nanmean(inference_times)
    )

    mean_postprocess_ms = float(
        np.nanmean(postprocess_times)
    )

    mean_ultralytics_total_ms = float(
        np.nanmean(ultralytics_total_times)
    )

    mean_pipeline_ms = float(
        np.nanmean(pipeline_times)
    )


    inference_fps = (
        1000.0 / mean_inference_ms
        if np.isfinite(mean_inference_ms)
        and mean_inference_ms > 0
        else np.nan
    )


    pipeline_fps = (
        1000.0 / mean_pipeline_ms
        if np.isfinite(mean_pipeline_ms)
        and mean_pipeline_ms > 0
        else np.nan
    )

    # STANDARD ULTRALYTICS TEST METRICS

    mask_map50 = np.nan
    mask_map5095 = np.nan
    mask_precision = np.nan
    mask_recall = np.nan


    if RUN_ULTRALYTICS_VAL:

        print(
            "\nRunning Ultralytics test-set evaluation..."
        )


        try:

            metrics = model.val(
                data=DATA_YAML,
                split="test",
                imgsz=IMG_SIZE,
                device=DEVICE,
                conf=CONF_THRESHOLD,
                iou=IOU_NMS_THRESHOLD,
                plots=SAVE_VAL_PLOTS,
                project=str(validation_dir),
                name="test_metrics",
                exist_ok=True,
                verbose=True
            )


            mask_map50 = float(
                metrics.seg.map50
            )

            mask_map5095 = float(
                metrics.seg.map
            )

            mask_precision = float(
                metrics.seg.mp
            )

            mask_recall = float(
                metrics.seg.mr
            )


        except Exception as error:

            print("\nWARNING:")
            print(
                "Ultralytics validation failed for this model."
            )
            print(
                "Custom IoU, counts, visualizations and timing "
                "were still calculated."
            )
            print(
                f"Validation error: {error}"
            )

    # SAVE PER-IMAGE RESULTS

    per_image_df = pd.DataFrame(
        per_image_results
    )


    per_image_csv = (
        model_output_dir /
        "per_image_metrics.csv"
    )


    per_image_df.to_csv(
        per_image_csv,
        index=False
    )

    # CREATE OVERALL METRICS TABLE

    model_result = {

        "Model":
            model_name,

        "Format":
            model_format,

        "Model_Path":
            str(model_path),

        "Model_Size_MB":
            model_size_mb,

        "Number_Test_Images":
            len(per_image_results),

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
            ),

        "Mean_Preprocess_ms":
            mean_preprocess_ms,

        "Mean_Inference_ms":
            mean_inference_ms,

        "Mean_Postprocess_ms":
            mean_postprocess_ms,

        "Mean_Ultralytics_Total_ms":
            mean_ultralytics_total_ms,

        "Mean_Pipeline_ms":
            mean_pipeline_ms,

        "Inference_FPS":
            inference_fps,

        "Pipeline_FPS":
            pipeline_fps
    }


    overall_results = pd.DataFrame(
        [model_result]
    )


    overall_csv = (
        model_output_dir /
        "overall_metrics.csv"
    )


    overall_results.to_csv(
        overall_csv,
        index=False
    )


    all_models_results.append(
        model_result
    )

    # PRINT MODEL RESULTS

    print("\n")
    print("=" * 70)
    print(
        f"FINAL TEST RESULTS: {model_name}"
    )
    print("=" * 70)


    print("\nMODEL")

    print(
        f"Format                 : "
        f"{model_format}"
    )

    print(
        f"Model size             : "
        f"{model_size_mb:.2f} MB"
    )


    print("\nSEGMENTATION METRICS")

    print(
        f"Mask mAP50             : "
        f"{mask_map50:.4f}"
    )

    print(
        f"Mask mAP50-95          : "
        f"{mask_map5095:.4f}"
    )

    print(
        f"Mask Precision         : "
        f"{mask_precision:.4f}"
    )

    print(
        f"Mask Recall            : "
        f"{mask_recall:.4f}"
    )


    print("\nIoU")

    print(
        f"Crop IoU          : "
        f"{dataset_iou[0]:.4f}"
    )

    print(
        f"Weed IoU               : "
        f"{dataset_iou[1]:.4f}"
    )

    print(
        f"Overall IoU            : "
        f"{overall_iou:.4f}"
    )


    print("\nINSTANCE COUNTS")

    print(
        f"GT Crop           : "
        f"{groundtruth_counts[0]}"
    )

    print(
        f"Pred Crop        : "
        f"{predicted_counts[0]}"
    )

    print(
        f"GT Weed                : "
        f"{groundtruth_counts[1]}"
    )

    print(
        f"Pred Weed              : "
        f"{predicted_counts[1]}"
    )


    print("\nTIMING")

    print(
        f"Mean preprocess        : "
        f"{mean_preprocess_ms:.3f} ms/image"
    )

    print(
        f"Mean inference         : "
        f"{mean_inference_ms:.3f} ms/image"
    )

    print(
        f"Mean postprocess       : "
        f"{mean_postprocess_ms:.3f} ms/image"
    )

    print(
        f"Ultralytics total      : "
        f"{mean_ultralytics_total_ms:.3f} ms/image"
    )

    print(
        f"Measured pipeline      : "
        f"{mean_pipeline_ms:.3f} ms/image"
    )

    print(
        f"Inference FPS          : "
        f"{inference_fps:.2f}"
    )

    print(
        f"Pipeline FPS           : "
        f"{pipeline_fps:.2f}"
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


    # Free model reference before loading the next model.
    del model

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass


# SAVE COMBINED MODEL COMPARISON

if len(all_models_results) > 0:

    combined_df = pd.DataFrame(
        all_models_results
    )


    combined_csv = (
        output_dir /
        "combined_model_comparison.csv"
    )


    combined_df.to_csv(
        combined_csv,
        index=False
    )


    print("\n")
    print("=" * 80)
    print("COMBINED MODEL COMPARISON")
    print("=" * 80)


    display_columns = [
        "Model",
        "Format",
        "Model_Size_MB",
        "Mask_mAP50",
        "Mask_mAP50_95",
        "Mask_Precision",
        "Mask_Recall",
        "IoU_Crop",
        "IoU_Weed",
        "Overall_IoU",
        "Mean_Inference_ms",
        "Inference_FPS",
        "Mean_Pipeline_ms",
        "Pipeline_FPS"
    ]


    print(
        combined_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\nCombined CSV:",
        combined_csv
    )


else:

    print(
        "\nERROR: No models were successfully evaluated."
    )


print("\n")
print("=" * 80)
print("MULTI-FORMAT TEST EVALUATION COMPLETED")
print("=" * 80)
