# uav_MS_yolo RGBRN - MULTI-FORMAT INDEPENDENT TEST EVALUATION 
# The 
# Supports 5-channel RGBRN models:
#   - PyTorch (.pt)
#   - TensorRT (.engine), including FP32 / FP16 / INT8 engines
#
# For every model, outputs:
#   1. Prediction visualization
#       Crop      = GREEN
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

import os
import sys
import torch

print("\nPython executable:")
print(sys.executable)

from pathlib import Path

# uav_MS_yolo SETUP

clone_dir = r"uav_MS_yolo"

yolo_source_path = os.path.abspath(
    os.path.join(clone_dir, "code", "ultralytics_uav_MS_yolo")
)

code_path = os.path.abspath(
    os.path.join(clone_dir, "code")
)

sys.path.insert(0, yolo_source_path)
sys.path.insert(0, code_path)

from ultralytics import YOLO

import cv2
import yaml
import time
import numpy as np
import pandas as pd
import tifffile

# RGBRN DATASET SETTINGS

dataset_path = r"/processed_RGBRN"

images_train_path_RGBRN = os.path.join(dataset_path, "train")
images_val_path_RGBRN = os.path.join(dataset_path, "val")
images_test_path_RGBRN = os.path.join(dataset_path, "test")

number_of_channels = 5
number_of_classes = 2

class_names = [
    "Crop",
    "Weed"
]

# Create/refresh the same RGBRN YAML used by the custom multispectral pipeline.
output_yaml_RGBRN = os.path.join(
    dataset_path,
    "data_RGBRN.yaml"
)

DATA_YAML = output_yaml_RGBRN

print("\nRGBRN dataset YAML:")
print(DATA_YAML)

# MODELS TO TEST
#
# IMPORTANT:
# Every model listed here must be a 5-channel RGBRN segmentation model.
#
# Add/remove models as required. TensorRT engines must have been exported
# from the corresponding 5-channel RGBRN model.
#
# For fixed-shape TensorRT engines, IMG_SIZE must match the export size.

MODELS = [
    # {
    #     "name": "RGBRN_TensorRT_FP16",
    #     "path": os.path.join(
    #         dataset_path,
    #         "runs",
    #         "yolo",
    #         "weights",
    #         "RGBRN_TensorRT_FP16.engine"
    #     )
    # },

    # Uncomment and update these paths after creating the RGBRN engines.
    #
    # {
    #     "name": "RGBRN_TensorRT_FP32",
    #     "path": os.path.join(
    #         dataset_path,
    #         "runs",
    #         "Crop_Weed_RGBRN",
    #         "weights",
    #         "best_fp32.engine"
    #     )
    # },
    #
    # {
    #     "name": "RGBRN_TensorRT_FP16",
    #     "path": os.path.join(
    #         dataset_path,
    #         "runs",
    #         "Crop_Weed_RGBRN",
    #         "weights",
    #         "best_fp16.engine"
    #     )
    # },
    #
    # {
    #     "name": "RGBRN_TensorRT_INT8",
    #     "path": os.path.join(
    #         dataset_path,
    #         "runs",
    #         "Crop_Weed_RGBRN",
    #         "weights",
    #         "best_int8.engine"
    #     )
    # },
]

# TEST SETTINGS

OUTPUT_DIR = os.path.join(
    dataset_path,
    "test_evaluation_RGBRN_multi_format"
)

IMG_SIZE = 640

# Keep these identical across RGB and RGBRN comparisons.
CONF_THRESHOLD = 0.50
IOU_NMS_THRESHOLD = 0.70

# Warm-up is excluded from timing.
WARMUP_RUNS = 5

RUN_ULTRALYTICS_VAL = True
SAVE_VAL_PLOTS = True

device = 0 if torch.cuda.is_available() else "cpu"

print(f"\nUsing device: {device}")

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


# DEFINE RGBRN TEST IMAGE AND LABEL DIRECTORIES

# Expected RGBRN structure:
# processed_RGBRN/
# └── test/
#     ├── images/
#     └── labels/

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

if not test_image_dir.exists():
    raise FileNotFoundError(
        f"\nTest image directory does not exist:\n{test_image_dir}"
    )

if not test_label_dir.exists():
    raise FileNotFoundError(
        f"\nTest label directory does not exist:\n{test_label_dir}"
    )

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
                device=device,
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
    # FUNCTION:
    # LOAD RGB VISUALIZATION FROM 5-CHANNEL RGBRN IMAGE

    def load_rgb_visualization(image_path):

        """
        Load the 5-channel RGBRN TIFF and use only
        R, G, B for visualization.

        Expected channel order:
            R, G, B, RedEdge, NIR

        IMPORTANT:
        This function is ONLY for visualization.

        Model inference still receives the original
        5-channel TIFF through:
            source=str(image_path)
        """

        image_path = Path(image_path)

        suffix = image_path.suffix.lower()

        # MULTISPECTRAL TIFF

        if suffix in [".tif", ".tiff"]:

            image_ms = tifffile.imread(
                str(image_path)
            )

            # Format 1:
            # H x W x 5

            if (
                image_ms.ndim == 3
                and
                image_ms.shape[-1] >= 5
            ):

                rgb = image_ms[
                    :, :, :3
                ]

            # Format 2:
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
                    f"Unexpected RGBRN image shape "
                    f"for {image_path.name}: "
                    f"{image_ms.shape}"
                )

            # CONVERT TO UINT8 FOR VISUALIZATION ONLY

            if rgb.dtype != np.uint8:

                rgb_float = rgb.astype(
                    np.float32
                )

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


            # RGB -> BGR
            # OpenCV requires BGR for saving/drawing

            rgb_bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )


            return rgb_bgr


        # STANDARD IMAGE

        else:

            image = cv2.imread(
                str(image_path)
            )


            if image is None:

                raise ValueError(
                    f"Could not read image: "
                    f"{image_path}"
                )


            return image

    for image_index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{image_index}/{len(image_files)}] "
            f"{image_path.name}"
        )


        # READ ORIGINAL IMAGE

        # Load only the RGB channels for drawing/visualization.
        # Inference below still uses the ORIGINAL 5-channel image_path.
        try:
            image = load_rgb_visualization(
                image_path
            )
        except Exception as error:
            print(
                "WARNING: Could not load visualization image:",
                image_path,
                error
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

        # MODEL INFERENCE + TIMING

        synchronize_cuda()

        pipeline_start = time.perf_counter()


        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_NMS_THRESHOLD,
            device=device,
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
                f"{image_path.stem}.jpg"
            ),
            prediction_visualization
        )


        cv2.imwrite(
            str(
                groundtruth_dir /
                f"{image_path.stem}.jpg"
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
                f"{image_path.stem}.jpg"
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
                device=device,
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

        "Input_Channels":
            number_of_channels,

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
        f"GT Crop          : "
        f"{groundtruth_counts[0]}"
    )

    print(
        f"Pred Crop         : "
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
print("RGBRN MULTI-FORMAT TEST EVALUATION COMPLETED")
print("=" * 80)
