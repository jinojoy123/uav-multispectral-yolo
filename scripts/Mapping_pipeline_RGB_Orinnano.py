# RGB ORTHOMOSAIC -> WEED PRESCRIPTION MAP
# EDGE DEVICE DEPLOYMENT BENCHMARK
#
# Supported model configurations:
#
#   1. PyTorch FP32
#   2. TensorRT FP32
#   3. TensorRT FP16
#   4. TensorRT INT8
#
#
# PERFORMANCE MEASUREMENTS:
#
#   - Orthomosaic tiling time
#   - YOLO preprocessing time
#   - Model/engine inference time
#       * total inference time
#       * mean / median inference latency
#       * minimum / maximum latency
#       * P95 inference latency
#       * engine FPS
#   - Segmentation postprocessing time
#   - Complete model-call latency
#       * mean / median latency
#       * minimum / maximum latency
#       * P95 latency
#       * application FPS
#   - Total prescription-map processing time
#
# RESOURCE UTILIZATION:
#
#   - System RAM
#   - Process RAM
#   - GPU utilization
#   - Platform power consumption
#   - Energy consumed per orthomosaic
#
#
# ENERGY:
#
#       E = P x t
#
#   P = average platform power (W)
#   t = total operational processing time (s)
#
#   E = energy consumed (J)
#
#
# EXPERIMENTAL DESIGN:
#
#   - Model loading excluded from main benchmark
#   - 10 warm-up inferences
#   - Warm-up excluded from all measurements
#   - 3 complete benchmark runs
#   - Batch size = 1
#   - Same orthomosaic for all models
#   - Same confidence threshold for all models


import os
import shutil
import rasterio

from rasterio.windows import Window
from rasterio.merge import merge

from ultralytics import YOLO

from pathlib import Path

import numpy as np
import cv2

import geopandas as gpd

from shapely.geometry import box

import time
import threading
import subprocess
import csv
import re
import psutil
import statistics
import torch


# MODEL CONFIGURATION

# -------------------------------------------------------------------------
# Change these TWO variables for each model experiment.
#
# Examples:
#
# MODEL_VERSION = "PyTorch_FP32"
# MODEL_PATH = "models/RGB_PyTorch_FP32.pt"
#
# MODEL_VERSION = "TensorRT_FP32"
# MODEL_PATH = "models/RGB_TensorRT_FP32.engine"
#
# MODEL_VERSION = "TensorRT_FP16"
# MODEL_PATH = "models/RGB_TensorRT_FP16.engine"
#
# MODEL_VERSION = "TensorRT_INT8"
# MODEL_PATH = "models/RGB_TensorRT_INT8.engine"
# -------------------------------------------------------------------------

MODEL_VERSION = "PyTorch_FP32"


MODEL_PATH = (
    "/RGB_PyTorch_FP32.pt"
)


# ORIGINAL SETTINGS

INPUT_TIF = (
    "/Orthomosaic_RGB.tif"
)


TILE_SIZE = 640


CONF_THRESHOLD = 0.5


GRID_SIZE_M = 1.0


# BENCHMARK SETTINGS

# Number of model warm-up inference runs
WARMUP_RUNS = 10


# Number of complete experimental repetitions
NUM_BENCHMARK_RUNS = 3


# tegrastats sampling interval
# milliseconds
TEGRASTATS_INTERVAL_MS = 100

# MAIN BENCHMARK OUTPUT DIRECTORY

BENCHMARK_ROOT = os.path.join(
    "benchmark_results",
    MODEL_VERSION
)


os.makedirs(
    BENCHMARK_ROOT,
    exist_ok=True
)

# RESOURCE MONITORING

resource_samples = []

monitor_resources = False

# TEGRASTATS PARSER

def parse_tegrastats_line(line):

    sample = {

        "timestamp": time.time(),

        "ram_used_mb": np.nan,

        "ram_total_mb": np.nan,

        "process_ram_mb": np.nan,

        "gpu_util_percent": np.nan,

        "power_mw": np.nan

    }

    # SYSTEM RAM

    ram_match = re.search(
        r"RAM\s+(\d+)/(\d+)MB",
        line
    )


    if ram_match:

        sample["ram_used_mb"] = float(
            ram_match.group(1)
        )

        sample["ram_total_mb"] = float(
            ram_match.group(2)
        )

    # GPU UTILIZATION

    gpu_match = re.search(
        r"GR3D_FREQ\s+(\d+)%",
        line
    )


    if gpu_match:

        sample["gpu_util_percent"] = float(
            gpu_match.group(1)
        )


    # PROCESS RAM

    try:

        process = psutil.Process(
            os.getpid()
        )

        process_memory = (
            process.memory_info().rss
            /
            (1024 ** 2)
        )

        sample["process_ram_mb"] = (
            process_memory
        )

    except Exception:

        pass


    # PLATFORM POWER
    # Depending on Jetson / JetPack:
    # VDD_IN 12345mW/.....
    # or
    # POM_5V_IN 12345mW/.....


    power_match = re.search(
        r"VDD_IN\s+(\d+)mW",
        line
    )


    if power_match is None:

        power_match = re.search(
            r"POM_5V_IN\s+(\d+)mW",
            line
        )


    if power_match:

        sample["power_mw"] = float(
            power_match.group(1)
        )


    return sample


# RESOURCE MONITOR THREAD

def resource_monitor():

    global monitor_resources
    global resource_samples


    command = [

        "tegrastats",

        "--interval",

        str(
            TEGRASTATS_INTERVAL_MS
        )

    ]


    try:

        process = subprocess.Popen(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True,

            bufsize=1

        )

    except FileNotFoundError:

        print(
            "\nWARNING:"
            "\n'tegrastats' was not found."
            "\nJetson resource monitoring cannot be performed."
        )

        return


    try:

        while monitor_resources:

            line = process.stdout.readline()


            if not line:

                continue


            sample = parse_tegrastats_line(
                line
            )


            resource_samples.append(
                sample
            )


    finally:

        process.terminate()


        try:

            process.wait(
                timeout=2
            )

        except subprocess.TimeoutExpired:

            process.kill()



# GPU SYNCHRONIZATION


def synchronize_gpu():

    if torch.cuda.is_available():

        torch.cuda.synchronize()



# SAFE STATISTICS


def valid_values(samples, key):

    return [

        x[key]

        for x in samples

        if key in x

        and not np.isnan(
            x[key]
        )

    ]


def clean_numeric_values(values):

    return [
        x
        for x in values
        if not np.isnan(x)
    ]


def safe_mean(values):

    values = clean_numeric_values(values)

    if len(values) == 0:
        return np.nan

    return statistics.mean(values)


def safe_max(values):

    values = clean_numeric_values(values)

    if len(values) == 0:
        return np.nan

    return max(values)


def safe_min(values):

    values = clean_numeric_values(values)

    if len(values) == 0:
        return np.nan

    return min(values)


def safe_median(values):

    values = clean_numeric_values(values)

    if len(values) == 0:
        return np.nan

    return statistics.median(values)



# SAVE RESOURCE DATA


def save_resource_csv(
    samples,
    output_path
):

    fieldnames = [

        "timestamp",

        "ram_used_mb",

        "ram_total_mb",

        "process_ram_mb",

        "gpu_util_percent",

        "power_mw"

    ]


    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=fieldnames

        )


        writer.writeheader()


        writer.writerows(
            samples
        )



# SAVE PER-TILE PERFORMANCE DATA


def save_tile_performance_csv(
    tile_names,
    preprocess_times,
    engine_inference_times,
    postprocess_times,
    model_call_times,
    detection_counts,
    output_path
):

    with open(
        output_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "tile_number",
                "tile_name",
                "preprocess_ms",
                "engine_inference_ms",
                "postprocess_ms",
                "model_call_ms",
                "detection_count"
            ]
        )


        for i in range(
            len(tile_names)
        ):

            writer.writerow(
                [
                    i,
                    tile_names[i],
                    preprocess_times[i],
                    engine_inference_times[i],
                    postprocess_times[i],
                    model_call_times[i],
                    detection_counts[i]
                ]
            )



# LOAD MODEL
#
# Model loading is intentionally OUTSIDE the benchmark.


print("\n")
print("=" * 80)
print("LOADING MODEL")
print("=" * 80)


print(
    "Model version:",
    MODEL_VERSION
)


print(
    "Model path:",
    MODEL_PATH
)


model_load_start = time.perf_counter()


model = YOLO(
    MODEL_PATH,
    task="segment"
)


synchronize_gpu()


model_load_end = time.perf_counter()


MODEL_LOAD_TIME_MS = (

    model_load_end
    -
    model_load_start

) * 1000.0


print(
    f"\nModel loading time: "
    f"{MODEL_LOAD_TIME_MS:.2f} ms"
)


print(
    "\nModel loading time is NOT included "
    "in prescription-map processing time."
)



# STORAGE FOR ALL BENCHMARK RUNS


all_run_results = []



# COMPLETE BENCHMARK LOOP


for BENCHMARK_RUN in range(
    1,
    NUM_BENCHMARK_RUNS + 1
):


    print("\n")
    print("=" * 80)
    print(
        f"BENCHMARK RUN "
        f"{BENCHMARK_RUN} / "
        f"{NUM_BENCHMARK_RUNS}"
    )
    print("=" * 80)



    # RUN-SPECIFIC OUTPUT DIRECTORIES


    RUN_DIR = os.path.join(

        BENCHMARK_ROOT,

        f"run_{BENCHMARK_RUN:02d}"

    )


    TILE_OUTPUT_DIR = os.path.join(

        RUN_DIR,

        "tiles_for_inference"

    )


    ANNOTATED_TILE_DIR = os.path.join(

        RUN_DIR,

        "annotated_tiles"

    )


    os.makedirs(
        TILE_OUTPUT_DIR,
        exist_ok=True
    )


    os.makedirs(
        ANNOTATED_TILE_DIR,
        exist_ok=True
    )


    FINAL_OUTPUT = os.path.join(

        RUN_DIR,

        "stitched_orthomosaic.tif"

    )


    GRID_RASTER = os.path.join(

        RUN_DIR,

        "grid.tif"

    )


    COLOR_GRID_RASTER = os.path.join(

        RUN_DIR,

        "crop_weed_grid_map_25.tif"

    )


    RASTER_PRESCRIPTION = os.path.join(

        RUN_DIR,

        "weed_prescription_rate.tif"

    )


    DENSITY_RASTER = os.path.join(

        RUN_DIR,

        "weed_density_heatmap.tif"

    )


    COLORED_HEATMAP = os.path.join(

        RUN_DIR,

        "weed_density_heatmap_color.tif"

    )


    SHAPEFILE_PRESCRIPTION = os.path.join(

        RUN_DIR,

        "weed_prescription.shp"

    )


    SHAPEFILE_WGS84 = os.path.join(

        RUN_DIR,

        "weed_prescription_WGS84.shp"

    )



    # RESET BENCHMARK VARIABLES


    resource_samples = []

    # Ultralytics internal timing for every tile
    per_tile_preprocess_ms = []
    per_tile_engine_inference_ms = []
    per_tile_postprocess_ms = []

    # Complete model(...) wall-clock time
    per_tile_model_call_ms = []

    # Number of final detections returned for every tile
    per_tile_detection_count = []

    # Accumulated times
    total_engine_inference_ms = 0.0
    total_model_call_seconds = 0.0



    # STEP 1
    #
    # TILE ORTHOMOSAIC
    #
    # This stage IS measured.


    print("\n")
    print("-" * 80)
    print("STEP 1: TILING ORTHOMOSAIC")
    print("-" * 80)


    synchronize_gpu()


    tiling_start = time.perf_counter()


    tile_geotransforms = []


    with rasterio.open(
        INPUT_TIF
    ) as src:


        width = src.width

        height = src.height

        crs = src.crs

        full_transform = src.transform


        tile_id = 0


        for top in range(
            0,
            height,
            TILE_SIZE
        ):


            for left in range(
                0,
                width,
                TILE_SIZE
            ):


                win_width = min(
                    TILE_SIZE,
                    width - left
                )


                win_height = min(
                    TILE_SIZE,
                    height - top
                )


                window = Window(

                    left,

                    top,

                    win_width,

                    win_height

                )


                tile_transform = (
                    src.window_transform(
                        window
                    )
                )


                tile_geotransforms.append(
                    tile_transform
                )


                tile = src.read(
                    window=window
                )


                tile = np.transpose(
                    tile,
                    (1, 2, 0)
                )


                if tile.dtype != np.uint8:

                    tile = tile.astype(
                        np.float32
                    )


                    tile = (

                        255

                        *

                        (
                            tile
                            -
                            tile.min()
                        )

                        /

                        (
                            tile.max()
                            -
                            tile.min()
                            +
                            1e-6
                        )

                    )


                    tile = tile.astype(
                        np.uint8
                    )


                if tile.shape[2] == 4:

                    tile = cv2.cvtColor(

                        tile,

                        cv2.COLOR_RGBA2RGB

                    )


                elif tile.shape[2] == 1:

                    tile = cv2.cvtColor(

                        tile,

                        cv2.COLOR_GRAY2RGB

                    )


                else:

                    tile = tile[:, :, :3]


                out_path = os.path.join(

                    TILE_OUTPUT_DIR,

                    f"tile_{tile_id:04d}.png"

                )


                cv2.imwrite(

                    out_path,

                    cv2.cvtColor(
                        tile,
                        cv2.COLOR_RGB2BGR
                    )

                )


                tile_id += 1


    synchronize_gpu()


    tiling_end = time.perf_counter()


    tiling_seconds = (

        tiling_end
        -
        tiling_start

    )


    tiling_ms = (

        tiling_seconds
        *
        1000.0

    )


    print(
        f"\nTiled {tile_id} images"
    )


    print(
        f"Tiling time: "
        f"{tiling_ms:.2f} ms"
    )



    # GET TILE FILES


    tile_files = sorted(

        [

            f

            for f in os.listdir(
                TILE_OUTPUT_DIR
            )

            if f.endswith(".png")

        ]

    )


    if len(tile_files) == 0:

        raise RuntimeError(
            "No tiles were generated."
        )



    # MODEL WARM-UP
    #
    # IMPORTANT:
    #
    # Warm-up is performed AFTER tiling.
    #
    # It is NOT included in:
    #
    #   - inference latency
    #   - FPS
    #   - resource utilization
    #   - power
    #   - energy
    #   - total prescription processing time


    print("\n")
    print("-" * 80)
    print("MODEL WARM-UP")
    print("-" * 80)


    warmup_tile = os.path.join(

        TILE_OUTPUT_DIR,

        tile_files[0]

    )


    synchronize_gpu()


    warmup_start = time.perf_counter()


    for warmup_i in range(
        WARMUP_RUNS
    ):


        _ = model(

            warmup_tile,

            conf=CONF_THRESHOLD,

            verbose=False

        )


        synchronize_gpu()


    warmup_end = time.perf_counter()


    warmup_time_ms = (

        warmup_end
        -
        warmup_start

    ) * 1000.0


    print(
        f"Completed "
        f"{WARMUP_RUNS} warm-up runs."
    )


    print(
        f"Warm-up time: "
        f"{warmup_time_ms:.2f} ms"
    )


    print(
        "Warm-up excluded from benchmark."
    )



    # STEP 2A
    #
    # CREATE GRID


    px_w = abs(
        full_transform.a
    )


    px_h = abs(
        full_transform.e
    )


    grid_px_x = int(
        GRID_SIZE_M
        /
        px_w
    )


    grid_px_y = int(
        GRID_SIZE_M
        /
        px_h
    )


    grid_cols = (
        width
        //
        grid_px_x
    )


    grid_rows = (
        height
        //
        grid_px_y
    )


    grid_transform = (
        rasterio.transform.from_origin(

            full_transform.c,

            full_transform.f,

            GRID_SIZE_M,

            GRID_SIZE_M

        )
    )


    crop_count = np.zeros(

        (
            grid_rows,
            grid_cols
        ),

        dtype=np.uint16

    )


    weed_count = np.zeros(

        (
            grid_rows,
            grid_cols
        ),

        dtype=np.uint16

    )


    def world_to_grid(
        x,
        y
    ):


        col = int(

            (
                x
                -
                grid_transform.c
            )

            /

            grid_transform.a

        )


        row = int(

            (
                grid_transform.f
                -
                y
            )

            /

            abs(
                grid_transform.e
            )

        )


        return row, col



    # START RESOURCE MONITORING
    #
    # Resource monitoring starts AFTER warm-up.
    #
    # Therefore warm-up does not contaminate:
    #
    #   GPU utilization
    #   RAM
    #   Power
    #   Energy


    resource_samples = []


    monitor_resources = True


    monitor_thread = threading.Thread(

        target=resource_monitor,

        daemon=True

    )


    monitor_thread.start()


    # Allow tegrastats thread to initialize.
    # This delay is NOT included in processing time.

    time.sleep(
        0.2
    )



    # START TIMING INFERENCE + POSTPROCESSING SECTION


    synchronize_gpu()


    processing_start = time.perf_counter()



    # STEP 2B
    #
    # RUN INFERENCE


    print("\n")
    print("-" * 80)
    print("STEP 2: YOLO INFERENCE")
    print("-" * 80)


    annotated_paths = []


    for idx, tile_file in enumerate(
        tile_files
    ):


        tile_path = os.path.join(

            TILE_OUTPUT_DIR,

            tile_file

        )


        annotated_path = os.path.join(

            ANNOTATED_TILE_DIR,

            tile_file

        )



        # BEGIN COMPLETE MODEL-CALL TIMER
        #
        # This wall-clock measurement includes:
        #
        #   preprocessing
        #   model/engine inference
        #   segmentation postprocessing
        #   Python/runtime overhead
        #
        # It must NOT be reported as pure inference latency.


        synchronize_gpu()

        model_call_start = time.perf_counter()


        results = model(

            tile_path,

            conf=CONF_THRESHOLD,

            verbose=False

        )[0]


        synchronize_gpu()

        model_call_end = time.perf_counter()



        # COMPLETE model(...) WALL-CLOCK TIME


        model_call_seconds = (

            model_call_end
            -
            model_call_start

        )


        model_call_ms = (

            model_call_seconds
            *
            1000.0

        )


        per_tile_model_call_ms.append(
            model_call_ms
        )


        total_model_call_seconds += (
            model_call_seconds
        )



        # ULTRALYTICS INTERNAL TIMING
        #
        # results.speed normally contains:
        #
        # preprocess
        # inference
        # postprocess
        #
        # All values are milliseconds per image.


        preprocess_ms = results.speed.get(
            "preprocess",
            np.nan
        )


        engine_inference_ms = results.speed.get(
            "inference",
            np.nan
        )


        postprocess_ms = results.speed.get(
            "postprocess",
            np.nan
        )


        per_tile_preprocess_ms.append(
            preprocess_ms
        )


        per_tile_engine_inference_ms.append(
            engine_inference_ms
        )


        per_tile_postprocess_ms.append(
            postprocess_ms
        )


        if not np.isnan(
            engine_inference_ms
        ):

            total_engine_inference_ms += (
                engine_inference_ms
            )



        # NUMBER OF FINAL DETECTIONS
        #
        # Useful for investigating segmentation postprocessing outliers.


        if results.boxes is not None:

            detection_count = len(
                results.boxes
            )

        else:

            detection_count = 0


        per_tile_detection_count.append(
            detection_count
        )



        # DEBUG DISPLAY
        #
        # Only print first 10 tiles.
        # All tiles are still stored in the CSV.


        if idx < 10:

            print(
                f"{MODEL_VERSION} "
                f"{tile_file} | "
                f"Preprocess: "
                f"{preprocess_ms:.3f} ms | "
                f"Inference: "
                f"{engine_inference_ms:.3f} ms | "
                f"Postprocess: "
                f"{postprocess_ms:.3f} ms | "
                f"Model call: "
                f"{model_call_ms:.3f} ms | "
                f"Detections: "
                f"{detection_count}"
            )



        # ORIGINAL ANNOTATION CODE


        annotated = results.plot()


        cv2.imwrite(

            annotated_path,

            cv2.cvtColor(

                annotated,

                cv2.COLOR_RGB2BGR

            )

        )


        annotated_paths.append(
            annotated_path
        )


        if results.boxes is None:

            continue


        boxes = (
            results.boxes.xyxy
            .cpu()
            .numpy()
        )


        classes = (
            results.boxes.cls
            .cpu()
            .numpy()
        )


        # DEBUG:
        # print first detections
        # from first 2 tiles

        if idx < 2:


            print(
                f"\nTile {tile_file}"
            )


            for box_value, cls in zip(

                boxes[:5],

                classes[:5]

            ):


                print(

                    "Class:",

                    int(cls),

                    "Box:",

                    box_value

                )


        tile_transform = (
            tile_geotransforms[idx]
        )


        for box_value, cls in zip(

            boxes,

            classes

        ):


            xc = (

                box_value[0]
                +
                box_value[2]

            ) / 2


            yc = (

                box_value[1]
                +
                box_value[3]

            ) / 2


            xw, yw = rasterio.transform.xy(

                tile_transform,

                yc,

                xc

            )


            r, c = world_to_grid(

                xw,

                yw

            )


            if (

                0 <= r < grid_rows

                and

                0 <= c < grid_cols

            ):


                if int(cls) == 0:

                    crop_count[
                        r,
                        c
                    ] += 1


                elif int(cls) == 1:

                    weed_count[
                        r,
                        c
                    ] += 1


    print(
        "\nYOLO inference + grid counting done"
    )


    print(
        "\n=== TOTAL COUNTS ==="
    )


    print(
        "Total crop detections:",
        crop_count.sum()
    )


    print(
        "Total weed detections:",
        weed_count.sum()
    )



    # STEP 2C
    #
    # SAVE GRID RASTER


    grid_meta = {

        "driver": "GTiff",

        "height": grid_rows,

        "width": grid_cols,

        "count": 1,

        "dtype": "uint8",

        "crs": crs,

        "transform": grid_transform

    }


    with rasterio.open(

        GRID_RASTER,

        "w",

        **grid_meta

    ) as dst:


        dst.write(

            np.zeros(

                (
                    grid_rows,
                    grid_cols
                ),

                dtype=np.uint8

            ),

            1

        )


    print(
        "\n=== NON-EMPTY GRID CELLS ==="
    )


    non_empty = 0


    for r in range(
        grid_rows
    ):


        for c in range(
            grid_cols
        ):


            if (

                crop_count[r, c] > 0

                or

                weed_count[r, c] > 0

            ):


                print(

                    f"Grid "
                    f"(row={r}, col={c}) -> "
                    f"Crops: "
                    f"{crop_count[r,c]}, "
                    f"Weeds: "
                    f"{weed_count[r,c]}"

                )


                non_empty += 1


    print(
        f"\nTotal non-empty grid cells: "
        f"{non_empty}"
    )



    # STEP 2D
    #
    # COLORED GRID


    rgb_grid = np.zeros(

        (
            3,
            grid_rows,
            grid_cols
        ),

        dtype=np.uint8

    )


    for r in range(
        grid_rows
    ):


        for c in range(
            grid_cols
        ):


            if (

                crop_count[r, c] == 0

                and

                weed_count[r, c] == 0

            ):

                continue


            elif (

                weed_count[r, c]

                >

                crop_count[r, c]

            ):


                rgb_grid[
                    :,
                    r,
                    c
                ] = [

                    255,
                    0,
                    0

                ]


            else:


                rgb_grid[
                    :,
                    r,
                    c
                ] = [

                    0,
                    255,
                    0

                ]


    grid_meta.update(
        count=3
    )


    with rasterio.open(

        COLOR_GRID_RASTER,

        "w",

        **grid_meta

    ) as dst:


        dst.write(
            rgb_grid
        )


    print(
        "Saved grid rasters"
    )



    # STEP 3
    #
    # STITCH ANNOTATED TILES


    print("\n")
    print("-" * 80)
    print("STEP 3: STITCHING")
    print("-" * 80)


    src_files = []


    for idx, path in enumerate(
        annotated_paths
    ):


        img = cv2.imread(
            path
        )


        img = cv2.cvtColor(

            img,

            cv2.COLOR_BGR2RGB

        )


        img = np.transpose(

            img,

            (2, 0, 1)

        )


        with rasterio.open(

            path,

            "w",

            driver="GTiff",

            height=img.shape[1],

            width=img.shape[2],

            count=3,

            dtype=img.dtype,

            crs=crs,

            transform=tile_geotransforms[idx]

        ) as dst:


            dst.write(
                img
            )


        src_files.append(

            rasterio.open(
                path
            )

        )


    mosaic, out_transform = merge(
        src_files
    )


    meta = src_files[0].meta.copy()


    meta.update(

        height=mosaic.shape[1],

        width=mosaic.shape[2],

        transform=out_transform

    )


    with rasterio.open(

        FINAL_OUTPUT,

        "w",

        **meta

    ) as dst:


        dst.write(
            mosaic
        )


    for s in src_files:

        s.close()


    print(
        "Final stitched orthomosaic saved"
    )



    # ADDITIONAL PRESCRIPTION MAP GENERATION


    print(
        "\nGenerating variable-rate "
        "prescription map..."
    )


    MAX_RATE = 3.0


    rate_grid = np.zeros(

        (
            grid_rows,
            grid_cols
        ),

        dtype=np.float32

    )


    weed_density_grid = np.zeros(

        (
            grid_rows,
            grid_cols
        ),

        dtype=np.float32

    )


    for r in range(
        grid_rows
    ):


        for c in range(
            grid_cols
        ):


            total = (

                crop_count[r, c]

                +

                weed_count[r, c]

            )


            if total == 0:


                weed_density = 0.0

                rate = 0.0


            else:


                #weed_density = weed_count[r, c] / total

                #grid_area = GRID_SIZE_M * GRID_SIZE_M

                #weed_density = weed_count[r,c] / grid_area


                MAX_WEEDS_PER_CELL = 5


                weed_density = min(

                    weed_count[r, c]

                    /

                    MAX_WEEDS_PER_CELL,

                    1.0

                )


                rate = (

                    MAX_RATE

                    *

                    weed_density

                )


            weed_density_grid[
                r,
                c
            ] = weed_density


            rate_grid[
                r,
                c
            ] = rate


    print(
        "Prescription rate grid created"
    )


    print(
        "Minimum rate:",
        rate_grid.min()
    )


    print(
        "Maximum rate:",
        rate_grid.max()
    )



    # SAVE PRESCRIPTION GEOTIFF


    rate_meta = {

        "driver": "GTiff",

        "height": grid_rows,

        "width": grid_cols,

        "count": 1,

        "dtype": "float32",

        "crs": crs,

        "transform": grid_transform

    }


    with rasterio.open(

        RASTER_PRESCRIPTION,

        "w",

        **rate_meta

    ) as dst:


        dst.write(

            rate_grid,

            1

        )


    print(
        "Saved raster prescription:",
        RASTER_PRESCRIPTION
    )



    # SAVE WEED DENSITY HEATMAP


    with rasterio.open(

        DENSITY_RASTER,

        "w",

        **rate_meta

    ) as dst:


        dst.write(

            weed_density_grid,

            1

        )


    print(
        "Saved weed density heatmap "
        "(continuous):",
        DENSITY_RASTER
    )



    # COLORED HEATMAP


    density_uint8 = (

        weed_density_grid

        *

        255

    ).astype(
        np.uint8
    )


    heatmap_color = cv2.applyColorMap(

        density_uint8,

        cv2.COLORMAP_TURBO

    )


    zero_mask = (
        weed_count == 0
    )


    heatmap_color[
        zero_mask
    ] = [

        0,
        0,
        0

    ]


    heatmap_color = np.transpose(

        heatmap_color,

        (2, 0, 1)

    )


    color_meta = (
        rate_meta.copy()
    )


    color_meta.update(

        count=3,

        dtype="uint8"

    )


    with rasterio.open(

        COLORED_HEATMAP,

        "w",

        **color_meta

    ) as dst:


        dst.write(
            heatmap_color
        )


    print(
        "Saved colored weed density "
        "heatmap:",
        COLORED_HEATMAP
    )



    # CREATE SHAPEFILE


    polygons = []

    rates = []

    densities = []


    for r in range(
        grid_rows
    ):


        for c in range(
            grid_cols
        ):


            if rate_grid[r, c] == 0:

                continue


            x_min = (

                grid_transform.c

                +

                c * GRID_SIZE_M

            )


            y_max = (

                grid_transform.f

                -

                r * GRID_SIZE_M

            )


            x_max = (

                x_min

                +

                GRID_SIZE_M

            )


            y_min = (

                y_max

                -

                GRID_SIZE_M

            )


            polygons.append(

                box(

                    x_min,

                    y_min,

                    x_max,

                    y_max

                )

            )


            rates.append(

                float(
                    rate_grid[r, c]
                )

            )


            densities.append(

                float(
                    weed_density_grid[
                        r,
                        c
                    ]
                )

            )


    gdf = gpd.GeoDataFrame(

        {

            "RATE": rates,

            "DENSITY": densities

        },

        geometry=polygons,

        crs=crs

    )


    gdf.to_file(
        SHAPEFILE_PRESCRIPTION
    )


    print(
        "Saved shapefile prescription:",
        SHAPEFILE_PRESCRIPTION
    )



    # WGS84


    gdf_wgs84 = gdf.to_crs(
        "EPSG:4326"
    )


    gdf_wgs84.to_file(
        SHAPEFILE_WGS84
    )


    print(
        "Saved WGS84 shapefile "
        "for UAV import:",
        SHAPEFILE_WGS84
    )


    print(
        "\nPrescription + heatmap "
        "generation complete."
    )



    # STOP PROCESSING TIMER


    synchronize_gpu()


    processing_end = time.perf_counter()


    downstream_processing_seconds = (

        processing_end
        -
        processing_start

    )



    # STOP RESOURCE MONITORING


    monitor_resources = False


    monitor_thread.join(
        timeout=3
    )



    # TOTAL OPERATIONAL TIME
    #
    # Warm-up is NOT included.
    #
    # Total =
    #
    # tiling
    # +
    # inference
    # +
    # grid processing
    # +
    # stitching
    # +
    # prescription generation
    # +
    # output generation


    total_processing_seconds = (

        tiling_seconds

        +

        downstream_processing_seconds

    )


    total_processing_ms = (

        total_processing_seconds

        *

        1000.0

    )



    # MODEL PERFORMANCE RESULTS
    #
    # IMPORTANT:
    #
    # We report TWO different types of performance:
    #
    # 1. ENGINE/MODEL INFERENCE
    #       results.speed["inference"]
    #
    # 2. APPLICATION MODEL-CALL LATENCY
    #       complete wall-clock model(...)
    #
    # They must not be treated as the same measurement.




    # PREPROCESSING


    mean_preprocess_ms = safe_mean(
        per_tile_preprocess_ms
    )


    median_preprocess_ms = safe_median(
        per_tile_preprocess_ms
    )



    # ENGINE/MODEL INFERENCE


    mean_engine_inference_ms = safe_mean(
        per_tile_engine_inference_ms
    )


    median_engine_inference_ms = safe_median(
        per_tile_engine_inference_ms
    )


    min_engine_inference_ms = safe_min(
        per_tile_engine_inference_ms
    )


    max_engine_inference_ms = safe_max(
        per_tile_engine_inference_ms
    )


    if len(
        per_tile_engine_inference_ms
    ) > 0:

        p95_engine_inference_ms = float(

            np.nanpercentile(

                per_tile_engine_inference_ms,

                95

            )

        )

    else:

        p95_engine_inference_ms = np.nan



    # ENGINE FPS
    #
    # FPS = number of images / total engine inference time
    #
    # Equivalent to approximately:
    #
    #       1000 / mean inference latency
    #
    # when every tile contributes one timing measurement.


    if total_engine_inference_ms > 0:

        engine_fps = (

            len(tile_files)

            /

            (
                total_engine_inference_ms
                /
                1000.0
            )

        )

    else:

        engine_fps = np.nan



    # POSTPROCESSING


    mean_postprocess_ms = safe_mean(
        per_tile_postprocess_ms
    )


    median_postprocess_ms = safe_median(
        per_tile_postprocess_ms
    )


    max_postprocess_ms = safe_max(
        per_tile_postprocess_ms
    )


    if len(
        per_tile_postprocess_ms
    ) > 0:

        p95_postprocess_ms = float(

            np.nanpercentile(

                per_tile_postprocess_ms,

                95

            )

        )

    else:

        p95_postprocess_ms = np.nan



    # COMPLETE MODEL-CALL LATENCY
    #
    # Includes:
    #
    # preprocessing
    # +
    # inference
    # +
    # postprocessing
    # +
    # Python/runtime overhead


    total_model_call_ms = (

        total_model_call_seconds

        *
        1000.0

    )


    mean_model_call_ms = safe_mean(
        per_tile_model_call_ms
    )


    median_model_call_ms = safe_median(
        per_tile_model_call_ms
    )


    min_model_call_ms = safe_min(
        per_tile_model_call_ms
    )


    max_model_call_ms = safe_max(
        per_tile_model_call_ms
    )


    if len(
        per_tile_model_call_ms
    ) > 0:

        p95_model_call_ms = float(

            np.nanpercentile(

                per_tile_model_call_ms,

                95

            )

        )

    else:

        p95_model_call_ms = np.nan



    # APPLICATION FPS
    #
    # This is based on the complete model(...) call.


    if total_model_call_seconds > 0:

        application_fps = (

            len(tile_files)

            /

            total_model_call_seconds

        )

    else:

        application_fps = np.nan



    # DETECTION STATISTICS


    mean_detections_per_tile = safe_mean(
        per_tile_detection_count
    )


    max_detections_per_tile = safe_max(
        per_tile_detection_count
    )



    # PIPELINE THROUGHPUT


    if total_processing_seconds > 0:


        pipeline_tiles_per_second = (

            len(tile_files)

            /

            total_processing_seconds

        )


    else:


        pipeline_tiles_per_second = np.nan



    # RESOURCE VALUES


    ram_values = valid_values(

        resource_samples,

        "ram_used_mb"

    )


    process_ram_values = valid_values(

        resource_samples,

        "process_ram_mb"

    )


    gpu_values = valid_values(

        resource_samples,

        "gpu_util_percent"

    )


    power_values_mw = valid_values(

        resource_samples,

        "power_mw"

    )



    # RAM


    average_ram_mb = safe_mean(
        ram_values
    )


    peak_ram_mb = safe_max(
        ram_values
    )


    average_process_ram_mb = safe_mean(
        process_ram_values
    )


    peak_process_ram_mb = safe_max(
        process_ram_values
    )



    # GPU UTILIZATION


    average_gpu_utilization = safe_mean(
        gpu_values
    )


    peak_gpu_utilization = safe_max(
        gpu_values
    )



    # POWER


    if len(power_values_mw) > 0:


        average_power_w = (

            safe_mean(
                power_values_mw
            )

            /

            1000.0

        )


        peak_power_w = (

            safe_max(
                power_values_mw
            )

            /

            1000.0

        )


    else:


        average_power_w = np.nan

        peak_power_w = np.nan



    # ENERGY
    #
    # E = P x t
    #
    # IMPORTANT:
    #
    # Power monitoring begins after warm-up.
    #
    # Tiling occurs before monitoring because the model requires a real tile
    # for warm-up.
    #
    # Therefore average processing power measured here corresponds to
    # inference + postprocessing.
    #
    # For total operational energy, we use the measured processing power
    # multiplied by total operational time as an estimate.
    #
    # See discussion below the code regarding precise tiling-energy measurement.



    # ENERGY
    #
    # Monitoring covers the downstream processing interval after warm-up,
    # but not the tiling stage. Therefore:
    #
    # downstream_energy_* = measured downstream average power x downstream time
    #
    # estimated_total_energy_* = downstream average power x total operational
    # time. This second quantity is an estimate because tiling power is not
    # independently measured in the current workflow.


    if not np.isnan(
        average_power_w
    ):

        downstream_energy_j = (
            average_power_w
            *
            downstream_processing_seconds
        )

        downstream_energy_wh = (
            downstream_energy_j
            /
            3600.0
        )

        estimated_total_energy_j = (
            average_power_w
            *
            total_processing_seconds
        )

        estimated_total_energy_wh = (
            estimated_total_energy_j
            /
            3600.0
        )

    else:

        downstream_energy_j = np.nan
        downstream_energy_wh = np.nan

        estimated_total_energy_j = np.nan
        estimated_total_energy_wh = np.nan



    # SAVE RESOURCE LOG


    RESOURCE_CSV = os.path.join(

        RUN_DIR,

        "resource_usage.csv"

    )


    save_resource_csv(

        resource_samples,

        RESOURCE_CSV

    )



    # SAVE DETAILED PER-TILE PERFORMANCE LOG


    TILE_PERFORMANCE_CSV = os.path.join(

        RUN_DIR,

        "tile_performance.csv"

    )


    save_tile_performance_csv(

        tile_files,

        per_tile_preprocess_ms,

        per_tile_engine_inference_ms,

        per_tile_postprocess_ms,

        per_tile_model_call_ms,

        per_tile_detection_count,

        TILE_PERFORMANCE_CSV

    )



    # RUN SUMMARY


    run_summary = {

        "model_version":
            MODEL_VERSION,

        "model_path":
            MODEL_PATH,

        "run":
            BENCHMARK_RUN,

        "orthomosaic":
            INPUT_TIF,

        "number_of_tiles":
            len(tile_files),

        "model_load_time_ms":
            MODEL_LOAD_TIME_MS,

        "warmup_runs":
            WARMUP_RUNS,

        "warmup_time_ms":
            warmup_time_ms,

        "tiling_time_ms":
            tiling_ms,


        # PREPROCESSING


        "mean_preprocess_ms":
            mean_preprocess_ms,

        "median_preprocess_ms":
            median_preprocess_ms,



        # PURE MODEL / ENGINE INFERENCE


        "total_engine_inference_ms":
            total_engine_inference_ms,

        "mean_engine_inference_ms":
            mean_engine_inference_ms,

        "median_engine_inference_ms":
            median_engine_inference_ms,

        "minimum_engine_inference_ms":
            min_engine_inference_ms,

        "maximum_engine_inference_ms":
            max_engine_inference_ms,

        "p95_engine_inference_ms":
            p95_engine_inference_ms,

        "engine_fps":
            engine_fps,



        # SEGMENTATION POSTPROCESSING


        "mean_postprocess_ms":
            mean_postprocess_ms,

        "median_postprocess_ms":
            median_postprocess_ms,

        "maximum_postprocess_ms":
            max_postprocess_ms,

        "p95_postprocess_ms":
            p95_postprocess_ms,



        # COMPLETE model(...) CALL


        "total_model_call_ms":
            total_model_call_ms,

        "mean_model_call_ms":
            mean_model_call_ms,

        "median_model_call_ms":
            median_model_call_ms,

        "minimum_model_call_ms":
            min_model_call_ms,

        "maximum_model_call_ms":
            max_model_call_ms,

        "p95_model_call_ms":
            p95_model_call_ms,

        "application_fps":
            application_fps,



        # DETECTIONS


        "mean_detections_per_tile":
            mean_detections_per_tile,

        "maximum_detections_per_tile":
            max_detections_per_tile,

        "downstream_processing_time_ms":
            (
                downstream_processing_seconds
                *
                1000.0
            ),

        "total_prescription_time_ms":
            total_processing_ms,

        "pipeline_tiles_per_second":
            pipeline_tiles_per_second,

        "average_system_ram_mb":
            average_ram_mb,

        "peak_system_ram_mb":
            peak_ram_mb,

        "average_process_ram_mb":
            average_process_ram_mb,

        "peak_process_ram_mb":
            peak_process_ram_mb,

        "average_gpu_utilization_percent":
            average_gpu_utilization,

        "peak_gpu_utilization_percent":
            peak_gpu_utilization,

        "average_power_w":
            average_power_w,

        "peak_power_w":
            peak_power_w,

        "downstream_energy_j":
            downstream_energy_j,

        "downstream_energy_wh":
            downstream_energy_wh,

        "estimated_total_energy_j":
            estimated_total_energy_j,

        "estimated_total_energy_wh":
            estimated_total_energy_wh,

        "crop_detections":
            int(
                crop_count.sum()
            ),

        "weed_detections":
            int(
                weed_count.sum()
            )

    }


    all_run_results.append(
        run_summary
    )



    # SAVE INDIVIDUAL RUN SUMMARY


    RUN_SUMMARY_CSV = os.path.join(

        RUN_DIR,

        "benchmark_summary.csv"

    )


    with open(

        RUN_SUMMARY_CSV,

        "w",

        newline=""

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=run_summary.keys()

        )


        writer.writeheader()


        writer.writerow(
            run_summary
        )



    # PRINT RUN RESULTS


    print("\n")
    print("=" * 80)
    print(
        f"RUN {BENCHMARK_RUN} RESULTS"
    )
    print("=" * 80)


    print(
        f"\nModel: "
        f"{MODEL_VERSION}"
    )


    print(
        f"Tiles: "
        f"{len(tile_files)}"
    )


    print("\n--- TIMING ---")


    print(
        f"Tiling time: "
        f"{tiling_ms:.2f} ms"
    )


    print(
        f"Total prescription-map time: "
        f"{total_processing_ms:.2f} ms"
    )


    print("\n--- MODEL / ENGINE INFERENCE ---")


    print(
        f"Total engine inference time: "
        f"{total_engine_inference_ms:.2f} ms"
    )


    print(
        f"Mean engine inference latency: "
        f"{mean_engine_inference_ms:.3f} ms/tile"
    )


    print(
        f"Median engine inference latency: "
        f"{median_engine_inference_ms:.3f} ms/tile"
    )


    print(
        f"P95 engine inference latency: "
        f"{p95_engine_inference_ms:.3f} ms/tile"
    )


    print(
        f"Engine FPS: "
        f"{engine_fps:.2f}"
    )


    print("\n--- PREPROCESSING ---")


    print(
        f"Mean preprocessing time: "
        f"{mean_preprocess_ms:.3f} ms/tile"
    )


    print("\n--- SEGMENTATION POSTPROCESSING ---")


    print(
        f"Mean postprocessing time: "
        f"{mean_postprocess_ms:.3f} ms/tile"
    )


    print(
        f"Median postprocessing time: "
        f"{median_postprocess_ms:.3f} ms/tile"
    )


    print(
        f"P95 postprocessing time: "
        f"{p95_postprocess_ms:.3f} ms/tile"
    )


    print(
        f"Maximum postprocessing time: "
        f"{max_postprocess_ms:.3f} ms/tile"
    )


    print("\n--- COMPLETE MODEL CALL ---")


    print(
        f"Mean model-call latency: "
        f"{mean_model_call_ms:.3f} ms/tile"
    )


    print(
        f"Median model-call latency: "
        f"{median_model_call_ms:.3f} ms/tile"
    )


    print(
        f"P95 model-call latency: "
        f"{p95_model_call_ms:.3f} ms/tile"
    )


    print(
        f"Application FPS: "
        f"{application_fps:.2f}"
    )


    print("\n--- SYSTEM RAM ---")


    print(
        f"Average RAM: "
        f"{average_ram_mb:.2f} MB"
    )


    print(
        f"Peak RAM: "
        f"{peak_ram_mb:.2f} MB"
    )


    print("\n--- PROCESS RAM ---")


    print(
        f"Average process RAM: "
        f"{average_process_ram_mb:.2f} MB"
    )


    print(
        f"Peak process RAM: "
        f"{peak_process_ram_mb:.2f} MB"
    )


    print("\n--- GPU ---")


    print(
        f"Average GPU utilization: "
        f"{average_gpu_utilization:.2f} %"
    )


    print(
        f"Peak GPU utilization: "
        f"{peak_gpu_utilization:.2f} %"
    )


    print("\n--- POWER ---")


    print(
        f"Average power: "
        f"{average_power_w:.3f} W"
    )


    print(
        f"Peak power: "
        f"{peak_power_w:.3f} W"
    )


    print("\n--- ENERGY ---")


    print(
        f"Measured downstream energy: "
        f"{downstream_energy_j:.3f} J"
    )

    print(
        f"Measured downstream energy: "
        f"{downstream_energy_wh:.6f} Wh"
    )

    print(
        f"Estimated total operational energy: "
        f"{estimated_total_energy_j:.3f} J"
    )

    print(
        f"Estimated total operational energy: "
        f"{estimated_total_energy_wh:.6f} Wh"
    )


    print(
        "\nRun results saved in:"
    )


    print(
        RUN_DIR
    )



# SAVE ALL RUN RESULTS


ALL_RUNS_CSV = os.path.join(

    BENCHMARK_ROOT,

    "all_benchmark_runs.csv"

)


with open(

    ALL_RUNS_CSV,

    "w",

    newline=""

) as f:


    writer = csv.DictWriter(

        f,

        fieldnames=all_run_results[0].keys()

    )


    writer.writeheader()


    writer.writerows(
        all_run_results
    )



# FINAL MEAN +/- SD SUMMARY


print("\n")
print("=" * 80)
print("FINAL BENCHMARK SUMMARY")
print("=" * 80)


print(
    f"\nModel: {MODEL_VERSION}"
)


print(
    f"Number of complete runs: "
    f"{NUM_BENCHMARK_RUNS}"
)



# METRICS TO SUMMARIZE


summary_metrics = [

    # Complete prescription workflow
    "total_prescription_time_ms",
    "downstream_processing_time_ms",
    "pipeline_tiles_per_second",

    # Engine/model inference
    "total_engine_inference_ms",
    "mean_engine_inference_ms",
    "median_engine_inference_ms",
    "minimum_engine_inference_ms",
    "maximum_engine_inference_ms",
    "p95_engine_inference_ms",
    "engine_fps",

    # Preprocessing
    "mean_preprocess_ms",

    # Segmentation postprocessing
    "mean_postprocess_ms",
    "median_postprocess_ms",
    "p95_postprocess_ms",
    "maximum_postprocess_ms",

    # Complete model call
    "total_model_call_ms",
    "mean_model_call_ms",
    "median_model_call_ms",
    "minimum_model_call_ms",
    "maximum_model_call_ms",
    "p95_model_call_ms",
    "application_fps",

    # System resources
    "average_system_ram_mb",
    "peak_system_ram_mb",

    "average_process_ram_mb",
    "peak_process_ram_mb",

    "average_gpu_utilization_percent",
    "peak_gpu_utilization_percent",

    # Power
    "average_power_w",
    "peak_power_w",

    # Energy
    "downstream_energy_j",
    "downstream_energy_wh",
    "estimated_total_energy_j",
    "estimated_total_energy_wh"

]



# CALCULATE FINAL MEAN +/- SD


final_summary_rows = []


for metric in summary_metrics:

    values = [
        run[metric]
        for run in all_run_results
        if not np.isnan(run[metric])
    ]

    if len(values) > 0:

        metric_mean = statistics.mean(values)

        if len(values) > 1:
            metric_sd = statistics.stdev(values)
        else:
            metric_sd = 0.0

    else:

        metric_mean = np.nan
        metric_sd = np.nan

    final_summary_rows.append(
        {
            "metric": metric,
            "mean": metric_mean,
            "standard_deviation": metric_sd
        }
    )

    print(f"\n{metric}")
    print(f"Mean = {metric_mean:.4f}")
    print(f"SD   = {metric_sd:.4f}")



# SAVE FINAL SUMMARY


FINAL_SUMMARY_CSV = os.path.join(

    BENCHMARK_ROOT,

    "final_mean_sd_summary.csv"

)


with open(

    FINAL_SUMMARY_CSV,

    "w",

    newline=""

) as f:


    writer = csv.DictWriter(

        f,

        fieldnames=[

            "metric",

            "mean",

            "standard_deviation"

        ]

    )


    writer.writeheader()


    writer.writerows(
        final_summary_rows
    )



# FINAL OUTPUT


print("\n")
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)


print(
    "\nAll individual runs:"
)


print(
    ALL_RUNS_CSV
)


print(
    "\nMean +/- SD summary:"
)


print(
    FINAL_SUMMARY_CSV
)


print(
    "\nBenchmark completed successfully."
)