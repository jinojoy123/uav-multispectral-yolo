# STACK FIVE ORTHOMOSAIC BANDS FOR RGBRN INFERENCE
#
# Purpose:
#   Create ONE georeferenced 5-band RGBRN orthomosaic for the prescription-map
#   inference script.
#
# Band order:
#   1 = Red
#   2 = Green
#   3 = Blue
#   4 = RedEdge
#   5 = NIR
#
# IMPORTANT:
#   - No train/val/test split
#   - No labels
#   - No 640 x 640 resizing
#   - No RGB-only dataset is created
#   - Spatial dimensions/georeferencing are preserved
#   - Each band is converted to uint8 using the SAME per-band min-max
#     normalization principle used in RGBRN_data_arrangement.py
#
# The output is a single 5-band GeoTIFF that can then be tiled internally by
# prescription_map_RGBRN_edgedevices_corrected.py.

import os
import numpy as np
import rasterio


# CONFIGURATION


# Input single-band orthomosaics
# Change ONLY these paths to your actual registered/aligned orthomosaics.


RED_PATH = r"/red.tif"
GREEN_PATH = r"/green.tif"
BLUE_PATH = r"/blue.tif"
REDEDGE_PATH = r"/re.tif"
NIR_PATH = r"/nir.tif"


# Output stacked orthomosaic
OUTPUT_RGBRN_TIF = r"/RGBRN.tif"


# Output datatype expected by the current RGBRN training pipeline
OUTPUT_DTYPE = np.uint8


# Process the orthomosaic in blocks/windows so the complete large image does
# not need to be held in RAM.
BLOCK_SIZE = 1024


# INPUT BAND DEFINITIONS

BAND_PATHS = [
    RED_PATH,
    GREEN_PATH,
    BLUE_PATH,
    REDEDGE_PATH,
    NIR_PATH
]

BAND_NAMES = [
    "Red",
    "Green",
    "Blue",
    "RedEdge",
    "NIR"
]


# HELPER FUNCTIONS

def check_input_files():

    print("\nChecking input orthomosaics...")

    for band_name, band_path in zip(
        BAND_NAMES,
        BAND_PATHS
    ):

        if not os.path.isfile(
            band_path
        ):

            raise FileNotFoundError(
                f"{band_name} orthomosaic not found:\n"
                f"{band_path}"
            )

        print(
            f"{band_name}: {band_path}"
        )


def get_reference_information():

    with rasterio.open(
        RED_PATH
    ) as reference:

        information = {
            "width": reference.width,
            "height": reference.height,
            "crs": reference.crs,
            "transform": reference.transform,
            "dtype": reference.dtypes[0],
            "nodata": reference.nodata,
            "profile": reference.profile.copy()
        }

    return information


def validate_alignment(reference_information):

    print("\nValidating band alignment...")

    for band_name, band_path in zip(
        BAND_NAMES,
        BAND_PATHS
    ):

        with rasterio.open(
            band_path
        ) as src:

            if src.count < 1:

                raise ValueError(
                    f"{band_name} contains no raster band."
                )


            if (
                src.width
                !=
                reference_information["width"]
            ):

                raise ValueError(
                    f"{band_name} width does not match Red reference. "
                    f"{src.width} != "
                    f"{reference_information['width']}"
                )


            if (
                src.height
                !=
                reference_information["height"]
            ):

                raise ValueError(
                    f"{band_name} height does not match Red reference. "
                    f"{src.height} != "
                    f"{reference_information['height']}"
                )


            if (
                src.crs
                !=
                reference_information["crs"]
            ):

                raise ValueError(
                    f"{band_name} CRS does not match Red reference.\n"
                    f"{src.crs}\n"
                    f"!=\n"
                    f"{reference_information['crs']}"
                )


            if not src.transform.almost_equals(
                reference_information["transform"]
            ):

                raise ValueError(
                    f"{band_name} geotransform does not match "
                    f"the Red reference."
                )


            print(
                f"{band_name}: alignment OK | "
                f"shape=({src.height}, {src.width}) | "
                f"dtype={src.dtypes[0]}"
            )


    print(
        "\nAll five bands have matching dimensions, CRS, "
        "and geotransform."
    )


def find_band_min_max(
    band_path
):
    """
    Determine the global valid-data minimum and maximum for one orthomosaic.

    This allows block-wise conversion while reproducing global per-band
    min-max normalization:
        output = 255 * (x - min) / (max - min)

    Nodata and non-finite values are excluded.
    """

    global_min = np.inf
    global_max = -np.inf


    with rasterio.open(
        band_path
    ) as src:

        for _, window in src.block_windows(
            1
        ):

            band = src.read(
                1,
                window=window,
                masked=True
            )


            valid = band.compressed()


            if valid.size == 0:

                continue


            valid = valid[
                np.isfinite(
                    valid
                )
            ]


            if valid.size == 0:

                continue


            block_min = float(
                valid.min()
            )

            block_max = float(
                valid.max()
            )


            if block_min < global_min:

                global_min = block_min


            if block_max > global_max:

                global_max = block_max


    if (
        not np.isfinite(global_min)
        or
        not np.isfinite(global_max)
    ):

        raise ValueError(
            f"No valid pixels found in {band_path}"
        )


    return global_min, global_max


def normalize_band_to_uint8(
    band,
    valid_mask,
    band_min,
    band_max
):
    """
    Match the per-band cv2.NORM_MINMAX behavior used in the training-data
    preparation, while keeping nodata pixels at 0.
    """

    output = np.zeros(
        band.shape,
        dtype=np.uint8
    )


    if band_max <= band_min:

        return output


    band_float = band.astype(
        np.float32
    )


    scaled = (
        255.0
        *
        (
            band_float
            -
            band_min
        )
        /
        (
            band_max
            -
            band_min
        )
    )


    scaled = np.clip(
        scaled,
        0,
        255
    )


    output[
        valid_mask
    ] = scaled[
        valid_mask
    ].astype(
        np.uint8
    )


    return output


# MAIN

def main():

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "RGBRN ORTHOMOSAIC STACKING FOR INFERENCE"
    )

    print(
        "=" * 80
    )


    
    # Check files
    

    check_input_files()


    
    # Read reference geospatial information
    

    reference_information = (
        get_reference_information()
    )


    print(
        "\nReference orthomosaic:"
    )

    print(
        f"Width: "
        f"{reference_information['width']}"
    )

    print(
        f"Height: "
        f"{reference_information['height']}"
    )

    print(
        f"CRS: "
        f"{reference_information['crs']}"
    )

    print(
        f"Transform: "
        f"{reference_information['transform']}"
    )


    
    # Verify that the five orthomosaics are already registered/aligned.
    #
    # This script intentionally DOES NOT resample or register the bands.
    

    validate_alignment(
        reference_information
    )


    
    # Calculate global min/max separately for each band.
    #
    # Your original RGBRN_data_arrangement.py normalizes each channel
    # independently to 0-255. We preserve that behavior here.
    

    print(
        "\nCalculating global per-band min/max values..."
    )


    band_statistics = []


    for band_name, band_path in zip(
        BAND_NAMES,
        BAND_PATHS
    ):

        band_min, band_max = (
            find_band_min_max(
                band_path
            )
        )


        band_statistics.append(
            (
                band_min,
                band_max
            )
        )


        print(
            f"{band_name}: "
            f"min={band_min:.6f}, "
            f"max={band_max:.6f}"
        )


    
    # Create output profile from the Red reference.
    

    output_profile = (
        reference_information[
            "profile"
        ].copy()
    )

    output_profile.update(
    driver="GTiff",
    count=5,
    dtype="uint8",
    nodata=0,
    compress="deflate",

    # Use valid internal GeoTIFF tile dimensions.
    # TIFF block sizes must be multiples of 16.
    tiled=True,
    blockxsize=512,
    blockysize=512,

    # Five-band orthomosaic can exceed the normal TIFF size limit.
    BIGTIFF="YES"
    )



    output_parent = os.path.dirname(
        OUTPUT_RGBRN_TIF
    )


    if output_parent:

        os.makedirs(
            output_parent,
            exist_ok=True
        )


    
    # Open all five source orthomosaics.
    

    sources = [
        rasterio.open(
            path
        )
        for path in BAND_PATHS
    ]


    try:

        width = (
            reference_information[
                "width"
            ]
        )

        height = (
            reference_information[
                "height"
            ]
        )


        print(
            "\nWriting stacked RGBRN orthomosaic..."
        )


        with rasterio.open(
            OUTPUT_RGBRN_TIF,
            "w",
            **output_profile
        ) as dst:


            # Add descriptive band names.
            for output_band_index, band_name in enumerate(
                BAND_NAMES,
                start=1
            ):

                dst.set_band_description(
                    output_band_index,
                    band_name
                )


            total_windows = (
                (
                    height
                    +
                    BLOCK_SIZE
                    -
                    1
                )
                //
                BLOCK_SIZE
            ) * (
                (
                    width
                    +
                    BLOCK_SIZE
                    -
                    1
                )
                //
                BLOCK_SIZE
            )


            window_number = 0


            for row_off in range(
                0,
                height,
                BLOCK_SIZE
            ):

                for col_off in range(
                    0,
                    width,
                    BLOCK_SIZE
                ):

                    window_number += 1


                    window_height = min(
                        BLOCK_SIZE,
                        height
                        -
                        row_off
                    )


                    window_width = min(
                        BLOCK_SIZE,
                        width
                        -
                        col_off
                    )


                    window = rasterio.windows.Window(
                        col_off=col_off,
                        row_off=row_off,
                        width=window_width,
                        height=window_height
                    )


                    for band_index in range(
                        5
                    ):

                        src = sources[
                            band_index
                        ]


                        band_masked = src.read(
                            1,
                            window=window,
                            masked=True
                        )


                        band_data = np.asarray(
                            band_masked.data
                        )


                        valid_mask = (
                            ~np.ma.getmaskarray(
                                band_masked
                            )
                        )


                        valid_mask &= np.isfinite(
                            band_data
                        )


                        band_min, band_max = (
                            band_statistics[
                                band_index
                            ]
                        )


                        band_uint8 = (
                            normalize_band_to_uint8(
                                band_data,
                                valid_mask,
                                band_min,
                                band_max
                            )
                        )


                        dst.write(
                            band_uint8,
                            band_index + 1,
                            window=window
                        )


                    if (
                        window_number == 1
                        or
                        window_number % 25 == 0
                        or
                        window_number == total_windows
                    ):

                        print(
                            f"Processed window "
                            f"{window_number}/"
                            f"{total_windows}"
                        )


    finally:

        for src in sources:

            src.close()


    
    # Verify output
    

    print(
        "\nVerifying stacked output..."
    )


    with rasterio.open(
        OUTPUT_RGBRN_TIF
    ) as stacked:

        print(
            f"Output: {OUTPUT_RGBRN_TIF}"
        )

        print(
            f"Shape: "
            f"{stacked.height} x "
            f"{stacked.width}"
        )

        print(
            f"Bands: {stacked.count}"
        )

        print(
            f"Dtypes: {stacked.dtypes}"
        )

        print(
            f"CRS: {stacked.crs}"
        )

        print(
            f"Band descriptions: "
            f"{stacked.descriptions}"
        )


        if stacked.count != 5:

            raise RuntimeError(
                "Output validation failed: "
                "stacked orthomosaic does not have five bands."
            )


        expected_descriptions = tuple(
            BAND_NAMES
        )


        if (
            stacked.descriptions
            !=
            expected_descriptions
        ):

            print(
                "WARNING: Band descriptions do not exactly "
                "match the expected names."
            )


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "RGBRN STACKING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )

    print(
        "\nUse this file as INPUT_TIF in "
        "prescription_map_RGBRN_edgedevices_corrected.py:"
    )

    print(
        OUTPUT_RGBRN_TIF
    )

    print(
        "\nBecause this output is already uint8, set:"
    )

    print(
        'RGBRN_SCALING_MODE = "none"'
    )


if __name__ == "__main__":

    main()
