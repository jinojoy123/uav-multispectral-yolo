# Installation Guide

**Project:** [UAV Multispectral
YOLO](https://github.com/jinojoy123/uav-multispectral-yolo) ---
RGB/RGBRN instance segmentation and UAV agricultural orthomosaic mapping
on NVIDIA Jetson devices.

This guide describes how to set up the **research code**. The repository
is not yet a one-command installer: model weights, datasets, and several
script-specific paths must be supplied by the user. The instructions
below do **not** require running a training job during installation.

## 1\. Repository layout

``` text
uav-multispectral-yolo/
├── README.md
├── requirements.txt
├── INSTALLATION.md
├── scripts/                    # Training, evaluation, export, and mapping scripts
└── uav\_MS\_yolo/
    └── code/
        └── ultralytics/        # Modified Ultralytics source
```

Keep the research scripts together in `scripts/`. All commands below
assume your terminal's **current working directory is the repository
root**, unless otherwise stated.

> \*\*Note:\*\* Some original scripts still refer to an older source path,
> `uav\_MS\_yolo/code/ultralytics\_uav\_MS\_yolo`. In this repository, the
> modified Python package is under `uav\_MS\_yolo/code/ultralytics`.
> Review and update those references before running the affected
> scripts.

## 2\. Download the repository

Using Git:

``` bash
git clone https://github.com/jinojoy123/uav-multispectral-yolo.git
cd uav-multispectral-yolo
```

Alternatively, on GitHub select **Code → Download ZIP**, extract the
archive, and open a terminal in the extracted repository directory.

## 3\. Create a Python environment (desktop / workstation)

Choose a Python version compatible with the **actual PyTorch and
modified Ultralytics installation** you plan to use. The repository does
not yet specify a single tested cross-platform Python version.

**Windows PowerShell:**

``` powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

**Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use the environment's Python executable
directly rather than changing system-wide execution policies.

## 4\. Install dependencies

The root `requirements.txt` lists the project's general Python packages,
including PyTorch, torchvision, NumPy, OpenCV, rasterio, tifffile,
pandas, Shapely, Pillow, PyYAML, Matplotlib, and SciPy.

**Desktop / workstation:** First select and install a
PyTorch/torchvision build compatible with your operating system and GPU,
using the [official PyTorch installation
selector](https://pytorch.org/get-started/locally/). Then run:

``` bash
python -m pip install -r requirements.txt
```

The requirements file is currently **unversioned**. This command
installs available package versions, **not necessarily the exact
versions used for the published experiments**. If you need strict
numerical reproducibility, record the working versions separately.

Do **not** assume the same installation command will install GPU-enabled
PyTorch correctly on an NVIDIA Jetson. See Section 8.

## 5\. Make Python use the modified Ultralytics implementation

This repository bundles a modified `ultralytics` Python package. For
RGBRN experiments, it is important not to import an unrelated
`ultralytics` installation from `site-packages`.

**Windows PowerShell (from the repository root):**

``` powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "uav\_MS\_yolo\\code")
python -c "import ultralytics; print(ultralytics.\_\_version\_\_); print(ultralytics.\_\_file\_\_)"
```

**Linux (from the repository root):**

``` bash
export PYTHONPATH="$PWD/uav\_MS\_yolo/code${PYTHONPATH:+:$PYTHONPATH}"
python -c "import ultralytics; print(ultralytics.\_\_version\_\_); print(ultralytics.\_\_file\_\_)"
```

The displayed source path should end with:

``` text
uav\_MS\_yolo/code/ultralytics/\_\_init\_\_.py
```

The modified package in the research archive identifies itself as
version **8.4.112**. A matching version number alone is not proof that
the correct source was loaded; **check the printed file path**.

`PYTHONPATH` is set only for the current terminal session. Set it again
in a new session or adjust your run configuration. Some scripts add
their own paths internally; correct outdated paths there if needed.

## 6\. Prepare the input data and models

The repository does not include a complete distributable dataset or
model weights. Supply your own data and trained/pretrained models.

### RGB input

Use the collected RGB images and the corresponding YOLO
instance-segmentation annotations.

### RGBRN input

Use **five registered bands in this exact order**:

1. Red (R)
2. Green (G)
3. Blue (B)
4. Red Edge (RE)
5. Near-Infrared (NIR)

Check that the training, validation, and test files use the band layout
and numerical representation expected by the modified loader. The
repository includes `scripts/data\_arrangement.py` and
`scripts/RGBRN\_orthomosaic\_for\_inference.py` for relevant
data-preparation stages.

**Preprocessing caution:** The data-arrangement code uses per-band
min--max normalization in part of its workflow, whereas some edge
mapping scripts specify `RGBRN\_SCALING\_MODE = "divide\_256"`. These
transformations are not equivalent. Determine the complete preprocessing
applied to the data used to train a particular model and match it during
inference. Do not change preprocessing when attempting to reproduce an
existing experiment without documenting the change.

### Dataset and weights paths

Several scripts contain **machine-specific or placeholder paths**,
including `/processed\_RGBRN`, `/yolomodel.pt`, and
`/Orthomosaic\_RGBRN.tif`. Update these paths for your local machine
**before execution**.

In particular:

* `scripts/train\_RGB.py` expects a model file and a dataset YAML.
* `scripts/train\_RGBRN.py` expects a pretrained model, a five-channel
dataset, and a dataset YAML; inspect the YAML-generation and path
assumptions before running.
* Test, TensorRT, and mapping scripts each require their own
model/input/output paths.

Dataset labels, class names, band order, and YAML metadata must agree.
The sugarbeet study uses crop and weed as its two target categories;
confirm the numeric class IDs and names in your actual annotation files.

## 7\. Training and evaluation (desktop)

From the repository root, after activating your environment and
configuring the scripts:

``` bash
python scripts/train\_RGB.py
python scripts/train\_RGBRN.py
```

For evaluation:

``` bash
python scripts/test\_RGB.py
python scripts/test\_RGBRN.py
```

These are **entry-point examples**, not guaranteed plug-and-play
commands. Before running, inspect each script's dataset paths, weights,
device selection, and output locations. The RGB script does not itself
force the modified package to be imported; use the `PYTHONPATH` setup
from Section 5 if needed.

Training and evaluation use the YOLO implementation and experiment
settings configured in the corresponding scripts. Do not assume all
architecture variants, weights, or datasets are selected automatically.

## 8\. NVIDIA Jetson deployment

The repository includes separate RGB and RGBRN mapping scripts for
**Jetson Orin Nano** and **Jetson AGX Orin**. No separate Jetson
requirements file is maintained at this stage.

**Important:** On Jetson, first install a PyTorch/torchvision build
compatible with the installed **JetPack, CUDA, Python, and device**.
TensorRT is generally supplied through a compatible NVIDIA
JetPack/TensorRT environment. Follow the documentation for your
installed JetPack release; do not blindly replace NVIDIA-provided GPU
packages with generic PyPI builds.

After establishing a working device-specific environment:

1. Install any remaining compatible general Python dependencies from
`requirements.txt`, taking care not to overwrite the working Jetson
PyTorch/torchvision stack.
2. Set `PYTHONPATH` to the local modified Ultralytics source (Section
5).
3. Copy or otherwise make available the trained model and orthomosaic
input files.
4. Edit the selected Jetson script's model, input, output, grid, and
preprocessing settings.
5. If using TensorRT, **export or build the engine for the target
deployment environment**; do not assume an engine from another
device or TensorRT version is portable.
6. Review the output maps and geospatial metadata before operational
use.

### TensorRT-related scripts

Script                               Purpose

\---

`scripts/export\_tensorrt\_RGB.py`     Export RGB model
`scripts/export\_tensorrt\_RGBRN.py`   Export RGBRN model
`scripts/test\_RGB\_TensorRT.py`       Evaluate RGB TensorRT engine
`scripts/test\_RGBRN\_TensorRT.py`     Evaluate RGBRN TensorRT engine

TensorRT FP32, FP16, and INT8 are research configurations, not a promise
that every precision mode works on every hardware/software combination.
INT8 export may require representative calibration data.

### Orthomosaic mapping scripts

\---

Device                  RGB                                          RGBRN

\---

Jetson Orin Nano        `scripts/Mapping\_pipeline\_RGB\_Orinnano.py`   `scripts/Mapping\_pipeline\_RGBRN\_OrinNano.py`

## Jetson AGX Orin         `scripts/Mapping\_pipeline\_RGB\_AGXorin.py`    `scripts/Mapping\_pipeline\_RGBRN\_AGXorin.py`

Verify filenames against the current GitHub tree if they have been
renamed. The mapping workflows depend on correctly georeferenced
orthomosaics and field-specific settings; prescription maps are
**research outputs** and require agronomic and geospatial validation
before use for spraying.

## 9\. Troubleshooting

**`ModuleNotFoundError: ultralytics`**  
Check that `uav\_MS\_yolo/code` exists, your environment is active, and
`PYTHONPATH` points to that directory.

**The wrong Ultralytics package loads**  
Print `ultralytics.\_\_file\_\_`. It must point inside this repository, not
an unrelated `site-packages` directory.

**File not found / dataset YAML missing**  
Update the absolute paths in the script and confirm the actual dataset
folder structure. Do not create empty directories as a substitute for
supplying data.

**Five-channel model fails or input dimensions mismatch**  
Check the band order, file shape, preprocessing, model weights, and the
modified source import path.

**PyTorch GPU unavailable or TensorRT import fails on Jetson**  
Check JetPack, Python, CUDA, PyTorch, and TensorRT compatibility using
the device-specific NVIDIA documentation.

**Raster outputs appear incorrect**  
Check band order, numerical scaling, CRS, pixel resolution, and the
output settings in the mapping script.

## 10\. Reproducibility and citation

For a reproducible experiment, record the model architecture, weights,
training/evaluation dataset, band preprocessing, seed, relevant package
versions, JetPack/CUDA/TensorRT versions (when applicable), and the
exact Git commit or release.

See [README.md](README.md), [CITATION.cff](CITATION.cff),
[LICENSE](LICENSE), and [THIRD\_PARTY\_NOTICES.md](THIRD_PARTY_NOTICES.md)
for project details and attribution.

