# Vesuvio Calibration Scripts

In this subdirectory of the EVSVesuvio repository the following is provided:
- Vesuvio calibration scripts, composed of 3 python modules:
    1. `calibration_scripts.calibrate_vesuvio_fit`
    2. `calibration_scripts.calibrate_vesuvio_analysis`
    3. `calibration_scripts.calibrate_vesuvio_helper_functions`

- A script that is designed to run in `mantid workbench` to register the calibration algorithms:
    `unpackaged\vesuvio_calibration\load_calibration_algorithms.py`

- Unit tests to ensure the consistent functionality of the functions that make up the provided algorithms.
    1. `tests\unit\test_calibrate_vesuvio_fit.py`
    2. `tests\unit\test_calibrate_vesuvio_analysis.py`
    3. `tests\unit\test_calibrate_vesuvio_misc.py`

- System tests to ensure the correct output of the overall calibration script.
    1. `tests\system\test_system_fit.py`
    2. `tests\system\test_system_analysis.py`

## Running the Calibration Scripts in mantid `workbench`.

To run the calibration scripts in mantid, a script is provided: `unpackaged\vesuvio_calibration\load_calibration_algorithms.py`.

This script can be loaded into `workbench` via `File` > `Open Script`.

Before running this script, the script directory `unpackaged\vesuvio_calibration` must be added via:
1. `File` > `Manage User Directories`.
2. Navigate to `Python Script Directories` tab.
3. Click `Browse To Directory` and select the `unpackaged\vesuvio_calibration` directory.
4. Press `Ok`.

Upon running the script, the two calibration algorithms will be registered under the `VesuvioCalibration` heading in the `Algorithms` pane.


# Running Calibration Script Tests

## Create Conda Environment from the command line.

1. Ensure `conda`and or `mamba` is installed on your machine using `conda --verison`/`mamba --version`.

2. If no such module is found install `mamba` (recommended): https://mamba.readthedocs.io/en/latest/installation.html or `conda`: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

3. From the root of the repository run `conda env create -f environment.yml`.

## Running the unit tests from the command line.

1. Actviate the conda environment using `conda activate vesuvio-env`.

2. From `<root of the repository>/unpackaged/vesuvio_calibration` run `python -m unittest discover -s ./tests/unit`

## Running the system tests from the command line.

1. Actviate the conda environment using `conda activate vesuvio-env`.

2. From `<root of the repository>/unpackaged/vesuvio_calibration`  run `python -m unittest discover -s ./tests/system`
