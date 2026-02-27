# Mantid VESUVIO

[![Nightly Build Status](https://github.com/mantidproject/vesuvio/actions/workflows/deploy_package_nightly.yml/badge.svg)](https://github.com/mantidproject/vesuvio/actions/workflows/deploy_package_nightly.yml)
[![Coverage Status](https://coveralls.io/repos/github/mantidproject/vesuvio/badge.svg?branch=main)](https://coveralls.io/github/mantidproject/vesuvio?branch=main)
[![Anaconda-Server Badge](https://anaconda.org/mantid/mvesuvio/badges/latest_release_relative_date.svg)](https://anaconda.org/mantid/mvesuvio)
[![Anaconda-Server Badge](https://anaconda.org/mantid/mvesuvio/badges/version.svg)](https://anaconda.org/mantid/mvesuvio)
[![Anaconda-Server Badge](https://anaconda.org/mantid/mvesuvio/badges/downloads.svg)](https://anaconda.org/mantid/mvesuvio)

This repository contains:
- `mvesuvio` package containing Neutron Compron Profile (NCP) analysis procedures for Vesuvio, published nightly.
- Vesuvio calibration scripts, under the `tools` folder

Currently only the NCP analysis is usable, the calibration scripts are not yet ready. 

## Installing mvesuvio package (Try this option first)
The `mvesuvio` package is meant to be used inside the [Mantid software](https://www.mantidproject.org/index.html), so you'll need to install Mantid first if you haven't.

Once you have a working version of Mantid, go to the `IPython` tab on the bottom center of the window and inside the tab type the following command:

**If you're on Linux:**

`mamba install mantid/label/nightly::mvesuvio`

**If you're on Windows:**

`pip install mvesuvio`

To check if the package was successfully installed and to do all the necessary setup, type:

`!mvesuvio config`

If you see some output then the package is successfully installed and setup!

**WARNING: This way of installing the package is not officially supported by Mantid, it just so happens to work. If this stops working or you encounter issues please contact me, as I would like to know. You can still install mvesuvio by following the instructions in the next section.**

### Installing mantid and mvesuvio using conda/mamba (If the first option failed)

If the previous installation attempt was unsuccessful, then you'll have to use Mantid inside a conda/mamba environment. This is the official recommended way of using the mantid with the mvesuvio package.

To install `mamba`, follow the steps at:
https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

To check you have mamba installed, run:

`mamba --version`

You should see some output with the versions available in your system.

Now create a new conda environment, for example I'll call it `mantid-mvesuvio`:

`mamba create -n mantid-mvesuvio mantidworkbench mantid/label/nightly::mvesuvio`

And activate the environment you created:

`mamba activate mantid-mvesuvio`

You can now start Mantid with mvesuvio already installed by typing:

`workbench`

## Updating versions

If you want to check the current version of the package, go to the IPython editor in Mantid and type:

`!mvesuvio version`

The easiest way to update the mvesuvio package is to uninstall the current package in your environment and install it again.
You'll need to use `pip`, `conda` or `mamba` depending on which one you used to install the package.

So for example if you did `mamba install mvesuvio` then the uninstall command is:

`mamba uninstall mvesuvio`

Otherwise if you used pip:

`pip uninstall mvesuvio`

If you do not remember which command you used during the installation, then just run both `pip` and `mamba` uninstall commands, one of them will fail but the other one will succeed.

Once you have uninstalled the package, you can install it again to get the latest version.
If you're looking for a specific version, you can do:

`mamba install mvesuvio=1.0`

Which will install version 1.0 of the package.

### Quickstart (Running your first analysis)

To run your very first analysis (and check that everything is working), go to your home folder and find the folder `.mvesuvio`.
The `.` in front of the directory name means this folder might be hidden by your OS, so you might have to turn on the option in your file browser of showing hidden folders.

If the folder does not exist, go the the `IPython` tab in Mantid and type:

`!mvesuvio config`

You should see some output of default inputs, which will lead you to the `.mvesuvio` folder.

Once you have located the `.mvesuvio` folder, open Mantid workbench and inside it open the script `analysis_inputs.py` located inside `.mvesuvio`.

This script represents the basics for passing in the inputs of the analysis routine.
Click the run button on the workbench to start the execution of the script.
(Check that you have the archive search enabled, the facility is set to ISIS and the instrument set to VESUVIO, otherwise the Vesuvio runs might not be found).
This scipt is an example of a well-behaved sample and it should run without issues.

If the run was successfull, you will notice that a new folder was created inside `.mvesuvio` containing all sorts of outputs for this script.

**IMPORTANT:To run a new sample with different inputs, you should *copy* the example script `analysis_inputs.py` and place it in *any* folder of your choice outside `.mvesuvio`. 
For providing the instrument parameters files, place them inside `.mvesuvio/ip_files/`.**
(You can change the directory of the instrument files too, consult next section).

For a more detailed explanation on what the inputs in the `analysis_inputs.py` mean, read [USERGUIDE.md](./USERGUIDE.md)

## Tips and useful commands

A very useful command is:

`mvesuvio version`

Which returns the version of mvesuvio that you have currently installed.
If you're running a Python script or have access to a Python interpreter (like the IPython tab in Mantid), you can do:

```
import mvesuvio
mvesuvio.version()
```
In a Python interpreter like the IPython tab in Mantid, you can also run terminal commands by starting the command with `!`:

`!mvesuvio version`

Or to see the current configuration:

`!mvesuvio config`


## Advanced Usage (CLI)
If you're using a conda environment and have installed Mantid and mvesuvio with conda/mamba, then you might be interested in the CLI options of mvesuvio.
With your environment activated you can type in the terminal:

`mvesuvio -h`

And this will list all of the currently available commands.
Currently the commands that are stable are `config`, `run` and `version`. All other commands are available but are still in development.

#### mvesuvio config

The `config` command is used to display or set the analysis inputs script or the folder to look for the instrument parameters.
You can do so by providing two optional arguments:
- `--analysis-inputs` - Set the location of the analysis inputs python file (default is `analysis_inputs.py` in `.mvesuvio` folder).
- `--ip-folder` - Set the directory for the instrument parameter files (default is `ip_files` in `.mvesuvio` folder).

If you run `mvesuvio config` with no arguments then the output will tell you the current locations for the analysis inputs file and the instrument parameters folder.

Usage examples:
- `mvesuvio config --ip-folder C:\IPFolder` - Set instrument parameters folder.
- `mvesuvio config --analysis-inputs C:\Vesuvio\experiment\inputs.py` - Set inputs file.

#### mvesuvio run

The `run` command does not take any arguments and simply runs the routine based on the current configuration.

Usage example:
- `mvesuvio run`- Run NCP analysis.

### Python API

The commands available in the CLI can be triggered from Python by calling the method with the same name.
So for example to set the configuration from a Python script (or from IPython tab):

```
import mvesuvio
mv.config(analysis_inputs='C:\Vesuvio\experiment\inputs.py', ip_folder='C:\IPFolder')
```
In fact, this functionality is what you see at the end of the `analysis_inputs.py` file, which sets the inpucts script to the currently openned script:

```
import mvesuvio
from pathlib import Path

mvesuvio.config(analysis_inputs=str(Path(__file__)))
mvesuvio.run()
```
