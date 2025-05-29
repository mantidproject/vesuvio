# Mantid VESUVIO

[![Nightly Build Status](https://github.com/mantidproject/vesuvio/actions/workflows/deploy_conda_nightly.yml/badge.svg)](https://github.com/mantidproject/vesuvio/actions/workflows/deploy_conda_nightly.yml)
[![Coverage Status](https://coveralls.io/repos/github/mantidproject/vesuvio/badge.svg?branch=main)](https://coveralls.io/github/mantidproject/vesuvio?branch=main)

This repository contains:
- `mvesuvio` package containing the Optimized NCP analysis procedures, published nightly.
- Vesuvio calibration scripts, under the `tools` folder

## Install mvesuvio package
The `mvesuvio` package is meant to be used inside the [Mantid software](https://www.mantidproject.org/index.html). If you've never used Mantid before, don't worry, you can follow these steps to install both Mantid and the `mvesuvio` package.

### Installing mantid and mvesuvio using conda/mamba

To install `mvesuvio` you need to have `conda` installed (or preferably  `mamba`, a much faster implementation of `conda`).

This is also the recommended best practice way of using the mantid packages.

To download and install mamba:
- https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

To check you have mamba installed, run:
- `mamba --version`

You should see some output with the versions available in your system.

Now create a new conda environment (replace `<environment-name>` with something descriptive, for example `mantid-mvesuvio`):
- `mamba create -n <environment_name>`

And activate the environment you created:
- `mamba activate <environment_name>`

Install the latest version of mantid workbench:
- `mamba install mantidworkbench`

Finally, install `mvesuvio` through the mantid channel:
- `mamba install -c mantid/label/nightly mvesuvio`

You need to run the following command so that `mvesuvio` sets up some defaults:
- `mvesuvio config`

You can now start workbench by typing:
- `workbench`

### Quickstart (running first analysis)

To run your very first analysis (and check that everything is working), go to your home folder and find the folder `.mvesuvio`.
(The `.` in front of the directory name means this folder might be hidden by your OS, so you might have to turn on the option of showing hidden folders.
)

If the folder does not exist, try running `mvesuvio config` from the environment's terminal.
Alternatively, you can open Mantid workbench, go the editor and run:

```
import mvesuvio as mv
mv.set_config()
```

Once you have located the `.mvesuvio` folder, open Mantid workbench and inside it open the script `analysis_inputs.py` located inside `.mvesuvio`.

This script represents the basics for passing in the inputs of the analysis routine.
Click the run button on the workbench to start the execution of the script.
(Check that you have the archive search enabled, the facility is set to ISIS and the instrument set to VESUVIO, otherwise the Vesuvio runs might not be found).
This scipt is an example of a well-behaved sample and it should run without issues.

If the run was successfull, you will notice that a new folder was created inside `.mvesuvio` containing all of the relevant outputs for this script.

**IMPORTANT**:To run a new sample with different inputs, you should **copy** the example script `analysis_inputs.py` and place it in **any** folder of your choice outside `.mvesuvio`. 
For providing the instrument parameters files, place them inside `.mvesuvio/ip_files/`.
(You can change the directory of the instrument files too, consult next section).


## Advanced Usage (CLI)

### Using mvesuvio via the command line (CLI)
If using a conda installation, you can use the Command Line Interface (CLI) to change some configurations and execute your python scripts.
This allows for setting the inputs file or the instrument parameters folder through terminal commands and run the analysis in the terminal (without 
the need for opening mantid workbench).

There are two commands available: `mvesuvio config` and `mvesuvio run`.

#### config

The `config` command has two optional arguments:
- `--set-inputs` - Sets the location of the inputs python file.
- `--set-ipfolder` - Sets the directory in which `mvesuvio` will look for instrument parameter files.

If any of these arguments are not provided, a default location will be selected. 
These will be output on the running of `mvesuvo config`.

Usage examples:
- `mvesuvio config --set-ipfolder C:\IPFolder` - Set instrument parameters folder.
- `mvesuvio config --set-inputs C:\Vesuvio\experiment\inputs.py` - Set inputs file.

#### run

Usage example:
- `mvesuvio run`- Run the vesuvio analysis, will wait for user input when prompted.

### Importing mvesuvio in workbench

If you wish to write a small script using the mvesuvio package and have it run inside workbench, 
`mvesuvio` can be directly imported into the workbench.

In the workbench script editor you must first import mvesuvio:

- `import mvesuvio as mv`

After this you can set the config if desired, as above in the command line example. All arguments are optional.

- `mv.set_config(inputs_file='C:\Vesuvio\experiment\inputs.py', ip_folder='C:\IPFolder')`

Following the setting of the config, you can use workbench to open and edit the analysis input file created in the relevant experiment directory.
Once the inputs have been ammended and the file saved, run the analysis:

- `mv.run(yes_to_all=True)`
