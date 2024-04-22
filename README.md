# VESUVIO REPOSITORY

This repository contains:
- `mvesuvio` package containing the Optimized NCP analysis procedures, published nightly.
- Vesuvio calibration script

## Install mvesuvio package

### For stand-alone (non-conda) Mantid installation
Currently this method is tested up until Mantid 6.8. 

The easiest way to install the package is to install it through pip.
To pip install mvesuvio, follow these [instructions](https://docs.mantidproject.org/nightly/concepts/PipInstall.html) and replace `<insert_package_name>` with `mvesuvio`.

Then open mantid workbench and in the editor run the following script:

```
import mvesuvio as mv
mv.set_config()
```

You will see that the output shows two directories: one for the inputs file and another for the instrument parameters (IP) file.

Both of these directories sit inside a default location of a `.mvesuvio` folder.

The `.` in front of the directory name means this folder might be hidden by your OS, so you might have to turn on the option of showing hidden folders.

You should place of your IP files inside this IP directory, you will notice that I have placed some example files in there already.

The other directory in the output of the config command tells you which inputs file is being used for running the analysis.

You should go to the directory of the inputs file and copy-and-paste this file into any directory inside your desktop where you want to run the analysis for a given sample.

And that's the end of the installation and setup!

## Using mvesuvio

The easiest way to run the analysis scripts is to start with an inputs file and run it from inside mantid workbench.

Open mantid workbench and navigate to the directory of the inputs file you copy-and-pasted in the last section.

**Warning**: Do not run the analysis routine on the default location for the inputs file. This will clutter the default `.mvesuvio` folder and may cause problems later down the line.

Now you can alter the inputs of the file as you see fit. When you are finished, press run. This will run the routine and create a new directory for the output files inside the parent directory of the inputs file.

To run a new sample, simply copy-and-paste one of the inputs file into any directory you see fit, change the inputs and press run.

## Alternative Installation and Usage (Conda and CLI)
We also provide an alternative conda installation for users wanting to use mantid workbench inside a conda environment.

### Install mamba

To use the `mvesuvio` package you will need to use the `conda` package manager (or preferably  `mamba`, a much faster implementation of `conda`).

This is also the recommended best practice way of using the mantid packages.

To download and install mamba:
- https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

### Create and activate an environment for mantidworkbench and mvesuvio

Firstly create a conda environment where `mvesuvio` is going to be installed:
- `mamba create -n <environment_name>`

Next activate the environment you created:
- `mamba activate <environment_name>`

Install mantid workbench (the version is pinned to 6.8 for now):
- `mamba install mantidworkbench=6.8`

Finally, install `mvesuvio` through the mantid channel:
- `mamba install -c mantid/label/nightly mvesuvio`

Start workbench using the command line:
- `workbench`

### Using mvesuvio via the command line (CLI)
If using a conda installation, the Command Line Interface (CLI) of the mvesuvio package becomes an attractive feature. 

This allows for setting the inputs file or the IP folder through terminal commands and run the analysis in the terminal (without 
the need for opening mantid workbench).

You can use `mvesuvio` via the command line. There are two commands available: `config` and `run`.

#### config

The `config` command has two optional arguments:
- `--set-inputs` - Sets the location of the inputs python file.
- `--set-ipfolder` - Sets the directory in which `mvesuvio` will look for instrument parameter files.

If any of these arguments are not provided a default location will be selected. These will be output on the running of `mvesuvo config`

Usage examples:
- `mvesuvio config --set-ipfolder C:\IPFolder` - Set instrument parameters folder.
- `mvesuvio config --set-inputs C:\Vesuvio\experiment\inputs.py` - Set inputs file.

#### run

The `run` command has one optional argument:
- `--yes` - If provided, this argument automatically inputs `Y` when prompted for user input.

Usage example:
- `mvesuvio run --yes` - Run the vesuvio analysis, automatically providing `Y` when prompted.
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
