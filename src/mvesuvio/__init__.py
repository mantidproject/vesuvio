"""
Vesuvio
=============

Vesuvio is an instrument that performs Neuton Compton Scattering, based at ISIS, RAL, UK. This code processes raw output data to determine
nuclear kinetic energies and moment distributions.
"""

from mvesuvio._version import __version__
from mvesuvio.main import main

__project_url__ = "https://github.com/mantidproject/vesuvio"

__all__ = ["__version__", "__project_url__"]


class ArgInputs:
    def __init__(self, command):
        self.__command = command

    @property
    def command(self):
        return self.__command


class ConfigArgInputs(ArgInputs):
    def __init__(self, set_inputs, set_ipfolder):
        super().__init__("config")
        self.__set_inputs = set_inputs
        self.__set_ipfolder = set_ipfolder

    @property
    def set_inputs(self):
        return self.__set_inputs

    @property
    def set_ipfolder(self):
        return self.__set_ipfolder


class RunArgInputs(ArgInputs):
    def __init__(self, back_workspace, front_workspace, minimal_output, outputs_dir):
        super().__init__("run")
        self.__back_workspace = back_workspace
        self.__front_workspace = front_workspace
        self.__minimal_output = minimal_output
        self.__outputs_dir = outputs_dir

    @property
    def back_workspace(self):
        return self.__back_workspace

    @property
    def front_workspace(self):
        return self.__front_workspace

    @property
    def minimal_output(self):
        return self.__minimal_output

    @property
    def outputs_dir(self):
        return self.__outputs_dir


def set_config(inputs_file="", ip_folder=""):
    config_args = ConfigArgInputs(inputs_file, ip_folder)
    main(config_args)


def run(back_workspace="", front_workspace="", minimal_output=False, outputs_dir=""):
    run_args = RunArgInputs(back_workspace, front_workspace, minimal_output, outputs_dir)
    main(run_args)
