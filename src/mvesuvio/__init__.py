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
    def __init__(self, analysis_inputs, ip_folder):
        super().__init__("config")
        self.__set_inputs = analysis_inputs
        self.__set_ipfolder = ip_folder

    @property
    def analysis_inputs(self):
        return self.__set_inputs

    @property
    def ip_folder(self):
        return self.__set_ipfolder


class RunArgInputs(ArgInputs):
    def __init__(self):
        super().__init__("run")


def config(analysis_inputs="", ip_folder=""):
    config_args = ConfigArgInputs(analysis_inputs, ip_folder)
    main(config_args)


def run():
    run_args = RunArgInputs()
    main(run_args)


def version():
    """Returns the version of the mvesuvio package."""
    return __version__
