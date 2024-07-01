"""
Vesuvio
=============

Vesuvio is an instrument that performs Neuton Compton Scattering, based at ISIS, RAL, UK. This code processes raw output data to determine
nuclear kinetic energies and moment distributions.
"""

from mvesuvio._version import __version__
__project_url__ = "https://github.com/mantidproject/vesuvio"

from mvesuvio.scripts import main

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
    def __init__(self, yes):
        super().__init__("run")
        self.__yes = yes

    @property
    def yes(self):
        return self.__yes


def set_config(inputs_file="", ip_folder=""):
    config_args = ConfigArgInputs(inputs_file, ip_folder)
    main(config_args)
    
def run(yes_to_all=False):
    run_args = RunArgInputs(yes_to_all)
    main(run_args)
