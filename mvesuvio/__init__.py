"""
Vesuvio
=============

Vesuvio is an instrument that performs Neuton Compton Scattering, based at ISIS, RAL, UK. This code processes raw output data to determine
nuclear kinetic energies and moment distributions.
"""

version_info = (0, 0, 0)
__version__ = ".".join(map(str, version_info))
__project_url__ = "https://github.com/mantidproject/vesuvio"

from mvesuvio.scripts import main

class ArgInputs:
    def __init__(self, command):
        self.__command = command

    @property
    def command(self):
        return self.__command

class ConfigArgInputs(ArgInputs):
    def __init__(self, set_cache, set_experiment, set_ipfolder):
        super().__init__("config")
        self.__set_cache = set_cache
        self.__set_experiment = set_experiment
        self.__set_ipfolder = set_ipfolder
    
    @property
    def set_cache(self):
        return self.__set_cache
    
    @property
    def set_experiment(self):
        return self.__set_experiment

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


def set_config(cache_directory="", experiment_id="", ip_folder=""):
    config_args = ConfigArgInputs(cache_directory, experiment_id, ip_folder)
    main(config_args)
    
def run(yes_to_all=False):
    run_args = RunArgInputs(yes_to_all)
    main(run_args)
