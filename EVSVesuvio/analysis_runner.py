import time
from pathlib import Path
import importlib
import sys
from os import path
#from EVSVesuvio.vesuvio_analysis.core_functions.bootstrap import runBootstrap
#from EVSVesuvio.vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from EVSVesuvio.vesuvio_analysis.core_functions.run_script import runScript
from mantid.api import AnalysisDataService
from EVSVesuvio.scripts import handle_config


def run():
    scriptName =  handle_config.read_config_var('caching.experiment')
    experimentsPath = Path(handle_config.read_config_var('caching.location')) / "experiments" / scriptName # Path to the repository
    inputs_path = experimentsPath / "analysis_inputs.py"
    ai = import_from_path(inputs_path, "analysis_inputs")

    ipFilesPath = Path(path.dirname(path.dirname(handle_config.__file__))) / "vesuvio_analysis" / "ip_files"

    start_time = time.time()

    wsBackIC = ai.LoadVesuvioBackParameters(ipFilesPath)
    wsFrontIC = ai.LoadVesuvioFrontParameters(ipFilesPath)
    bckwdIC = ai.BackwardInitialConditions(ipFilesPath)
    fwdIC = ai.ForwardInitialConditions
    yFitIC = ai.YSpaceFitInitialConditions
    bootIC = ai.BootstrapInitialConditions
    userCtr = ai.UserScriptControls

    runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

    end_time = time.time()
    print("\nRunning time: ", end_time-start_time, " seconds")


def import_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    run()
