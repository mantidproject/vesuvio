import time
from pathlib import Path
import importlib
import sys
from mvesuvio.vesuvio_analysis.run_script import runScript
from mvesuvio.scripts import handle_config


def run(yes_to_all=False):
    scriptName = handle_config.read_config_var("caching.experiment")
    experimentsPath = (
        Path(handle_config.read_config_var("caching.location"))
        / "experiments"
        / scriptName
    )  # Path to the repository
    inputs_path = experimentsPath / "analysis_inputs.py"
    ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))
    ai = import_from_path(inputs_path, "analysis_inputs")

    start_time = time.time()

    wsBackIC = ai.LoadVesuvioBackParameters(ipFilesPath)
    wsFrontIC = ai.LoadVesuvioFrontParameters(ipFilesPath)
    bckwdIC = ai.BackwardInitialConditions(ipFilesPath)
    fwdIC = ai.ForwardInitialConditions
    yFitIC = ai.YSpaceFitInitialConditions
    bootIC = ai.BootstrapInitialConditions
    userCtr = ai.UserScriptControls

    runScript(
        userCtr,
        scriptName,
        wsBackIC,
        wsFrontIC,
        bckwdIC,
        fwdIC,
        yFitIC,
        bootIC,
        yes_to_all,
    )

    end_time = time.time()
    print("\nRunning time: ", end_time - start_time, " seconds")


def import_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    run()
