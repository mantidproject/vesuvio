import os
from shutil import copyfile, copytree, ignore_patterns
from pathlib import Path


### PATH CONSTANTS ###
VESUVIO_PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VESUVIO_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".mvesuvio")
VESUVIO_PROPERTIES_FILE = "vesuvio.user.properties"
ANALYSIS_INPUTS_FILE = "analysis_inputs.py"
MANTID_CONFIG_FILE = "Mantid.user.properties"
PLOTS_CONFIG_FILE = "vesuvio.plots.mplstyle"
SCRIPT_FIGUES_FILE = "script_to_create_figures.py"
IP_FOLDER = "ip_files"
######################


def set_default_config_vars():
    set_config_vars(
        {
            "caching.inputs": str(Path(VESUVIO_CONFIG_PATH, ANALYSIS_INPUTS_FILE)),
            "caching.ipfolder": str(Path(VESUVIO_CONFIG_PATH, IP_FOLDER)),
        }
    )


def __read_config(config_file_path, throw_on_not_found=True):
    lines = ""
    try:
        with open(config_file_path, "r") as file:
            lines = file.readlines()
    except IOError:
        if throw_on_not_found:
            raise RuntimeError(f"Could not read from vesuvio config file: {config_file_path}")
    return lines


def set_config_vars(var_dict):
    file_path = Path(VESUVIO_PACKAGE_PATH, "config", VESUVIO_PROPERTIES_FILE)
    lines = __read_config(file_path)

    updated_lines = []
    for line in lines:
        match = False
        for var in var_dict:
            if line.startswith(var):
                new_line = f"{var}={var_dict[var]}"
                updated_lines.append(f"{new_line}\n")
                match = True
                print(f"Setting: {new_line}")
                break

        if not match:
            updated_lines.append(line)

    with open(file_path, "w") as file:
        file.writelines(updated_lines)


def read_config_var(var, throw_on_not_found=True):
    file_path = Path(VESUVIO_PACKAGE_PATH, "config", VESUVIO_PROPERTIES_FILE)
    lines = __read_config(file_path, throw_on_not_found)

    result = ""
    for line in lines:
        if line.startswith(var):
            result = line.split("=", 2)[1].strip("\n")
            break
    if not result and throw_on_not_found:
        raise ValueError(f"{var} was not found in the vesuvio config")
    return result


def get_script_name():
    filename = os.path.basename(read_config_var("caching.inputs"))
    scriptName = filename.removesuffix(".py")
    return scriptName


def get_plots_config_file() -> str:
    return os.path.abspath(os.path.join(VESUVIO_CONFIG_PATH, PLOTS_CONFIG_FILE))


def refresh_config_dir_and_contents():
    setup_config_dir()
    setup_plots_config_file()
    setup_script_figures_file()
    setup_default_inputs()
    setup_default_ipfile_dir()


def setup_script_figures_file():
    if not Path(VESUVIO_CONFIG_PATH, SCRIPT_FIGUES_FILE).exists():
        copyfile(
            Path(VESUVIO_PACKAGE_PATH, "config", SCRIPT_FIGUES_FILE),
            Path(VESUVIO_CONFIG_PATH, SCRIPT_FIGUES_FILE),
        )


def setup_plots_config_file():
    if not Path(VESUVIO_CONFIG_PATH, PLOTS_CONFIG_FILE).exists():
        copyfile(
            Path(VESUVIO_PACKAGE_PATH, "config", PLOTS_CONFIG_FILE),
            Path(VESUVIO_CONFIG_PATH, PLOTS_CONFIG_FILE),
        )


def setup_config_dir():
    if not os.path.isdir(VESUVIO_CONFIG_PATH):
        os.makedirs(VESUVIO_CONFIG_PATH)


def setup_default_inputs():
    copyfile(
        Path(VESUVIO_PACKAGE_PATH, "config", ANALYSIS_INPUTS_FILE),
        Path(VESUVIO_CONFIG_PATH, ANALYSIS_INPUTS_FILE),
    )


def setup_default_ipfile_dir():
    if not Path(VESUVIO_CONFIG_PATH, IP_FOLDER).is_dir():
        copytree(
            Path(VESUVIO_PACKAGE_PATH, "config", "ip_files"),
            Path(VESUVIO_CONFIG_PATH, IP_FOLDER),
            ignore=ignore_patterns("__*"),
        )


def config_set():
    if read_config_var("caching.inputs", False):
        return True
    else:
        return False


def check_dir_exists(type, path):
    if not os.path.isdir(path):
        print(f"Directory of {type} could not be found at location: {path}")
        return False
    return True
