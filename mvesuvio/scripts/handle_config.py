import os
from shutil import copyfile, copytree, ignore_patterns


def __parse_config_env_var():
    env_var = os.getenv("VESUVIOPROPERTIES")
    if env_var:
        config_path, config_file = os.path.split(env_var)
    else:
        config_path = os.path.join(os.path.expanduser("~"), ".mvesuvio")
        config_file = "vesuvio.user.properties"
    return config_path, config_file


### PATH CONSTANTS ###
VESUVIO_CONFIG_PATH, VESUVIO_CONFIG_FILE = __parse_config_env_var()
VESUVIO_INPUTS_FILE = "analysis_inputs.py"
VESUVIO_INPUTS_PATH = os.path.join(VESUVIO_CONFIG_PATH, VESUVIO_INPUTS_FILE)
VESUVIO_PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANTID_CONFIG_FILE = "Mantid.user.properties"
VESUVIO_IPFOLDER_PATH = os.path.join(VESUVIO_CONFIG_PATH, "ip_files")
######################


def __read_config(config_file_path, throw_on_not_found=True):
    lines = ""
    try:
        with open(config_file_path, "r") as file:
            lines = file.readlines()
    except IOError:
        if throw_on_not_found:
            raise RuntimeError(
                f"Could not read from vesuvio config file: {config_file_path}"
            )
    return lines


def set_config_vars(var_dict):
    file_path = os.path.join(VESUVIO_CONFIG_PATH, VESUVIO_CONFIG_FILE)
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
    file_path = f"{VESUVIO_CONFIG_PATH}{os.path.sep}{VESUVIO_CONFIG_FILE}"
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


def setup_config_dir(config_dir):
    success = __mk_dir("config", config_dir)
    if success:
        copyfile(
            os.path.join(VESUVIO_PACKAGE_PATH, "config", VESUVIO_CONFIG_FILE),
            os.path.join(config_dir, VESUVIO_CONFIG_FILE),
        )
        copyfile(
            os.path.join(VESUVIO_PACKAGE_PATH, "config", MANTID_CONFIG_FILE),
            os.path.join(config_dir, MANTID_CONFIG_FILE),
        )


def setup_default_inputs():
    if not os.path.isfile(VESUVIO_INPUTS_PATH):
        copyfile(
            os.path.join(VESUVIO_PACKAGE_PATH, "config", VESUVIO_INPUTS_FILE),
            os.path.join(VESUVIO_INPUTS_PATH),
        )


def setup_default_ipfile_dir():
    if not os.path.isdir(VESUVIO_IPFOLDER_PATH):
        copytree(
            os.path.join(VESUVIO_PACKAGE_PATH, "config", "ip_files"),
            VESUVIO_IPFOLDER_PATH,
            ignore=ignore_patterns("__*"),
        )


def __mk_dir(type, path):
    success = False
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
            success = True
        except:
            print(f"Unable to make {type} directory at location: {path}")
    return success


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
