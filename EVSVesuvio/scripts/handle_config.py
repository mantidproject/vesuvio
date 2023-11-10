import os

VESUVIO_CONFIG_PATH = os.path.join(os.path.expanduser("~"), '.mvesuvio')
VESUVIO_CONFIG_FILE = "vesuvio.user.properties"


def set_config_vars(var_dict):
    file_path = f'{VESUVIO_CONFIG_PATH}/{VESUVIO_CONFIG_FILE}'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        match = False
        for var in var_dict:
            if line.startswith(var):
                updated_lines.append(f'{var}={var_dict[var]}')
                match = True
                break
        if not match:
            updated_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)


def read_config_var(var):
    file_path = f'{VESUVIO_CONFIG_PATH}/{VESUVIO_CONFIG_FILE}'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    result = ""
    for line in lines:
        if line.startswith(var):
            result = line.split("=", 2)[1].strip('\n')
            break
    if not result:
        raise ValueError(f'{var} was not found in the vesuvio config')
    return result
