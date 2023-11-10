"""Package defining top-level application
and entry points.
"""
import argparse
import os
from shutil import copyfile

VESUVIO_CONFIG_FILE = "vesuvio.user.properties"


def main():
    parser = __set_up_parser()
    args = parser.parse_args()
    config_dir = os.path.join(os.path.expanduser("~"), '.mvesuvio')
    cache_dir = config_dir if not args.set_cache else args.set_cache
    experiment = "default" if not args.set_experiment else args.set_experiment

    if __setup_config_dir(config_dir):
        __set_config_vars(config_dir, {'caching.location': cache_dir,
                                       'caching.experiment': experiment})
    __setup_expr_dir(cache_dir, experiment)


def __set_up_parser():
    parser = argparse.ArgumentParser(description="Package to analyse Vesuvio instrument data")
    parser.add_argument("--set-cache", "-c", help="set the cache directory", default="", type=str)
    parser.add_argument("--set-experiment", "-e", help="set the current experiment", default="", type=str)
    return parser


def __setup_config_dir(config_dir):
    success = __mk_dir('config', config_dir)
    if success:
        copyfile('EVSVesuvio/config/vesuvio.user.properties', f'{config_dir}/{VESUVIO_CONFIG_FILE}')
    return success


def __setup_expr_dir(cache_dir, experiment):
    expr_path = os.path.join(cache_dir, "experiments", experiment)
    __mk_dir('experiment', expr_path)


def __mk_dir(type, path):
    success = False
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
            success = True
        except:
            print(f'Unable to make {type} directory at location: {path}')
    return success


def __set_config_vars(config_dir, var_dict):
    file_path = f'{config_dir}/{VESUVIO_CONFIG_FILE}'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        match = False
        for var in var_dict:
            if line.startswith(var):
                updated_lines.append(f'{var}={var_dict[var]}\n')
                match = True
                break
        if not match:
            updated_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)


if __name__ == '__main__':
    print("test")
    main()
