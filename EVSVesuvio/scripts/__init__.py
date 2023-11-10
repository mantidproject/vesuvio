"""Package defining entry points.
"""
import argparse
import os
from shutil import copyfile
from EVSVesuvio.scripts import handle_config


def main():
    parser = __set_up_parser()
    args = parser.parse_args()
    config_dir = handle_config.VESUVIO_CONFIG_PATH
    cache_dir = config_dir if not args.set_cache else args.set_cache
    experiment = "default" if not args.set_experiment else args.set_experiment

    if __setup_config_dir(config_dir):
        handle_config.set_config_vars({'caching.location': cache_dir,
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
        copyfile('EVSVesuvio/config/vesuvio.user.properties', f'{config_dir}/{handle_config.VESUVIO_CONFIG_FILE}')
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


if __name__ == '__main__':
    print("test")
    main()
