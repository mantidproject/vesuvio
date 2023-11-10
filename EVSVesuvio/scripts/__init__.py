"""Package defining top-level application
and entry points.
"""
import os
from shutil import copyfile


def main():
    make_config_dir()


def make_config_dir():
    config_dir = os.path.join(os.path.expanduser("~"), '.mvesuvio')
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)
        copyfile('EVSVesuvio/config/vesuvio.user.properties', f'{config_dir}/vesuvio.user.properties')


if __name__ == '__main__':
    main()
