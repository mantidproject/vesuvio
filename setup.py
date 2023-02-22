"""
Vesuvio
=============

#ABOUT HERE
"""
from __future__ import print_function

import fnmatch
import os

from setuptools import find_packages, setup

from vesuvio import __project_url__, __version__

# ==============================================================================
# Constants
# ==============================================================================
NAME = 'vesuvio'
THIS_DIR = os.path.dirname(__file__)
_package_dirs = ['ip_files', 'experiments']
_package_exts = ['.OPJ', '.par', '.DAT', '.DAT3', '.DAT6', '.txt', '.NXS', '.nxs', '.np']


# ==============================================================================
# Package requirements helper
# ==============================================================================

def read_requirements_from_file(filepath):
    """Read a list of requirements from the given file and split into a
    list of strings. It is assumed that the file is a flat
    list with one requirement per line.
    :param filepath: Path to the file to read
    :return: A list of strings containing the requirements
    """
    with open(filepath, 'rU') as req_file:
        return req_file.readlines()


def get_package_data():
    """Return data_files in a platform dependent manner"""
    package_data = []
    package_dir = os.path.join(THIS_DIR, NAME)
    for root, dirnames, filenames in os.walk(package_dir):
        if [dir for dir in _package_dirs if dir in root]:
            for ext in _package_exts:
                for filename in fnmatch.filter(filenames, f'*{ext}'):
                    package_data.append(os.path.relpath(os.path.join(root, filename), start=package_dir))
    return {NAME: package_data}


# ==============================================================================
# Setup arguments
# ==============================================================================
setup_args = dict(name=NAME,
                  version=__version__,
                  description='determine nuclear kinetic energies and moment distributions from Neutron Compton Scattering data',
                  author='The Mantid Project',
                  author_email='mantid-help@mantidproject.org',
                  url=__project_url__,
                  packages=find_packages(),
                  package_data=get_package_data(),
                  # Install this as a directory
                  zip_safe=False,
                  classifiers=['Operating System :: MacOS',
                               'Operating System :: Microsoft :: Windows',
                               'Operating System :: POSIX :: Linux',
                               'Programming Language :: Python :: 3.8',
                               'Development Status :: 4 - Beta',
                               'Topic :: Scientific/Engineering'])

# ==============================================================================
# Setuptools deps
# ==============================================================================
# Running setup command requires the following dependencies
setup_args['setup_requires'] = read_requirements_from_file(os.path.join(THIS_DIR, 'setup-requirements.txt'))

# User installation requires the following dependencies
install_requires = setup_args['install_requires'] = \
    read_requirements_from_file(os.path.join(THIS_DIR, 'install-requirements.txt'))
# Testing requires
setup_args['tests_require'] = read_requirements_from_file(os.path.join(THIS_DIR, 'test-requirements.txt')) \
    + install_requires

# ==============================================================================
# Scripts
# ==============================================================================
# Scripts to be callable from command line
setup_args['entry_points'] = {'console_scripts':
                              ['run_test=scripts.test:print_test', ],
                              }

# ==============================================================================
# Main setup
# ==============================================================================
setup(**setup_args)
