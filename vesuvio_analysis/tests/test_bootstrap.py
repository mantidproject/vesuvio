from ..core_functions.bootstrap import runBootstrap
from ..ICHelpers import completeICFromInputs
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import icWSFront, fwdIC, wsBootIC, yfitIC
testPath = Path(__file__).absolute().parent 


