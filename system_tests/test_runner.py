"""
The tests below only pass in Mantid version 6.2,
other versions give fail.
"""

import system_tests.test_analysis as analysis
import system_tests.test_bootstrap as bootstrap
import system_tests.test_jackknife as jackknife
import system_tests.test_yspace_fit as yspacefit
import system_tests.test_yspace_fit_GC as yspacefitGC
import unittest

# Initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Import modules to be tested
# Add tests to the test suite
suite.addTests(loader.loadTestsFromModule(analysis))
suite.addTests(loader.loadTestsFromModule(bootstrap))
suite.addTests(loader.loadTestsFromModule(jackknife))
suite.addTests(loader.loadTestsFromModule(yspacefit))
suite.addTests(loader.loadTestsFromModule(yspacefitGC))

# Initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)
