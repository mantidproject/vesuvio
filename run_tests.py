import unittest

# Import modules to be tested

# Initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# # Import modules to be tested
# import vesuvio_analysis.tests.test_analysis as analysis
# # Add tests to the test suite
# suite.addTests(loader.loadTestsFromModule(analysis))

# import vesuvio_analysis.tests.test_yspace_fit as yspacefit
# suite.addTests(loader.loadTestsFromModule(yspacefit))

import vesuvio_analysis.tests.test_bootstrap as bootstrap
suite.addTests(loader.loadTestsFromModule(bootstrap))

# Initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)