import unittest

# Import modules to be tested
# import joint.tests.test_analysis as analysis
import joint.tests.test_yspace_fit as yspacefit


# Initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# Add tests to the test suite
# suite.addTests(loader.loadTestsFromModule(analysis))
suite.addTests(loader.loadTestsFromModule(yspacefit))

# Initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)