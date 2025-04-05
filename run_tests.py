import unittest
import os

os.environ['GP3_ROOT'] = os.path.abspath(os.path.dirname(__file__))

loader = unittest.TestLoader()
suite = loader.discover('tests')

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
