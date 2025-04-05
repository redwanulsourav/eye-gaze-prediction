import unittest
from ..src import generate_run_cfgs

class TestGenerateRunCFGs(unittest.TestCase):
    def TestGenerateCFGString(self):
        result = generate_run_cfgs.generate_cfg_string(stride=1, length=1, t=0)
        expected = f'dataset:\n  persons: [0]\n  stride: {1}\n  length: {1}\n  batch_size: 1\nmodel:\n  type: 0\noptimizer:\n  lr: 0.001\n'
        assert result == expected, f'CFGs do not match\nExpected:\n{expected}\nReturned:\n{result}\n'

obj = TestGenerateRunCFGs()
obj.TestGenerateCFGString()
print('Generate run CFG tests passed')

