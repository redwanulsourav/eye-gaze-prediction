import unittest
import os
import shutil

from utils import Utils

class test_UnitEval(unittest.TestCase):
    """ Create necessary directory for eval """

    """ Necessary directories:
        <saved_model_root>/<run_id>/evals/eval0/<eval_id>/
                                                    |
                                                    |---------- pickles
                                                    |---------- pngs
                                                    |---------- config.yml
                                                    |---------- history.yml
    """
    
    def setUp(self):
        self.savedModelRoot = os.path.join(os.environ['GP3_ROOT'], 'tests', 'res', 'dummySavedModelRoot')
        self.runId = '0'
    
    def tearDown(self): 
        try:
            shutil.rmtree(self.savedModelRoot)
        except Exception as e:
            print(f'Exception: {e}')

    def test_createEvalDirs(self): 
        """ Should create necessary directories for eval """

        """ The method should raise error if there is no directory for runId """
        with self.assertRaises(FileNotFoundError):
            Utils.Eval.createEvalDirs(self.savedModelRoot, self.runId)
        
        runPath = os.path.join(self.savedModelRoot, self.runId)
        os.makedirs(runPath)
        evalPath = Utils.Eval.createEvalDirs(self.savedModelRoot, self.runId)

        """ Expected paths """
        picklePath = os.path.join(evalPath, 'pickles')
        pngPath = os.path.join(evalPath, 'pngs')

        """ The method should create picklePath """
        self.assertTrue(os.path.isdir(picklePath))

        """ The method should create pngPath """
        self.assertTrue(os.path.isdir(pngPath))
    