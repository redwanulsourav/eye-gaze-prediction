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
            pass

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
    
    def test_verifyEvalConfig(self):
        """ Verify eval config file has necessary keys """

        config = {
            'run_id': 0,
            'base_path_model': '',
            'base_path_dataset': ''
        }

        config1 = {
            'base_path_model': ''
        }

        config2 = {
            'base_path_model': '',
            'base_path_dataset': '',
        }

        with self.assertRaises(KeyError):
            Utils.Eval.verifyEvalConfig(config1)
        
        with self.assertRaises(KeyError):
            Utils.Eval.verifyEvalConfig(config2)
        
        Utils.Eval.verifyEvalConfig(config)
    
    def test_mergeTrainEvalConfig(self):
        """ Merge Informations From Train and Eval Config """

        """ Test if the values are pulled from each configs properly """
        evalConfig = {
            'base_path_model': '/data/rsourave/saved_model/Coutrot',
            'base_path_dataset': '/data/rsourave/datasets/Coutrot',
            'run_id': 0,
            'videos': [0],
            'viewers': [0]
        }

        trainConfig = {
            'base_path': '/data/rsourave/datasets/Coutrot',
            'batch_size': 6,
            'epochs': 10,
            'length': 2,
            'lr': 0.001,
            'model_type': 0,
            'shuffle': True,
            'stride': 10,
            'videos':  [1, 11, 3],
            'viewers': [0, 3],
            'output_dir': '/data/rsourave/saved_models/Coutrot/'
        }

        mergedConfig = Utils.Eval.mergeTrainEvalConfig(evalConfig, trainConfig)

        """ Merged config should have following keys """
        requiredKeys = ('base_path_model', 'base_path_dataset', 'run_id', 'videos', 'viewers', 'length', 'model_type', 'stride')
        
        for key in requiredKeys:
            self.assertTrue(key in mergedConfig)
        
        self.assertEquals(mergedConfig['base_path_model'], evalConfig['base_path_model'])
        self.assertEquals(mergedConfig['base_path_dataset'], evalConfig['base_path_dataset'])
        self.assertEquals(mergedConfig['run_id'], evalConfig['run_id'])
        self.assertEquals(len(mergedConfig['videos']), 1)
        self.assertEquals(len(mergedConfig['viewers']), 1)
        self.assertEquals(mergedConfig['length'], trainConfig['length'])
        self.assertEquals(mergedConfig['model_type'], trainConfig['model_type'])
        self.assertEquals(mergedConfig['stride'], trainConfig['stride'])
        
        """ Test if video and viewers are picked randomly from trained config properly, if they are missing from eval config """
        evalConfig1 = {
            'base_path_model': '/data/rsourave/saved_model/Coutrot',
            'base_path_dataset': '/data/rsourave/datasets/Coutrot',
            'run_id': 0,
        }

        mergedConfig = Utils.Eval.mergeTrainEvalConfig(evalConfig1, trainConfig)
        self.assertTrue(mergedConfig['videos'][0] in trainConfig['videos'])
        self.assertTrue(mergedConfig['viewers'][0] in trainConfig['viewers'])