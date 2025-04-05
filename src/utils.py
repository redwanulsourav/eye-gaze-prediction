import os

class Utils:
    class Eval:
        def createEvalDirs(savedModelsRoot: str, runId: str) -> str:
            runPath = os.path.join(savedModelsRoot, runId)

            if os.path.isdir(runPath) == False:
                raise FileNotFoundError
            
            evalPath = os.path.join(savedModelsRoot, runId, 'evals', 'eval0')
            picklesPath = os.path.join(evalPath, 'pickles')
            pngsPath = os.path.join(evalPath, 'pngs')

            os.makedirs(evalPath, exist_ok = True)
            os.makedirs(picklesPath)
            os.makedirs(pngsPath)

            return evalPath
            
