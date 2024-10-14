import os
import json
import yaml

class Database:
    def __init__(self, prefix):
        runs = os.listdir(f'{prefix}')
        self.db_dict = {}
        self.prefix = prefix
    
        for run in runs:
            config_path = f'{prefix}/{run}/run_config.yaml'
            with open(config_path) as f:
                config = yaml.safe_load(f)
            valid = True
            
            """
            if 'dataset' in config:
                if len(config['dataset']['persons']) != 1:
                    valid = False
                    print('Persons is false')
                if 'batch_size' in config['dataset']:
                    if config['dataset']['batch_size'] != 1:
                        valid = False
                        print('batch size is wrong')
            else:
                valid = False
            
            if 'optimizer' in config:
                if config['optimizer']['lr'] != 0.001:
                    valid = False
                    print('Learning rate is wrong')
            else:
                valid = False
                print('Optimizer is wrong')

            """


            if valid == True:
                stride = config['dataset']['stride']
                length = config['dataset']['length']
                model_type = config['model']['type']

                key_str = f's={stride},l={length},t={model_type}'
                if key_str not in self.db_dict:
                    self.db_dict[key_str] = []
                self.db_dict[key_str].append(run)

    def find(self, key):
        if key in self.db_dict:
            return self.db_dict[key]
        else:
            return ''

    def findmin(self, eval_mode, key):
        if eval_mode == '':
            if key in self.db_dict:
                runs = self.db_dict[key]
                runs = [int(x) for x in runs]
                global_min = 1000000000000 
                for r in runs:
                    f = open(f'{self.prefix}/{r}/history.json')
                    contents = f.read()
                    f.close()

                    data_dict = json.loads(contents)
                    min_error = min(data_dict['average_loss'])
                    global_min = min((global_min, min_error))

                return global_min 
            else:
                return None 
        else:
            if key in self.db_dict:
                runs = self.db_dict[key]
                runs = [int(x) for x in runs]
                
                global_min = 1000000000000.0000
                for run in runs:
                    epochs = os.listdir(f'{self.prefix}/{run}/epochs')
                    for i in epochs:
                        f = open(f'{self.prefix}/{run}/epochs/{i}/eval/{eval_mode}/avg_loss.txt','r')
                        contents = float(f.read())
                        f.close()

                        if global_min > contents:
                            global_min = contents

                return global_min 
            else:
                return None 
    
    def findargmin(self, eval_mode, key):
        if eval_mode == '':
            if key in self.db_dict:
                runs = self.db_dict[key]
                runs = [int(x) for x in runs]

                min_value = 100000000
                min_run = -1
                for r in runs:
                    f = open(f'{self.prefix}/{r}/history.json')
                    contents = f.read()
                    f.close()

                    data_dict = json.loads(contents)

                    error = data_dict['average_loss']

                    if error < min_value:
                        min_value = error
                        min_run = r
                    
                return min_run
            else:
                return -1
        else:
            if key in self.db_dict:
                runs = self.db_dict[key]
                for i in range(len(runs)):
                    runs[i] = int(runs[i])
                runs = sorted(runs)
                valid_runs = []
                for run in runs:
                    valid = True
                    for i in range(50):
                        if os.path.exists(f'{self.prefix}/{run}/epochs/{i}/eval/{eval_mode}/avg_loss.txt') == False:
                            valid = False
                        
                    if valid == True:
                        valid_runs.append(run)
                    
                min_value = 1000000000000.0000
                argmin = -1
                arg_best_epoch = -1
                for run in valid_runs:
                    min_error = 100000000000.000
                    best_epoch = -1                    
                    for i in range(50):
                        f = open(f'{self.prefix}/{run}/epochs/{i}/eval/{eval_mode}/avg_loss.txt','r')
                        contents = float(f.read())
                        f.close()
                        if min_error > contents:
                            min_error = contents
                            best_epoch = i
                    if min_value > min_error:
                        min_value = min_error
                        argmin = run
                        arg_best_epoch = best_epoch
                return argmin 
            else:
                return -1
