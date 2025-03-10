import argparse
import json
import yaml

from database_obj import Database

# def generate_cfg_string(stride, length, t):
#     data_str = f'dataset:\n  persons: [0]\n  stride: {stride}\n  length: {length}\n  batch_size: 1\nmodel:\n  type: {t}\noptimizer:\n  lr: 0.001\n'
#     return data_str


def getDatasetRoot(datasetName):
    prefix = f'/data/rsourave/datasets'
    if datasetName.lower() == 'gtea':
        return f'{prefix}/GTEA' # TODO: Use os.path.join
    elif datasetName.lower() == 'coutrot':
        return f'{prefix}/Coutrot' # TODO: use os.path.join



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--stride', required = True, type=int)
    ap.add_argument('-l', '--length', required = True, type=int)
    ap.add_argument('-t', '--type', required = True, type=int)
    ap.add_argument('-d', '--dataset', required = True, type=str)
    ap.add_argument('-v', '--video', required = True, type=int)
    ap.add_argument('-o', '--online', action='store_true')
    ap = ap.parse_args() 
    
    data_dict = ap
    
    data_dict['model_type'] = data_dict['type']
    del data_dict['type']
