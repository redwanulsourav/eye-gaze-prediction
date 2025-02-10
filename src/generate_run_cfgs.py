import argparse
import json
import yaml

from database_obj import Database

def generate_cfg_string(stride, length, t):
    data_str = f'dataset:\n  persons: [0]\n  stride: {stride}\n  length: {length}\n  batch_size: 1\nmodel:\n  type: {t}\noptimizer:\n  lr: 0.001\n'
    return data_str

def generate_online_cfg_string(stride, length, t, db):
    key = f's={stride},l={length},t={t}'
    run_id = db.findargmin('', key)
    path = f'/data/rsourave/projects/runs_new/runs/{run_id}/epochs/49/weights.pt'
    data_str = f'dataset:\n  persons: [0]\n  stride: {stride}\n  length: {length}\n  batch_size: 1\nmodel:\n  type: {t}\noptimizer:\n  lr: 0.00003\nweights:\n  weights_path:  {path}\n'
    return data_str



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--stride', required = True, type=int)
    ap.add_argument('-l', '--length', required = True, type=int)
    ap.add_argument('-t', '--type', required = True, type=int)
    ap.add_argument('-o', '--online', action='store_true')
    ap = ap.parse_args() 
    
    db = Database('/data/rsourave/projects/runs_new/runs')
    

    # weight_path = f'/data/rsourave/projects/runs/{run_id}/epochs/49/weights.pt'
    if ap.online == False:
        f = open(f'run_cfgs/s={ap.stride},l={ap.length},t={ap.type}.yaml', 'w')
        f.write(generate_cfg_string(ap.stride, ap.length, ap.type))
        f.close()
    else:
        print('dhhukse')
        f = open(f'run_cfgs/s={ap.stride},l={ap.length},t={ap.type}_o.yaml', 'w')
        f.write(generate_online_cfg_string(ap.stride, ap.length, ap.type, db))
        f.close()

