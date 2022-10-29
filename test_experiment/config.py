#!./env python

import argparse
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import shutil
import yaml
import json
import torch

from src.utils import check_path, check_path_remote
from fractions import Fraction

def check_num(num):
    if type(num) in [float, int]:
        return num

    if isinstance(num, str):
        return float(Fraction(num))

    raise TypeError(num)


def read_config(config_file='config.yaml', remote=False, server=None, remote_dir=''):

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -- hyperparas massage --
    for key in ['lr', 'wd', 'momentum', 'gamma',  'label_smoothing', 'loss_flooding']:
        if key in config and config[key] is not None:
            config[key] = check_num(config[key])

    if config['state_path']:
        config['state_path'] = os.path.join(os.getcwd(), config['checkpoint_dir'], config['state_path'])

    # -- set checkpoint name --
    config['checkpoint'] = '%s' % config['opt']
    config['checkpoint'] += '_%s' % config['model']
    config['checkpoint'] = config['dataset'] + '_' + config['checkpoint']
    config['checkpoint'] += '_epoch=%i' % config['epochs']
    if config['batch_size'] != 16:
        config['checkpoint'] += '_bs=%i' % config['batch_size']
    if config['wd'] is not None and config['wd'] > 0:
        config['checkpoint'] += '_wd=%g' % config['wd']
    if config['momentum'] is not None and config['momentum'] > 0:
        config['checkpoint'] += '_mom=%g' % config['momentum']
    if 'gradient_clipping' in config and config['gradient_clipping']:
        config['checkpoint'] += '_gc'
    if config['suffix']:
        config['checkpoint'] += '_%s' % config['suffix']
    del config['suffix']
    if config['test']:
        config['checkpoint'] = 'test_' + config['checkpoint']
    del config['test']

    path = os.path.join(config['checkpoint_dir'], config['checkpoint'])
    if remote:
        path = check_path_remote(path, config, server=server, remote_dir=remote_dir)
    else:
        path = check_path(path, config)
    _, checkpoint = os.path.split(path)
    config['checkpoint'] = checkpoint

    if config['resume']:
        config['resume_checkpoint'] = 'checkpoint.pth.tar'
        assert(os.path.isfile(os.path.join(path, config['resume_checkpoint']))), 'checkpoint %s not exists!' % config['resume_checkpoint']

    print("\n--------------------------- %s ----------------------------------" % config_file)
    for k, v in config.items():
        print('%s:'%k, v, type(v))
    print("---------------------------------------------------------------------\n")

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sequence-classification')
    parser.add_argument('--config', '-c', default='config.yaml', type=str, metavar='C', help='config file')
    parser.add_argument('--remote', action='store_true', help='run on remote server?')
    parser.add_argument('--server', '-s', default=None, type=str, help='remote server name')
    parser.add_argument('--remote_dir', default='/data', type=str, help='remote working dir')
    args = parser.parse_args()

    config = read_config(args.config, args.remote, args.server, args.remote_dir)
    if not args.remote:
        with open('%s/%s/para.json' % (config['checkpoint_dir'], config['checkpoint']), 'w') as f:
            json.dump(config, f)
    else:
        print(os.getcwd())
        with open('tmp/para.json', 'w') as f:
            json.dump(config, f)

    # reveal the path to bash
    with open('tmp/path.tmp', 'w') as f:
        f.write(config['checkpoint'])


