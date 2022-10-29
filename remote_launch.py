import os
import sys

import argparse
import fabric
import yaml
import re

import configure
import dynamic_yaml

def create_parser():
    parser = argparse.ArgumentParser("Remote Exp")
    parser.add_argument("--path_to_project_configuration", required=True)
    parser.add_argument("--path_to_local_experiment", required=True)

def run_experiment(c: fabric.Connection, cfg):

    results = c.run("; ".join(cfg.command), hide=True)
    print(results)


if __name__ == '__main__':
    with open("test_experiment/test_exp.yaml") as fileobj:
        cfg = dynamic_yaml.load(fileobj)

    print(vars(cfg))

    run_experiment(fabric.Connection(cfg.server_name), cfg)

# server = "ds-serv1"
# c = fabric.Connection(server)
# print(c.run('which python; ls ~/; source ~/anaconda3/bin/activate common; which python', hide=True))
#
# print(c.run('which python', hide=True))
#
#





