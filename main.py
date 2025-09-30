from argparse import ArgumentParser
from src.neoHookean import NeoHookean
from src.neoHookean_topo_opt import NeoHookeanTopoOpt

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config.py")

args = parser.parse_args()
config_path = args.config

problem = NeoHookeanTopoOpt(config_path=config_path)
# problem.forward()
# problem.save()
problem.optimize()
