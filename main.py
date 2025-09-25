from argparse import ArgumentParser
from src.neoHookean import NeoHookean

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config.py")

args = parser.parse_args()
config_path = args.config

problem = NeoHookean(config_path=config_path)
# problem.optimize()
problem.forward()
problem.save()
