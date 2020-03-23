

import argparse
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
import my_custom_envs



def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_yml', type=str, help='the input yaml file')
    args = parser.parse_args()
    return args


def run_trial(args):
    print('loading trial config from ' + args.input_yml)
    with open(args.input_yml,'r') as yml_file:
        yml_config = yaml.load(yml_file,Loader=yaml.FullLoader)
    run_config = list(yml_config.values())[0]
    run_config["config"].update({"env": run_config['env']})
    ray.init()
    trials = tune.run(
        run_config["run"],
        stop=run_config["stop"],
        config=run_config["config"],
        return_trials=True)
    print('done')
    return


if __name__ == "__main__":
    args = parse_cmd_line()
    register_env("L2P-v0", lambda config: my_custom_envs.L2PEnv(config))
    run_trial(args)


