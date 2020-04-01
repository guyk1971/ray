import argparse
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env
from my_scripts.sandbox.my_custom_envs import L2PEnv


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_yml', type=str, help='the input yaml file')
    args = parser.parse_args()
    return args


def on_episode_start(info):
    episode = info["episode"]
    print("on_episode_start {0} with the following info: ".format(episode.episode_id))
    print(info.keys())

def on_episode_step(info):
    episode = info["episode"]
    pole_angle = abs(episode.last_observation_for()[2])
    raw_angle = abs(episode.last_raw_obs_for()[2])
    assert pole_angle == raw_angle
    episode.user_data["pole_angles"].append(pole_angle)


def on_episode_end(info):
    episode = info["episode"]
    # pole_angle = np.mean(episode.user_data["pole_angles"])
    # print("GK episode {} ended with length {} and pole angles {}".format(
    #     episode.episode_id, episode.length, pole_angle))
    # episode.custom_metrics["pole_angle"] = pole_angle
    # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
    print("on_episode_end {0} with the following info: ".format(episode.episode_id))
    print(info.keys())


def on_sample_end(info):
    print("returned sample batch of size {}".format(info["samples"].count))


def on_train_result(info):
    print("on_train_results with info:")
    print(info.keys())
    # you can mutate the result dict to add new fields to return
    info["result"]["callback_ok"] = True


def on_postprocess_traj(info):
    episode = info["episode"]
    batch = info["post_batch"]
    print("postprocessed {} steps".format(batch.count))
    if "num_batches" not in episode.custom_metrics:
        episode.custom_metrics["num_batches"] = 0
    episode.custom_metrics["num_batches"] += 1



def run_trial(args):
    print('loading trial config from ' + args.input_yml)
    with open(args.input_yml,'r') as yml_file:
        yml_config = yaml.load(yml_file,Loader=yaml.FullLoader)
    run_config = list(yml_config.values())[0]
    run_config["config"].update({"env": run_config['env']})
    ray.init(local_mode=True)
    trials = tune.run(
        run_config["run"],
        stop=run_config["stop"],
        config=run_config["config"],
        return_trials=True)
    print('done')
    return


if __name__ == "__main__":
    args = parse_cmd_line()
    register_env("L2P-v0", lambda config: L2PEnv(config))
    run_trial(args)


