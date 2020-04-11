import argparse
import numpy as np
import yaml
import ray
from ray import tune
from ray.tune.registry import register_env

############################
# set the python path properly
import os
import sys
path_to_curr_file=os.path.realpath(__file__)
path_to_dir=os.path.split(path_to_curr_file)[0]
# print('path curr dir: '+path_to_dir )
path_to_par_dir=os.path.dirname(path_to_dir)
# print('path par dir: '+path_to_par_dir )
proj_root=os.path.split(path_to_par_dir)[0]
# print('proj_root: '+proj_root)
if proj_root not in sys.path:
    sys.path.append(proj_root)
# print(sys.path)
############################

from my_scripts.sandbox.my_custom_envs import L2PEnv


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_yml', type=str, help='the input yaml file')
    parser.add_argument('-cf', '--ckpt_freq', type=int, default=0, help='save checkpoint every cf training iterations')
    args = parser.parse_args()
    return args

#########################################
# callbacks
def on_episode_start(info):
    episode = info["episode"]
    print("+++++++++++ on_episode_start {0} with the following info: ++++++++++++".format(episode.episode_id))
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
    # print("GK episode {} ended with length {} and pole angles {}".format(episode.episode_id, episode.length))
    # episode.custom_metrics["pole_angle"] = pole_angle
    # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]
    print("------------ on_episode_end {0} with the following info: -------------".format(episode.episode_id))
    print(info.keys())


def on_sample_end(info):
    print("returned sample batch of size {}".format(info["samples"].count))


def on_train_result(info):
    print("================on_train_results with info:=========================")
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

supported_callbacks={'on_episode_start': on_episode_start,
                    'on_episode_step': on_episode_step,
                    'on_episode_end': on_episode_end,
                    'on_sample_end': on_sample_end,
                    'on_train_result': on_train_result,
                    'on_postprocess_traj': on_postprocess_traj }


##########################################
# parse flow config
def parse_flow_config(run_config):
    # parse callbacks
    flow_config = run_config.get('flow_config',None)
    if flow_config is None:
        return

    req_callback_dict = flow_config.get('callbacks',None)
    if req_callback_dict:
        sup_callback_dict = dict()
        for cb_name in req_callback_dict.keys():
            if req_callback_dict[cb_name]:  # can be True or False
                if cb_name in supported_callbacks.keys():
                    sup_callback_dict.update({cb_name:supported_callbacks[cb_name]})
                else:
                    print('Warning: unsupported callback '+cb_name)
        run_config['config'].update({'callbacks':sup_callback_dict})
    return








def run_trial(args):
    print('loading trial config from ' + args.input_yml)
    with open(args.input_yml,'r') as yml_file:
        yml_config = yaml.load(yml_file,Loader=yaml.FullLoader)
    run_config = list(yml_config.values())[0]
    # check that we loaded the correct configuration
    if run_config['run'] != 'contrib/DBCQ':
        print("wrong yaml file ! expecting DQN agent. Aborting")
        return

    # continue running...
    run_config["config"].update({"env": run_config['env']})

    # attach callbacks if needed
    parse_flow_config(run_config)

    # init ray and start the trial
    # ray.init(local_mode=True)     # uncomment in step-by-step
    ray.init()
    trials = tune.run(
        run_config["run"],
        stop=run_config["stop"],
        config=run_config["config"],
        checkpoint_freq=args.ckpt_freq,
        checkpoint_at_end=True,  # always save checkpoint at end
        return_trials=True)
    print('done')
    return


if __name__ == "__main__":
    args = parse_cmd_line()
    register_env("L2P-v0", lambda config: L2PEnv(config))
    run_trial(args)
    sys.path.remove(proj_root)

