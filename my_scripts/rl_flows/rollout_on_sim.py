import argparse
import numpy as np
import yaml
import pickle
import ray
import gym
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.registry import get_agent_class,CONTRIBUTED_ALGORITHMS,ALGORITHMS
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.policy.tests.test_policy import TestPolicy
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.memory import ray_get_and_free
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from ray.rllib.models.preprocessors import get_preprocessor
from tqdm import tqdm
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

from my_scripts.sandbox.my_custom_envs import L2PEnv,L2PEnv_def_cfg


def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str, help='agent class e.g. DQN,random,PPO,DBCQ. if not random, need to provide checkpoint',default='random')
    parser.add_argument('env',type=str,help='string of registered environment to run on')
    # note: the checkpoint is assumed to be saved in a folder <path_to_results>/checkpoint_<iter>
    # note that inside this folder there is a file with the same name 'checkpoint_<iter>'. we expect to get the folder.
    parser.add_argument('--ckpt',type=str,help='path to checkpoint (ignored for random agent)',default=None)
    parser.add_argument('-n',"--num-steps", type=int, default=1000, help='number of steps in the env to play with')
    parser.add_argument('-o','--output',type=str,help='location of output file')
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()
    return args

#########################################
#region  callbacks
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

#endregion
##########################################

#########################################
#region random policy
# see rollout_worker_custom_workflow.py for reference
class RandomPolicy(TestPolicy):
    """a random policy written from scratch.
    You might find it more convenient to extend TF/TorchPolicy instead
    for a real policy.
    """
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return random actions
        # todo : add action prob (uniform) as info
        return [self.action_space.sample() for _ in obs_batch], [], {}
#endregion
#########################################
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



def do_rollout_workset(env_creator, policy, config, n_timesteps):
    # Setup policy and policy evaluation actors
    # option 1 : defining the workers manually
    # output_creator = (lambda ioctx: JsonWriter(config["output"], ioctx, max_file_size=10000000))
    # workers = [
    #     RolloutWorker.as_remote().remote(env_creator,policy.__class__,output_creator=output_creator)
    #     for _ in range(config["num_workers"])
    # ]

    # option 2 : using worker set
    workers = WorkerSet(
        policy=policy.__class__,
        env_creator=env_creator,
        trainer_config = with_common_config(config),  # extending the COMMON_CONFIG of the trainer class. need to do COMMON_CONFIG.with_updates(config)
        num_workers=config['num_workers'])

    # Broadcast weights to the policy evaluation workers
    if not isinstance(policy,RandomPolicy):
        # set weights to the local worker
        # weights = policy.get_weights()        # this is the straight forward way.
        # weights = ray.put(policy.get_weights())         # see sync_samples_optimizer.py line 48 (in step method)
        weights = ray.put({DEFAULT_POLICY_ID: policy.get_weights()})  # see rollout_worker_custom_workflow.py
        workers.local_worker().set_weights(weights.value)
        # to check that the remote workers got the weights do:
        # workers.remote_workers()[0].get_weights.remote(DEFAULT_POLICY_ID).value
        # workers.local_worker().get_weights(DEFAULT_POLICY_ID)
        for w in workers.remote_workers():
            w.set_weights.remote(weights)


    # use workers to sample from env

    # the following code was copied from sync_samples_optimizer:
    samples = []
    # with tqdm(total=n_timesteps) as pbar:
    #     while sum(s.count for s in samples) < n_timesteps:
    #         if workers.remote_workers():
    #             samples.extend(ray_get_and_free([e.sample.remote() for e in workers.remote_workers()]))
    #         else:
    #             samples.append(workers.local_worker().sample())

    with tqdm(total=n_timesteps) as pbar:
        total_samples = 0
        while total_samples < n_timesteps:
            if workers.remote_workers():
                samples_batch=ray_get_and_free([e.sample.remote() for e in workers.remote_workers()])
                samples.extend(samples_batch)
            else:
                samples_batch=workers.local_worker().sample()
                samples.append(samples_batch)
            pbar.update(sum(s.count for s in samples)-total_samples)
            total_samples = sum(s.count for s in samples)

    # samples = SampleBatch.concat_samples(samples)
    return



# another option is to do a single agent rollout (see saving_experiences.py)
def do_rollout_single(env_creator, policy, config, n_timesteps):
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(config['output'])

    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:
    env = env_creator()

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    total_ts=0
    for eps_id in range(100):
        obs = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            # action = env.action_space.sample()
            action = policy.compute_single_action(obs)
            new_obs, rew, done, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=prep.transform(new_obs))
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        total_ts += t
        writer.write(batch_builder.build_and_reset())
        if total_ts > n_timesteps:
            break

def run_trial(args):

    if args.env == "L2P-v0":
        env_config = {}
        env_creator = lambda _: L2PEnv(L2PEnv_def_cfg)
    else:
        env_creator = lambda _: gym.make(args.env)


    env = env_creator(None)
    # init ray and start the trial
    ray.init(local_mode=True)     # uncomment in step-by-step
    # ray.init()

    # define the policy object
    if args.agent=='random':
        policy = RandomPolicy(env.observation_space, env.action_space, {})
    else:
        supported_algos = list(ALGORITHMS.keys())+list(CONTRIBUTED_ALGORITHMS.keys())
        # check that the agent exists
        assert args.agent in supported_algos, "Error: algorithm is not supported"
        # check that we have a checkpoint
        assert args.ckpt and os.path.exists(args.ckpt), "Error : checkpoint not provided or not found"
        # load the checkpoint
        cls=get_agent_class(args.agent)
        with open(os.path.join(os.path.split(args.ckpt)[0],'params.pkl'),'rb') as f:
            ckpt_cfg=pickle.load(f)
        assert args.env==ckpt_cfg['env'], "mismatch between environments"
        agent=cls(env=args.env,config=ckpt_cfg)
        # the checkpoint is saved as file in '<path_to_results>/checkpoint_XXX/checkpoint-XXX'
        agent.restore(os.path.join(args.ckpt,os.path.split(args.ckpt)[1].replace('_','-')))
        policy = agent.get_policy()


    # config should override keys from the trainer.COMMON_CONFIG
    config={'output':args.output,
            'num_gpus': (1 if args.gpu else 0),
            'num_workers':args.num_workers,
            'rollout_fragment_length':1000}


    do_rollout_workset(env_creator, policy, config, args.num_steps)
    # do_rollout_single(env_creator, policy, config, args.num_steps)

    print('done')
    return


if __name__ == "__main__":
    args = parse_cmd_line()
    register_env("L2P-v0", lambda config: L2PEnv(config))
    run_trial(args)
    sys.path.remove(proj_root)

