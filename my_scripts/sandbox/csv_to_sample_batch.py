import gym
import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
import argparse

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str, help='the input experience csv file')
    parser.add_argument('-o','--output_json_path', type=str, default=None,
                        help='folder to save the experience json file for ray')
    args = parser.parse_args()
    return args


def csv_to_bsjson(csv_filename, json_folder=None):
    if csv_filename.endswith('.csv') and os.path.exists(csv_filename):
        print('reading csv: ' + csv_filename)
        df = pd.read_csv(csv_filename)
    else:
        print('csv not found: ' + csv_filename)
        return
    if json_folder is None:
        json_folder = os.path.splitext(csv_filename)[0]

    episode_ids = df['episode_id'].unique()
    print('found {0} episodes'.format(len(episode_ids)))

    state_columns = [col for col in df.columns if col.startswith('state_feature')]
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(json_folder)
    print('start conversion...')
    for e_id_i in tqdm(range(len(episode_ids))):
        # progress_bar.update(e_id)
        e_id = episode_ids[e_id_i]
        df_episode_transitions = df[df['episode_id'] == e_id]
        if len(df_episode_transitions) < 2:
            # we have to have at least 2 rows in each episode for creating a transition
            print('dropped short episode {0}'.format(e_id))
            continue
        transitions = []
        for (_, current_transition), (_, next_transition) in zip(df_episode_transitions[:-1].iterrows(),
                                                                 df_episode_transitions[1:].iterrows()):
            obs = np.array([current_transition[col] for col in state_columns])
            obs_tp1 = np.array([next_transition[col] for col in state_columns])
            action = int(current_transition['action'])  # assuming Discrete action space
            reward = current_transition['reward']
            # info is extracted from the csv but currently not saved in the _storage
            info = {'all_action_probabilities': ast.literal_eval(current_transition['all_action_probabilities'])}
            transitions.append(
                {'obs_t': obs, 'action': action, 'reward': reward, 'obs_tp1': obs_tp1, 'done': False, 'info': info})
        # set the done flag of the last transition to True
        transitions[-1]['done'] = True
        # start write the batch_sample
        # prev_action = np.zeros_like(env.action_space.sample())
        prev_action = 0
        prev_reward = 0
        for t, trans in enumerate(transitions):
            action = trans['action']
            obs = trans['obs_t']
            new_obs = trans['obs_tp1']
            rew = trans['reward']
            done = trans['done']
            info = trans['info']
            batch_builder.add_values(
                t=t,
                eps_id=e_id,
                agent_index=0,
                obs=obs,
                actions=action,
                action_prob=info['all_action_probabilities'][action],  # putting the probability of the chosen action
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info,
                new_obs=new_obs)
            prev_action = action
            prev_reward = rew
        writer.write(batch_builder.build_and_reset())
    print('Done. experience json was saved in '+json_folder)
    return



if __name__ == "__main__":
    args = parse_cmd_line()
    csv_to_bsjson(args.input_csv,args.output_json_path)
