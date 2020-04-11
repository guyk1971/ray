import numpy as np
import gym
from gym.spaces import Discrete, Box
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env

tf = try_import_tf()


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1 if done else 0, done, {}

L2PEnv_def_cfg = {'obs_dim':7,'n_act':4}

class L2PEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env that imitates the L2P behaviour
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    def __init__(self, config):
        super(L2PEnv, self).__init__()
        # the observation space include obs_dim float values
        self.obs_dim = config.get('obs_dim',7)
        # Currently assuming discrete action space with n_act actions
        self.act_dim = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.action_space = Discrete(config.get('n_act',4))
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = Box(low=-np.inf, high=np.inf,shape=(self.obs_dim,), dtype=np.float32)
        self.max_path_length = 40

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.step_idx=0
        return self.observation_space.sample()


    def step(self, action):

        if (not isinstance(action,int)) or (action<0) or (action>=self.action_space.n):
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        self.step_idx += 1

        state = self.observation_space.sample()
        done = False
        if self.step_idx == self.max_path_length:
            done = True
            self.step_idx = 0
        reward = 1.0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return state, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass

register_env("L2P-v0", lambda config: L2PEnv(config))
