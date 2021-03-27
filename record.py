import torch
import argparse
from collections import deque
from policies import ImpalaCNN, CNNRecurrent
from procgen import ProcgenGym3Env
from ppo import PPO
import numpy as np
from gym3 import ViewerWrapper, ExtractDictObWrapper, ToBaselinesVecEnv
from vec_env import VecExtractDictObs, VecMonitor, VecNormalize
import pandas as pd
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env-name', type=str, default='starpilot')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=0)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--exp-name', type=str, default='trial01')
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--method-label', type=str, default='vanilla')

    # PPO parameters.
    parser.add_argument('--gpu', type=int, default=0, help="If on cpu, set to -1")
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--nsteps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=25_000_000)
    parser.add_argument('--recurrent-policy', type=int, default=0)
    parser.add_argument('--save-interval', type=int, default=100)

    configs = parser.parse_args()
    if configs.gpu == -1:
        configs.gpu = None # run on CPU

    return configs

# def get_render_func(venv):
#     """Get a render function"""
#     if hasattr(venv, 'envs'):
#         return venv.envs[0].render
#     elif hasattr(venv, 'venv'):
#         return get_render_func(venv.venv)
#     elif hasattr(venv, 'env'):
#         return get_render_func(venv.env)

    # return None

def create_venv(config, is_valid=False):
    venv = ProcgenGym3Env(
        num=config.num_envs,
        env_name=config.env_name,
        num_levels=0 if is_valid else config.num_levels,
        start_level=0 if is_valid else config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=1#, #in train_procgen num_threads = config.num_threads
        # render_mode="rgb_array"
    )
    #venv = ExtractDictObWrapper(venv, key="rgb")
    # venv = ViewerWrapper(venv, tps=15, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)



configs = parse_args()

venv = create_venv(configs, is_valid=True) #when true uses unlimited levels and lowest seed is 0
# TODO(Max/Joe) While recording, it's unneccessary (and much slower) to render
#  at the same time. This should be changed to whatever kind of venvs are used
#  during train/eval.

if configs.recurrent_policy:
    policy = CNNRecurrent(
        obs_space=venv.observation_space,
        act_space=venv.action_space,
    )
else:
    policy = ImpalaCNN(
        obs_space=venv.observation_space,
        num_outputs=venv.action_space.n,
    )

optimizer = torch.optim.Adam(policy.parameters(), lr=configs.lr, eps=1e-5)
agent = PPO(
        model=policy,
        optimizer=optimizer,
        gpu=configs.gpu,
        gamma=configs.gamma,
        lambd=configs.lam,
        value_func_coef=configs.vf_coef,
        entropy_coef=configs.ent_coef,
        update_interval=configs.nsteps * configs.num_envs,
        minibatch_size=configs.batch_size,
        epochs=configs.nepochs,
        clip_eps=configs.clip_range,
        clip_eps_vf=configs.clip_range,
        max_grad_norm=configs.max_grad_norm,
        recurrent=configs.recurrent_policy
)

agent.model.load_from_file(configs.model_file, agent.device)


steps = np.zeros(configs.num_envs, dtype=int)
ep_num = np.zeros(configs.num_envs, dtype=int)
obs = venv.reset()

env_max_steps=1000
episode_number = 0
# reward = [0] #setting for the first iteration (iteration 0)

column_names = ['episode', 'timestep', 'obs', 'reward', 'done', 'action', 'action_log_probs', 'agent_val_fx', 'init_recurrent_states', 'level_seed']
data = pd.DataFrame(columns = column_names)

while True:
    with agent.eval_mode():
        assert not agent.training
        print('recurrent states', agent.test_recurrent_states)
        # print('obs', obs)
        # print('steps', steps)
        # print('steps[0]', steps[0])

        #agent.test_recurrent_states
        #obs
        #action -- make sure this is action on the current obs
        #reward
        #done
        #steps

        # We need to record the observations and the hidden states of the agent
        # (see research plan for details).

        # You can find the recurrent states at each timestep in:
        # agent.test_recurrent_states
        #
        # The observations are in obs.
        #
        # level seed is in infos[0]['level_seed']

        action = agent.batch_act(obs)

        print('obs', obs)

        if steps > 0: 
            data = data.append({
                'episode': episode_number,
                'level_seed': infos[0]['level_seed'], #doesn't exist on 0th iteration
                'done': done, #doesn't exist on 0th iteration
                'timestep': steps[0], 
                'reward': reward[0],
                'agent_val_fx': agent.eval_values, #doesn't exist on 0th iteration
                'action': action[0],
                'action_log_probs': agent.eval_action_distrib, #doesn't exist on 0th iteration
                'init_recurrent_states': agent.test_recurrent_states, 
                'obs': obs
                }, ignore_index=True)
        # According to the image in the research plan, time increments here. t <- t+1
        # print('action', action)
        # print('env', venv.action_space)
        obs, reward, done, infos = venv.step(action)
        print('infos', infos)
        # print('level seed', infos[0]['level_seed'])
        steps += 1
        print('action distrib', agent.eval_action_distrib)
        print('eval values', agent.eval_values)
        print('done', done)
        print('steps', steps)
        reset = steps == env_max_steps
        steps[done] = 0
        if done[0] or reset:
            episode_number += 1
        #are steps resetting when env_max_steps is reached or only for dones?

        # self.agent.eval_values 
        # self.agent.train_action_distrib
        if steps % 10 == 0:
            data.to_csv('data.csv', index=False)
            print(data)

        agent.batch_observe(
            batch_obs=obs,
            batch_reward=reward,
            batch_done=done,
            batch_reset=reset
        )


class CoinrunDataset(Dataset):
    """Coinrun dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.coinrun_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.coinrun_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        # return sample

        return self.coinrun_frame.iloc[idx, 1:]


# Action log probabilities (vector)
# Agent value function output (scalar)
# Episode number (integer)
# better timestep variable see note 