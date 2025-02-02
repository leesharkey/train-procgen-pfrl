import torch
import argparse
from collections import deque
from policies import ImpalaCNN
from procgen import ProcgenGym3Env
from ppo import PPO
import numpy as np
from gym3 import ViewerWrapper, ExtractDictObWrapper, ToBaselinesVecEnv, VideoRecorderWrapper
from vec_env import VecExtractDictObs, VecMonitor, VecNormalize

def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env-name', type=str, default='starpilot')
    parser.add_argument('--num-envs', type=int, default=1)
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
    parser.add_argument('--save-interval', type=int, default=100)

    # render parameters
    parser.add_argument('--tps', type=int, default=15, help="env fps")
    parser.add_argument('--vid-dir', type=str, default=None)

    configs = parser.parse_args()
    return configs

def create_venv_render(config, is_valid=False):
    venv = ProcgenGym3Env(
        num=config.num_envs,
        env_name=config.env_name,
        num_levels=0 if is_valid else config.num_levels,
        start_level=0 if is_valid else config.start_level,
        distribution_mode=config.distribution_mode,
        num_threads=1,
        render_mode="rgb_array"
    )
    venv = ViewerWrapper(venv, tps=config.tps, info_key="rgb")
    if config.vid_dir is not None:
        venv = VideoRecorderWrapper(venv, directory=config.vid_dir,
                                    info_key="rgb", fps=config.tps)
    venv = ToBaselinesVecEnv(venv)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    return VecNormalize(venv=venv, ob=False)



configs = parse_args()

venv = create_venv_render(configs, is_valid=True)

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
    )


if configs.gpu >= 0:
    device = torch.device(f"cuda:{configs.gpu}")
else:
    device = torch.device('cpu')
agent.model.load_from_file(configs.model_file, device=device)


steps = np.zeros(configs.num_envs, dtype=int)
obs = venv.reset()

env_max_steps=1000

while True:
    with agent.eval_mode():
        assert not agent.training
        action = agent.batch_act(obs)
        obs, reward, done, infos = venv.step(action)
        steps += 1
        reset = steps == env_max_steps
        steps[done] = 0

        agent.batch_observe(
            batch_obs=obs,
            batch_reward=reward,
            batch_done=done,
            batch_reset=reset
        )
