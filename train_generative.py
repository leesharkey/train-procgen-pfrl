import torch
import argparse
from collections import deque
from policies import ImpalaCNN, CNNRecurrent
from procgen import ProcgenGym3Env
from ppo import PPO
import numpy as np
from gym3 import ViewerWrapper, ExtractDictObWrapper, ToBaselinesVecEnv
from vec_env import VecExtractDictObs, VecMonitor, VecNormalize

from util import logger
import os
from train_procgen import create_venv, safe_mean
from torchvision.utils import save_image
from torch.nn import functional as F
from generative_models import VAE



def parse_args():
    parser = argparse.ArgumentParser(
        description='Process procgen training arguments.')

    # Experiment parameters.
    parser.add_argument(
        '--distribution-mode', type=str, default='easy',
        choices=['easy', 'hard', 'exploration', 'memory', 'extreme'])
    parser.add_argument('--env-name', type=str, default='coinrun')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=0)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--exp-name', type=str, default='trial01')
    parser.add_argument('--log-dir', type=str, default='./log')
    parser.add_argument('--model-file', type=str, default=None)
    parser.add_argument('--agent-file', type=str)
    parser.add_argument('--method-label', type=str, default='vanilla')
    parser.add_argument('--epochs', type=int, default=500)


    # Model hyperparameters.
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
    parser.add_argument('--recurrent-policy', type=int, default=0)
    parser.add_argument('--max_recr_seq_len', type=int, default=None)

    configs = parser.parse_args()
    if configs.gpu == -1:
        configs.gpu = None # run on CPU

    return configs

def loss_function(recon_x, x, mu, logvar):
    """ Calculates the difference between predicted and actual:
        - observation
        - agent's recurrent hidden states
        - agent's logprobs
        - rewards

        If this is insufficient to produce high quality samples, then we'll
        add the attentive mask described in Rupprecht et al. (2019). And if
        _that_ is still insufficient, then we'll look into adding a GAN
        discriminator and loss term.
      """
    # TODO this function
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch, configs, train_loader, gen_model, agent, logger, log_dir, device):


    # Set up logging objects
    train_info_buf = deque(maxlen=100)
    # test_info_buf  = deque(maxlen=100)

    optimizer = torch.optim.Adam(gen_model.parameters(), lr=configs.lr) # ensure that the agent params don't update
    gen_model.train()

    logger.info('Start training for {} steps'.format(configs.max_steps))

    # Training cycle
    for batch_idx, (data, _) in enumerate(train_loader):

        optimizer.zero_grad()
        obs = torch.tensor(data['obs'], device=device).float()
        agent_h0 = torch.tensor(data['rec_h_state'], device=device)[:,0,:].float()
        recon_batch, mu, logvar = gen_model(obs, agent_h0)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_info_buf.extend(loss.item())
        optimizer.step()

        # Logging and saving
        if (batch_idx + 1) % configs.log_interval == 0:
            logger.logkv('epoch', epoch)
            logger.logkv('batches', batch_idx)
            logger.logkv('eprewmean',
                         safe_mean([loss for loss in train_info_buf]))
            logger.dumpkvs()

        if (batch_idx + 1) % configs.save_interval == 0:
            model_path = os.path.join(
                log_dir, 'model_epoch{}_batch{}.pt'.format(epoch, batch_idx))
            agent.model.save_to_file(model_path)
            logger.info('Model save to {}'.format(model_path))


def get_fake_data(num_obs, obs_per_input, obs_size, act_space_size, agent_h_size, batch_size):
    """
    Notes from proposal doc:
    Data:
        K-sized minibatch where each element contains:
            (J*observations from timestep T-J to T-1) and
            initial recurrent states at timestep T-J, where:
                J = the number of observations that we're feeding into the VAE
                T = the 0th timestep that we're actually predicting (i.e. the first timestep in the future)
                So we're feeding observations from timestep (T-J):(T-1) (inclusive) into the VAE
                And we also want the VAE to produce initial hidden states at timestep T-J
            Action log probabilities (vector)
            Action (integer)
            Agent value function output (scalar)
            Reward at time t (scalar)
            Timestep (integer)
            Episode number (integer)
            ‘Done’ (boolean/integer)
            Level seed (will help us reinstantiate the same levels later if we want to study them. 
    Labels: 
        Reconstruction part:
            the observations (same as data)
            The initial hidden states (same as data)
        Prediction part:
            k*(L*observations from timestep T to T+L))
    ############################################################
    Args:
        num_obs: total number of observations in the dataset
        obs_per_input: the number of observations in each VAE input (denoted by J in the docstring)
        obs_size: (tuple) - size of each observation (I think for coinrun we will use 64x64x3?)
        act_space_size: number of actions available (15 for coinrun?)
        batch_size: the number of VAE inputs in each batch
    Returns:
        batches of data of size batch_size
    """
    num_batches = num_obs // obs_per_input // batch_size
    obs_seq_size = (obs_per_input,) + obs_size
    data = []
    for batch in range(num_batches):
        actions, timestep, episode, done, lvl_seed = [
            np.random.randint(low=0, high=act_space_size,
                           size=(batch_size, obs_per_input)) for _ in range(5)
        ]
        action_vals, reward = [
            np.random.randn(batch_size, obs_per_input) for _ in range(2)
        ]
        obs = [np.random.rand(*obs_seq_size) for _ in range(batch_size)]
        rec_h_state = [np.random.rand(obs_per_input, agent_h_size) for _ in range(batch_size)]
        # This assumes that we have 1 action for every observation in our data. Not sure if it is standard to take
        # recordings finishing at the state or action.
        action_log_probs = [np.random.rand(obs_per_input,act_space_size)
                            for _ in range(batch_size)]

        # Since enumerating trainloader returns batch_idx, (data,_), we make this a tuple.
        data.append(({
            'actions': actions, 'action_vals': action_vals, 'reward': reward, 'timestep': timestep,
            'episode': episode, 'done': done, 'lvl_seed': lvl_seed, 'obs': obs, 'rec_h_state': rec_h_state,
            'action_log_probs': action_log_probs
        }, None))
    return data

def run():
    configs = parse_args()

    # Set device
    if configs.gpu is not None and configs.gpu>=0 and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Configure logger.
    log_dir = os.path.join(
        configs.log_dir,
        configs.env_name,
        'nlev_{}_{}'.format(configs.num_levels, configs.distribution_mode),
        configs.method_label,
        configs.exp_name,
    )
    logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])

    # Load agent
    train_venv = create_venv(configs, is_valid=False)
    if configs.recurrent_policy:
        policy = CNNRecurrent(
            obs_space=train_venv.observation_space,
            act_space=train_venv.action_space,
        )
    else:
        policy = ImpalaCNN(
            obs_space=train_venv.observation_space,
            num_outputs=train_venv.action_space.n,
        )
    del train_venv
    policy.load_from_file(configs.agent_file)
    # policy.to(device)

    logger.info('Loaded model from {}.'.format(configs.agent_file))


    agent = PPO(
        model=policy,
        optimizer=None,
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
        recurrent=configs.recurrent_policy,
        max_recurrent_sequence_len=configs.max_recr_seq_len,
        max_grad_norm=configs.max_grad_norm,
    )

    # Make gen model
    gen_model = VAE(agent, device)
    gen_model = gen_model.to(device)

    if configs.model_file is not None:
        gen_model.load_from_file(configs.model_file)
        logger.info('Loaded model from {}.'.format(configs.model_file))
    else:
        logger.info('Train agent from scratch.')

    # Set up dataset
    train_loader = None
    train_loader = get_fake_data(1000, 5, (3,64,64), 15, 256, 20)
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.ToTensor()),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)

    # Epoch cycle (train, validate, save samples)
    for epoch in range(1, configs.epochs + 1):
        train(epoch, configs, train_loader, gen_model, agent, logger, log_dir, device)
        # test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = gen_model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

if __name__ == "__main__":
    run()
    # train_loader = get_fake_data(12, 3, (2,2,1), 2, 2)