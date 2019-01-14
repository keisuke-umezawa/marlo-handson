import argparse
import os

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer

import marlo
from marlo import experiments

from PIL import Image


class Monitor(gym.Wrapper):
    def __init__(self, env):
        super(Monitor, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = Image.fromarray(obs)
        obs.thumbnail((84, 84), Image.ANTIALIAS)
        obs = np.asarray(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = Image.fromarray(obs)
        obs.thumbnail((84, 84), Image.ANTIALIAS)
        obs = np.asarray(obs)
        return obs


def make_env(env_name, env_seed=0, demo=False):
    join_tokens = marlo.make(
        env_name,
        params=dict(
            comp_all_commands=["move", "turn"],
            allowContinuousMovement=True,
            videoResolution=[336, 336],
            kill_clients_retry=10,
            step_sleep=0.01,
            kill_clients_after_num_rounds=100,
            prioritise_offscreen_rendering=not demo,
        ))
    env = marlo.init(join_tokens[0])
    env = Monitor(env)

    obs = env.reset()
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    env.seed(int(env_seed))
    return env


def parse_arch(arch, n_actions):
    if arch == 'nature':
        return links.Sequence(
            links.NatureDQNHead(n_input_channels=3),
            L.Linear(512, n_actions),
            DiscreteActionValue
        )

    elif arch == 'doubledqn':

        class SingleSharedBias(chainer.Chain):
            """Single shared bias used in the Double DQN paper.
            You can add this link after a Linear layer with nobias=True to implement a
            Linear layer with a single shared bias parameter.
            See http://arxiv.org/abs/1509.06461.
            """
        
            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.bias = chainer.Parameter(0, shape=1)
        
            def __call__(self, x):
                return x + F.broadcast_to(self.bias, x.shape)

        return links.Sequence(
            links.NatureDQNHead(n_input_channels=3),
            L.Linear(512, n_actions, nobias=True),
            SingleSharedBias(),
            DiscreteActionValue
        )

    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(n_input_channels=3),
            L.Linear(256, n_actions),
            DiscreteActionValue
        )

    elif arch == 'dueling':
        return DuelingDQN(n_actions, n_input_channels=3)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {
        'DQN': agents.DQN,
        'DoubleDQN': agents.DoubleDQN,
        'PAL': agents.PAL
    }[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MarLo-FindTheGoal-v0',
                        help='Marlo env to perform algorithm on.')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--arch', type=str, default='nature',
                        choices=['nature', 'nips', 'dueling', 'doubledqn'],
                        help='Network architecture to use.')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--max-episode-len', type=int,
                        default=30 * 60 * 60 // 4,  # 30 minutes with 60/4 fps
                        help='Maximum number of timesteps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--eval-interval', type=int, default=10 ** 5,
                        help='Frequency (in timesteps) of evaluation phase.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--agent', type=str, default='DQN',
                        choices=['DQN', 'DoubleDQN', 'PAL'])
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate.')
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Use prioritized experience replay.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    experiments.set_log_base_dir(args.out_dir)
    print('Output files are saved in {}'.format(args.out_dir))

    env = make_env(args.env, env_seed=args.seed, demo=args.demo)

    n_actions = env.action_space.n

    q_func = parse_arch(args.arch, n_actions)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func)
        # Turn off explorer
        explorer = explorers.Greedy()

    # Use the Nature paper's hyperparameters
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.6,
            beta0=0.4, betasteps=betasteps)
    else:
        rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        x = x.transpose(2, 0, 1)
        return np.asarray(x, dtype=np.float32) / 255

    Agent = parse_agent(args.agent)
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator='sum',
        phi=phi
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.out_dir,
            save_best_so_far_agent=False,
            max_episode_len=args.max_episode_len,
            eval_env=env,
        )


if __name__ == '__main__':
    main()
