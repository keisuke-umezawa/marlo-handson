from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

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


def main():
    parser = argparse.ArgumentParser()
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
                        type=int, default=10 ** 5,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.1,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.05,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--max-episode-len', type=int,
                        default=30 * 60 * 60 // 4,  # 30 minutes with 60/4 fps
                        help='Maximum number of timesteps for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=1000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval',
                        type=int, default=1 * 10 ** 4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--eval-interval', type=int, default=10 ** 5,
                        help='Frequency (in timesteps) of evaluation phase.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    experiments.set_log_base_dir(args.out_dir)
    print('Output files are saved in {}'.format(args.out_dir))

    def make_env(env_seed=0):
        join_tokens = marlo.make(
            "MarLo-FindTheGoal-v0",
            params=dict(
                allowContinuousMovement=["move", "turn"],
                videoResolution=[336, 336],
                kill_clients_after_num_rounds=500
            ))
        env = marlo.init(join_tokens[0])
        env = Monitor(env)

        obs = env.reset()
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        env.seed(int(env_seed))
        return env

    env = make_env(env_seed=args.seed)

    n_actions = env.action_space.n
    q_func = links.Sequence(
        links.NatureDQNHead(n_input_channels=3),
        L.Linear(512, n_actions),
        DiscreteActionValue
    )

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((3, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.out_dir, 'model'))

    # Use the same hyper parameters as the Nature paper's
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    def phi(x):
        # Feature extractor
        x = x.transpose(2, 0, 1)
        return np.asarray(x, dtype=np.float32) / 255

    agent = agents.DQN(
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
