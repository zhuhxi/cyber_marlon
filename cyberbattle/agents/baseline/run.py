#!/usr/bin/python3.9

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-
"""CLI to run the baseline Deep Q-learning and Random agents
   on a sample CyberBattle gym environment and plot the respective
   cummulative rewards in the terminal.

Example usage:

    python3.9 -m run --training_episode_count 50  --iteration_count 9000 --rewardplot_width 80  --chain_size=20 --ownership_goal 1.0

"""
import torch
import gym
import logging
import os
import sys
sys.path.append('../../../')
import asciichartpy
import numpy as np
import torch as th
import random
import argparse
import cyberbattle._env.cyberbattle_env as cyberbattle_env
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import cyberbattle.agents.baseline.agent_dql as dqla
import cyberbattle.agents.baseline.agent_wrapper as w
import cyberbattle.agents.baseline.plotting as p
import cyberbattle.agents.baseline.learner as learner
from scripts.Path_producer import Path_producer

parser = argparse.ArgumentParser(description='Run simulation with DQL baseline agent.')

parser.add_argument('--training_episode_count', default=30000, type=int)
parser.add_argument('--eval_episode_count', default=10, type=int)
parser.add_argument('--iteration_count', default=600, type=int)
parser.add_argument('--reward_goal', default=2180, type=int)
parser.add_argument('--ownership_goal', default=0.6, type=float)
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--env_name', default='ctf', type=str)
parser.add_argument('--env_index', default=0, type=int)
parser.add_argument('--maximum_node_cnt', default=10, type=int)
parser.add_argument('--local_vuls_lib_cnt', default=3, type=int)
parser.add_argument('--remote_vuls_lib_cnt', default=8, type=int)
parser.add_argument('--ports_lib_cnt', default=7, type=int)
parser.add_argument('--maximum_total_credentials', default=5, type=int)
parser.add_argument('--test_random_agent', default=False, type=bool)
parser.set_defaults(run_random_agent=False)

args = parser.parse_args()

# logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")


# seed fix
def setup_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(random_seed)
        th.cuda.manual_seed(random_seed)


setup_seed(args.random_seed)

print(f"torch cuda available={torch.cuda.is_available()}")


gym_name = f'CtfLocalvuls{args.local_vuls_lib_cnt}Remotevuls{args.remote_vuls_lib_cnt}Ports{args.ports_lib_cnt}' if args.env_name == 'ctf' else f'ActiveDirectory'
cyberbattleenv = gym.make(f'{gym_name}-v{args.env_index}',
                          attacker_goal=cyberbattle_env.AttackerGoal(
                              own_atleast_percent=args.ownership_goal,
                              reward=args.reward_goal),
                          maximum_node_count=10)

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=args.maximum_total_credentials,
    maximum_node_count=args.maximum_node_cnt,
    identifiers=cyberbattleenv.identifiers
)

# Run Deep Q-learning
if not args.test_random_agent:
    Path_producer().mkdir(f'/Outcome/DQL/seed{args.random_seed}')

    dqn_learning_run = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattleenv,
        environment_properties=ep,
        learner=dqla.DeepQLearnerPolicy(
            ep=ep,
            gamma=0.99,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.00025),  # torch default is 1e-2
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=0.90,
        render=False,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=args.training_episode_count*args.iteration_count//2,
        epsilon_minimum=0.10,
        verbosity=Verbosity.Quiet,
        seed=args.random_seed,
        title="DQL"
    )
else:
    Path_producer().mkdir(f'/Outcome/Random/seed{args.random_seed}')
    random_run = learner.epsilon_greedy_search(
        cyberbattleenv,
        ep,
        learner=learner.RandomPolicy(),
        episode_count=args.training_episode_count,
        iteration_count=args.iteration_count,
        epsilon=1.0,  # purely random
        render=False,
        verbosity=Verbosity.Quiet,
        title="Random"
    )

# all_runs = []
#
# all_runs.append(dqn_learning_run)
#
# if args.run_random_agent:
#     random_run = learner.epsilon_greedy_search(
#         cyberbattleenv,
#         ep,
#         learner=learner.RandomPolicy(),
#         episode_count=args.eval_episode_count,
#         iteration_count=args.iteration_count,
#         epsilon=1.0,  # purely random
#         render=False,
#         verbosity=Verbosity.Quiet,
#         title="Random search"
#     )
#     all_runs.append(random_run)
#
# colors = [asciichartpy.red, asciichartpy.green, asciichartpy.yellow, asciichartpy.blue]
#
# print("Episode duration -- DQN=Red, Random=Green")
# print(asciichartpy.plot(p.episodes_lengths_for_all_runs(all_runs), {'height': 30, 'colors': colors}))
#
# print("Cumulative rewards -- DQN=Red, Random=Green")
# c = p.averaged_cummulative_rewards(all_runs, args.rewardplot_width)
# print(asciichartpy.plot(c, {'height': 10, 'colors': colors}))
