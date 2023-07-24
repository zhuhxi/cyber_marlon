import sys
sys.path.append("/home/zhx/word/work/cyber_marlon")

from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from stable_baselines3 import A2C, PPO
import argparse

parser = argparse.ArgumentParser(description='cyber marlon ppo')
parser.add_argument('--env_name', default='', type=str)
parser.add_argument('--random_seed', default=1, type=int)
parser.add_argument('--alg_type', default='ppo', type=str)
parser.add_argument('--eval_episodes', default=5, type=int)
parser.add_argument('--training_episode_count', default=50000, type=int)
parser.add_argument('--iteration_count', default=2000, type=int)
parser.add_argument('--defender_maintain_sla', default=0.5, type=float)
parser.add_argument('--with_defender', default=0, type=int)
parser.add_argument('--defender_reset', default=0, type=int)
parser.add_argument('--ownership_goal', default=0.6, type=float)
parser.add_argument('--winning_reward', default=300, type=int)
parser.add_argument('--drl_max_node_cnt', default=10, type=int)
parser.add_argument('--local_vuls_lib_cnt', default=3, type=int)
parser.add_argument('--remote_vuls_lib_cnt', default=8, type=int)
parser.add_argument('--maximum_total_credentials', default=5, type=int)
parser.add_argument('--ports_lib_cnt', default=7, type=int)
parser.add_argument('--env_id', default='ctf', type=str)
parser.add_argument('--env_index', default='0', type=int)
parser.add_argument('--gpu_device', default=1, type=int)
args = parser.parse_args()
args.env_name = f"{args.env_name}_alg_{args.alg_type}_defender{args.with_defender}_defender_goal_{args.defender_maintain_sla}"


LEARN_EPISODES = args.training_episode_count * args.iteration_count
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = 0 # -1
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = True if args.defender_reset == 1 else False
EVALUATE_EPISODES = args.eval_episodes


def train(evaluate_after=False):
    universe = MultiAgentUniverse.build(
        env_id=args.env_id,
        attacker_builder=BaselineAgentBuilder(
            alg_type=PPO if args.alg_type == 'ppo' else A2C,
            policy='MultiInputPolicy',
            env_name=args.env_name
        ),
        defender_builder=BaselineAgentBuilder(
            alg_type=PPO if args.alg_type == 'ppo' else A2C,
            policy='MultiInputPolicy',
            env_name=args.env_name
        ) if args.with_defender == 1 else None,
        attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
        defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        attacker_loss_reward=0,
        defender_loss_reward=0,
        defender_maintain_sla=args.defender_maintain_sla,
        max_timesteps=args.iteration_count,
        args=args
    )

    universe.learn(
        total_timesteps=LEARN_EPISODES,
        n_eval_episodes=100000000,
    )

    if evaluate_after:
        universe.evaluate(
            n_episodes=args.eval_episodes
        )

def setup_seed(random_seed):
    import os
    import torch as th
    import random
    import numpy as np
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(random_seed)
        th.cuda.manual_seed(random_seed)
        th.cuda.set_device(args.gpu_device)

if __name__ == '__main__':
    setup_seed(args.random_seed)
    train(evaluate_after=True)
