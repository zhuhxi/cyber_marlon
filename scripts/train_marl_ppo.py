import sys
sys.path.append("/home/zhx/word/work/MARLon/")

from marlon.baseline_models.multiagent.multiagent_universe import MultiAgentUniverse
from marlon.baseline_models.multiagent.baseline_marlon_agent import BaselineAgentBuilder
from stable_baselines3 import A2C, PPO


ENV_MAX_TIMESTEPS = 1500
# LEARN_TIMESTEPS = 300_000
LEARN_TIMESTEPS = int(1e8)
# Set this to a large value to stop at LEARN_TIMESTEPS instead.
LEARN_EPISODES = 10_000_000
ATTACKER_INVALID_ACTION_REWARD_MODIFIER = 0
ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER = 0
DEFENDER_INVALID_ACTION_REWARD = 0 # -1
DEFENDER_RESET_ON_CONSTRAINT_BROKEN = False
EVALUATE_EPISODES = 5
ATTACKER_SAVE_PATH = 'ppo_marl_attacker.zip'
DEFENDER_SAVE_PATH = 'ppo_marl_defender.zip'


def train(evaluate_after=False):
    universe = MultiAgentUniverse.build(
        env_id='CyberBattleToyCtf-v0',
        attacker_builder=BaselineAgentBuilder(
            alg_type=PPO,
            policy='MultiInputPolicy'
        ),
        # defender_builder=BaselineAgentBuilder(
        #     alg_type=PPO,
        #     policy='MultiInputPolicy'
        # ),
        attacker_invalid_action_reward_modifier=ATTACKER_INVALID_ACTION_REWARD_MODIFIER,
        attacker_invalid_action_reward_multiplier=ATTACKER_INVALID_ACTION_REWARD_MULTIPLIER,
        defender_invalid_action_reward_modifier=DEFENDER_INVALID_ACTION_REWARD,
        defender_reset_on_constraint_broken=DEFENDER_RESET_ON_CONSTRAINT_BROKEN,
        attacker_loss_reward=0,
        defender_loss_reward=0,
        defender_maintain_sla=0.5
    )

    universe.learn(
        total_timesteps=LEARN_TIMESTEPS,
        n_eval_episodes=LEARN_EPISODES
    )

    universe.save(
        attacker_filepath=ATTACKER_SAVE_PATH,
        defender_filepath=DEFENDER_SAVE_PATH
    )

    if evaluate_after:
        universe.evaluate(
            n_episodes=EVALUATE_EPISODES
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
        th.cuda.set_device(1)

if __name__ == '__main__':
    setup_seed(1)
    train(evaluate_after=True)
