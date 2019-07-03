# Created by Xingyu Lin, 04/09/2018 

# Added by Ziwen Zhuang (27/06/2019) for registering rope environments into gym
import gym
from gym.envs.registration import register


REGISTERED = False
def register_rope_envs():
    # avoid multiple registering
    global REGISTERED
    if REGISTERED:
        return
    else:
        REGISTERED = True

    register(
        id='FetchPush-v0',
        entry_point='multiworld.envs.goal_env_ext.fetch.push:FetchPushEnv',
        kwargs= {
                'distance_threshold': 0.05,
                'distance_threshold_obs': 0.0,
                'horizon': 100,
                'image_size': 48,
                'noisy_reward_fn': None,
                'noisy_reward_fp': None,
                'state_estimation_input_noise': 0,
                'state_estimation_re...ard_noise': 0,
                'use_auxiliary_loss': True,
                'use_image_goal': True,
                'use_true_reward': False,
                'use_visual_observation': True,
                'with_goal': False,
            }, # copied from baselines_hrl (Xingyu)
    )
    register(
        id='FetchReach-v0',
        entry_point='multiworld.envs.goal_env_ext.fetch.reach:FetchReachEnv',
        kwargs= {
                'distance_threshold_obs': 0.0,
                'horizon': 100,
                'image_size': 48,
                'noisy_reward_fn': None,
                'noisy_reward_fp': None,
                'state_estimation_input_noise': 0,
                'state_estimation_re...ard_noise': 0,
                'use_auxiliary_loss': False,
                'use_image_goal': True,
                'use_true_reward': False,
                'use_visual_observation': True,
                'with_goal': False
            }, # copied from baselines_hrl (Xingyu)
    )
    register(
        id='Reacher-v0',
        entry_point='multiworld.envs.goal_env_ext.reacher.reacher_env:ReacherEnv',
        kwargs= {
                'horizon': 100,
                'n_substeps': 10,
                'action_type': 'velocity',
                'image_size': 48,
                'distance_threshold': 1e-2,
                'distance_threshold_obs': 0.,
                'with_goal': False,
                'use_image_goal': True,
                'use_visual_observation': True
            }, # copied from baselines_hrl (Xingyu)
    )

