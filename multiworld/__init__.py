from multiworld.envs.mujoco import register_mujoco_envs
from multiworld.envs.pygame import register_pygame_envs
from multiworld.envs.goal_env_ext import register_rope_envs


def register_all_envs():
    register_mujoco_envs()    
    register_pygame_envs()
    register_rope_envs()
