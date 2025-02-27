import numpy as np

from gym import utils
from envs.goal_env_ext.fetch import fetch_env


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', distance_threshold=0.05, **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 0.7,
            'table0:slide1': 0.3,
            'table0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/slide.xml', has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type, object_location_same_as_gripper_probability = 0.0, **kwargs)
        utils.EzPickle.__init__(self)
