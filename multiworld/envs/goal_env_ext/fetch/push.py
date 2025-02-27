from gym import utils
from multiworld.envs.goal_env_ext.fetch import fetch_env


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', distance_threshold=0.05, OLSAGP=0.0, n_substeps=20, **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/push.xml', has_object=True, block_gripper=True, n_substeps=n_substeps,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=distance_threshold,
            initial_qpos=initial_qpos, reward_type=reward_type, object_location_same_as_gripper_probability=OLSAGP,
            **kwargs)
        utils.EzPickle.__init__(self)
