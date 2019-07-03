from gym import utils
from multiworld.envs.goal_env_ext.goal_env_ext import GoalEnvExt
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path

import os
from collections import OrderedDict
import numpy as np
import mujoco_py


class ReacherEnv(GoalEnvExt, utils.EzPickle):
    def __init__(self, model_path='./reacher/reacher.xml', distance_threshold=1e-2, distance_threshold_obs=0,
                 n_substeps=10,
                 horizon=50, image_size=48, action_type='velocity',
                 with_goal=False,
                 use_visual_observation=True,
                 use_image_goal=True,
                 use_true_reward=False, **kwargs):

        GoalEnvExt.__init__(self, model_path=model_path, n_substeps=n_substeps, horizon=horizon, image_size=image_size,
                            use_image_goal=use_image_goal, use_visual_observation=use_visual_observation,
                            with_goal=with_goal, reward_type='sparse', distance_threshold=distance_threshold,
                            distance_threshold_obs=distance_threshold_obs, use_true_reward=use_true_reward, n_actions=2,
                            **kwargs)
        utils.EzPickle.__init__(self)

        self.action_type = action_type

    # Implementation of functions from GoalEnvExt
    # ----------------------------

    def _reset_sim(self):
        # Sample goal and render image
        qpos = self.np_random.uniform(low=-2 * np.pi, high=2 * np.pi, size=self.model.nq)
        self.set_state(qpos, qvel=self.init_qvel)
        self.goal_state = self.get_end_effector_location()

        qpos[-2:] = self.goal_state
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self.goal_observation = self.render(mode='rgb_array', depth=False)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal_state
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        return True

    def _get_obs(self):
        if self.use_visual_observation:
            obs = self.render(mode='rgb_array', depth=False)
        else:
            theta = self.sim.data.qpos.flat[:2]
            obs = np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_end_effector_location() - self.get_goal_location()
            ])
        if self.use_image_goal:
            desired_goal = self.goal_observation
            achieved_goal = obs
        else:
            desired_goal = self.get_goal_location()
            achieved_goal = self.get_end_effector_location()

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        info = {
            'ag_state': self.get_end_effector_location().copy(),
            'g_state': self.get_goal_location().copy()
        }
        return info

    def _set_action(self, ctrl):
        if self.action_type == 'force':
            self.send_control_command(ctrl)
        elif self.action_type == 'velocity':
            self.send_control_command(np.asarray([0., 0.]))  # Set the force to be zero
            self.set_joint_velocity(np.asarray(ctrl) * 10)

    def _viewer_setup(self):
        self.viewer.cam.lookat[0] = 0.0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 1.3

    def set_hidden_goal(self):
        self.sim.model.geom_rgba[9, :] = np.asarray([0., 0., 0, 0.])  # Make the goal transparent

    # Env specific helper functions
    # ----------------------------
    def set_goal_location(self, goalPos):
        self.sim.data.qpos[2] = goalPos[0]
        self.sim.data.qpos[3] = goalPos[1]

    def send_control_command(self, ctrl):
        assert len(ctrl) == 2
        self.sim.data.ctrl[0] = ctrl[0]
        self.sim.data.ctrl[1] = ctrl[1]

    def set_joint_velocity(self, jointVel):
        self.sim.data.qvel[0:2] = jointVel

    def get_end_effector_location(self):
        return np.squeeze(self.sim.data.body_xpos[3:4, 0:2]).copy()

    def get_goal_location(self):
        return np.squeeze(self.sim.data.body_xpos[4:5, 0:2]).copy()

    def _sample_goal(self):
        while True:
            goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(goal) < 2:
                break
        return goal


# Start Adding interface for multiworld environment collection

    def get_goal(self):
        ''' Get goal that are stored currently in this object.
            (refering to the rlkit interface, provide what it need...)
        '''
        return {
            'desired_goal': self.goal_state,
            'state_desired_goal': self.goal_state,
        }

    def set_to_goal(self, goal):
        ''' @brief: Set the goal which stores in this object and move the object to
                    the position by the given goal.
            @args: 'goal': must be a numpy array (not matrix)
        '''
        self.goal_state = goal['state_desired_goal']
        
        # It should be similar to reset the environment
        qpos = self.np_random.uniform(low=-2 * np.pi, high=2 * np.pi, size=self.model.nq)
        qpos[-2:] = self.goal_state
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self.goal_observation = self.render(mode='rgb_array', depth=False)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal_state
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        '''
        '''
        return self.set_to_goal(goal)

    def get_image(self, width=84, height=84, camera_name=None):
        assert width == height
        return self.render(mode='rgb_array', image_size=width, depth=False)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        # for stat_name in [
        #     'touch_distance',
        #     'hand_success',
        #     'obj_success',
        #     'hand_and_obj_success',
        #     'touch_success',
        #     'hand_distance',
        #     'obj_distance',
        #     'hand_and_obj_distance',
        #     'total_pickups',
        # ]: # these are copied from saywer_pickup_and_place.py
        for stat_name in [
            'ag_state', 'g_state', 'image_dist', 'image_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def _set_env_state(self, state):
        ''' According to multiworld, there is a base class. 
            But this is roughly already the base class along the inheritance chain.
            I put the implementation here.
        '''
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        # mocap_pos, mocap_quat = mocap_state
        # self.data.set_mocap_pos('body0', mocap_pos) # It seems it has no mocap in use
        # self.data.set_mocap_quat('body0', mocap_quat)
        self.sim.forward()

# End   Adding interface for multiworld environment collection
