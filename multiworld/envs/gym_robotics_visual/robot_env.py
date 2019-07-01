import os
import copy
import numpy as np
import glfw
import cv2 as cv
from numpy.random import random

import gym
from gym import error, spaces
from gym.utils import seeding
from mujoco_py.generated import const

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, image_size=100, horizon=100, visualization_mode=False):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self.obs_viewer = None
        self.image_size = image_size
        # Used to set the context buffer to address the image blending issue
        self.sim.render(camera_name='external_camera_0', width=self.image_size, height=self.image_size, depth=True)
        self.set_camera_location(camera_name='external_camera_0', pos=[1.4, 0.0, 0.9])
        self.set_camera_fov(camera_name='external_camera_0', fovy=50.0)
        self.set_camera_orientation(camera_name='external_camera_0',
                                    orientation_quat=[5.73665797e-01, 4.13561486e-01, 4.13890947e-01, 5.73209154e-01])

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self._max_episode_steps = horizon
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()

        self.visual_goal = None
        obs = self.reset()

        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self.goal_dim = np.prod(obs['achieved_goal'].shape)
        self.goal_state_dim = np.prod(self.goal.shape)

    def set_camera_location(self, camera_name=None, pos=[0.0, 0.0, 0.0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_pos[id] = pos

    def set_camera_fov(self, camera_name=None, fovy=50.0):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_fovy[id] = fovy

    def set_camera_orientation(self, camera_name=None, orientation_quat=[0, 0, 0, 0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_quat[id] = orientation_quat

    def set_camera_location(self, camera_name=None, pos=[0.0, 0.0, 0.0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_pos[id] = pos

    def set_camera_fov(self, camera_name=None, fovy=50.0):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_fovy[id] = fovy

    def set_camera_orientation(self, camera_name=None, orientation_quat=[0., 0., 0., 0.]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_quat[id] = orientation_quat

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_image_transformation(self, image, max_translation=10, max_angle=30):
        angle = np.random.uniform(-max_angle, max_angle)
        translation_x = np.random.uniform(-max_translation, max_translation)
        translation_y = np.random.uniform(-max_translation, max_translation)

        width = image.shape[1]
        height = image.shape[0]
        M1 = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        M2 = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

        transformed_img = cv.warpAffine(image, M1 + M2, (image.shape[1], image.shape[0]))
        return transformed_img, np.asarray([angle, translation_x, translation_y])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        transformed_img, transformation = self.random_image_transformation(next_frame)
        done = False

        info = {}
        # # TODO: Merge this
        # assert False
        # info = {
        #
        #     'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
        #     'prev_frame': prev_frame.flatten(),
        #     'next_frame': next_frame.flatten(),
        #     'action_taken': action,
        #     'transformed_frame': transformed_img.flatten(),
        #     'transformation': transformation
        # }
        _info = self.get_current_info()
        info.update(_info)

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        return obs, reward, done, info

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        raise NotImplementedError

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.

        did_reset_sim = False
        self.goal = self._sample_goal().copy()
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()

        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def render(self, mode='human', image_size=500, depth=True):
        self._render_callback()
        if mode == 'rgb_array':

            self.sim.render_contexts[0]._set_mujoco_buffers()

            if depth:
                image, depth = self.sim.render(camera_name='external_camera_0', width=image_size, height=image_size,
                                               depth=True)
                rgbd_data = np.dstack([image, depth])
                return rgbd_data[::-1, :, :]
            else:
                image = self.sim.render(camera_name='external_camera_0', width=image_size, height=image_size,
                                        depth=False)
                return image[::-1, :, :]

                # self._get_viewer().render()
                # width, height = glfw.get_framebuffer_size(self._get_viewer().window)
                # data = self._get_viewer().read_pixels(width, height, depth=False)
                # # original image is upside-down, so flip it
                # return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def get_image_obs(self, depth=True, hide_overlay=True, camera_id=-1):
        '''
       
        # Choose camera
        viewer = self._get_obs_viewer()
        viewer._hide_overlay = hide_overlay
        if camera_id == -1:
            viewer.cam.fixedcamid = camera_id
            viewer.cam.type = const.CAMERA_FREE
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 179.89
            viewer.cam.elevation = -47.03
        else:
            viewer.cam.fixedcamid = camera_id
            viewer.cam.type = const.CAMERA_FIXED

        # Render
        '''
        data = self.render(mode='rgb_array', image_size=self.image_size, depth=depth)
        return data
        # width, height = glfw.get_framebuffer_size(viewer.window)

    #         # original image is upside-down, so flip it
    #         if depth:
    #             rgb_data, depth_data = viewer.read_pixels(width, height, depth=True)
    #             rgbd_data = np.dstack([rgb_data, depth_data])
    #             return rgbd_data[::-1, :, :]
    #         else:
    #             rgb_data = viewer.read_pixels(width, height, depth=False)
    #             return rgb_data[::-1, :, :]

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    # def _get_obs_viewer(self):
    #     if self.obs_viewer is None:
    #         self.obs_viewer = mujoco_py.MjViewer(self.sim)
    #         self._obs_viewer_setup()
    #     return self.obs_viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    # def _is_success(self, achieved_goal, desired_goal):
    #     """Indicates whether or not the achieved goal successfully achieved the desired goal.
    #     """
    #     raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _get_info_state(self, achieved_goal, desired_goal):
        """Given g, ag in state space and return the distance and success
        """
        raise NotImplementedError

    def _get_info_obs(self, achieved_goal_obs, desired_goal_obs):
        """Given g, ag in state space and return the distance and success
        """
        raise NotImplementedError
