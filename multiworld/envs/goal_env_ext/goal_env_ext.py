# Created by Xingyu Lin, 30/08/2018                                                                                  
import os
from os import path
import numpy as np

import gym
from gym import GoalEnv
from gym import error, spaces
from gym.utils import seeding

import mujoco_py
from mujoco_py import load_model_from_path, MjSim, MjViewer

import cv2 as cv
import copy


class GoalEnvExt(GoalEnv):
    def __init__(self, model_path, n_substeps, n_actions, horizon, image_size, use_image_goal,
                 use_visual_observation, with_goal,
                 reward_type, distance_threshold, distance_threshold_obs, use_true_reward,
                 initial_qpos=None, default_camera_name='external_camera_0', use_auxiliary_loss=False,
                 noisy_reward_fp=None, noisy_reward_fn=None, state_estimation_reward_noise=0., **kwargs):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "./assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model, nsubsteps=n_substeps)

        self.data = self.sim.data
        self.viewer = None
        self.np_random = None
        self.seed()

        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.horizon = horizon
        self.image_size = image_size
        self.use_image_goal = use_image_goal
        self.use_visual_observation = use_visual_observation
        self.with_goal = with_goal

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.distance_threshold_obs = distance_threshold_obs
        self.use_true_reward = use_true_reward

        self.state_estimation_reward_noise = state_estimation_reward_noise

        self._max_episode_steps = horizon
        self.time_step = 0

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.goal_state = self.goal_observation = None
        if noisy_reward_fn is None and noisy_reward_fp is None:
            if (not use_visual_observation and not use_true_reward) or (
              use_visual_observation and distance_threshold_obs == 0. and not use_true_reward):
                self.compute_reward = self.compute_reward_zero
        else:
            self.compute_reward = self.compute_reward_noisy
            self.noisy_reward_fp = noisy_reward_fp
            self.noisy_reward_fn = noisy_reward_fn

        self.default_camera_name = default_camera_name
        self._set_camera()
        self.sim.render(camera_name=default_camera_name, width=self.image_size, height=self.image_size, depth=False,
                        mode='offscreen')

        self.use_auxiliary_loss = use_auxiliary_loss
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        obs = self.reset()

        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.goal_dim = np.prod(obs['achieved_goal'].shape)
        self.goal_state_dim = np.prod(self.goal_state.shape)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Compute distance between goal and the achieved goal. (only one of them)
        if self.use_true_reward:
            achieved_goal = info['ag_state']
            desired_goal = info['g_state']
            achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
            d_threshold = self.distance_threshold
        else:
            achieved_goal = achieved_goal.reshape([-1, self.goal_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_dim])

        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > d_threshold).astype(np.float32)
        else:
            return -d

    def compute_reward_zero(self, achieved_goal, desired_goal, info):
        if self.use_true_reward:
            achieved_goal = info['ag_state']
            desired_goal = info['g_state']
            achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        else:
            achieved_goal = achieved_goal.reshape([-1, self.goal_dim])
            desired_goal = desired_goal.reshape([-1, self.goal_dim])
        assert achieved_goal.shape == desired_goal.shape
        return np.alltrue(np.equal(achieved_goal, desired_goal), axis=-1) - 1.

# Start code to meet the RIG framework interface

    def compute_rewards(self, actions, obs):
        ''' @brief: both 'action' and 'obs' are a batch of data. (The batch size could be 1)
        '''
        # Considering the obs is a dictionary, we will not check batch size here (I think I can't)
        rewards = []
        info = self.get_current_info()
        for i in range(actions.shape[0]):
            act = actions[i]
            obser = {
                k: v[i] for k, v in obs.items()
            }
            rewards.append(self.compute_reward(obser['achieved_goal'], obser['desired_goal'], info))

        return np.array(rewards)


# End code to meet the RIG framework interface

    def compute_reward_noisy(self, achieved_goal, desired_goal, info):
        achieved_goal = info['ag_state']
        desired_goal = info['g_state']
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d_threshold = self.distance_threshold
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        bool_rewards = d <= d_threshold
        if self.noisy_reward_fp is not None:
            neg_idx = np.where(bool_rewards == False)[0]
            fp_idx = np.where(np.random.random(size=len(neg_idx)) < self.noisy_reward_fp)[0]
            bool_rewards[neg_idx[fp_idx]] = True
            return bool_rewards.astype(np.float32) - 1.
        elif self.noisy_reward_fn is not None:
            pos_idx = np.where(bool_rewards == True)[0]
            fn_idx = np.where(np.random.random(size=len(pos_idx)) < self.noisy_reward_fn)[0]
            bool_rewards[pos_idx[fn_idx]] = False
            return bool_rewards.astype(np.float32) - 1.
        else:
            raise NotImplementedError

    # This is for test purpose only
    # def compute_reward_noisy(self, achieved_goal, desired_goal, info, bool_rewards=None):
    #     if bool_rewards is None:
    #         achieved_goal = info['ag_state']
    #         desired_goal = info['g_state']
    #         achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
    #         desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
    #         d_threshold = self.distance_threshold
    #         d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    #         bool_rewards = d <= d_threshold
    #     else:
    #         bool_rewards = bool_rewards.copy()
    #     if self.noisy_reward_fp is not None:
    #         neg_idx = np.where(bool_rewards == False)[0]
    #         # print('neg_idx', neg_idx)
    #         fp_idx = np.where(np.random.random(size=len(neg_idx)) < self.noisy_reward_fp)[0]
    #         # print('fp_idx', fp_idx)
    #         bool_rewards[neg_idx[fp_idx]] = True
    #         # print('bool_noisy_rewards', bool_rewards)
    #         return bool_rewards.astype(np.float32) - 1.
    #     elif self.noisy_reward_fn is not None:
    #         pos_idx = np.where(bool_rewards == True)[0]
    #         fn_idx = np.where(np.random.random(size=len(pos_idx)) < self.noisy_reward_fn)[0]
    #         bool_rewards[pos_idx[fn_idx]] = False
    #         return bool_rewards.astype(np.float32) - 1.
    #     else:
    #         raise NotImplementedError

    # methods to override:
    # ----------------------------
    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        raise NotImplementedError

    def _get_obs(self):
        """
        Get observation
        """
        raise NotImplementedError

    def _set_action(self, ctrl):
        """
        Do simulation
        """
        raise NotImplementedError

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _set_camera(self):
        pass

    def get_current_info(self):
        """
        :return: The true current state, 'ag_state', and goal state, 'g_state'
        """
        raise NotImplementedError

    def _viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def set_hidden_goal(self):
        """
        Hide the goal position from image observation
        """
        pass

    def get_image_obs(self, depth=True, hide_overlay=True, camera_id=-1):
        assert False
        return

    def _sample_goal_state(self):
        """Samples a new goal in state space and returns it.
        """
        return None

    # Core functions framework
    # -----------------------------

    def reset(self):
        '''
        Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.
        '''
        self.time_step = 0
        if not self.with_goal:
            self.set_hidden_goal()

        goal_state = self._sample_goal_state()
        if goal_state is None:
            self.goal_state = None
        else:
            self.goal_state = goal_state.copy()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.use_auxiliary_loss:
            assert self.use_visual_observation
            self._set_action(action)
            self.sim.step()
            self._step_callback()
            obs = self._get_obs()
            # transformed_img, transformation = self.random_image_transformation(next_frame)
            # TODO for now, comment out all other auxiliary losses except action prediction
            aug_info = {
                'action_taken': action,
                # 'transformed_frame': transformed_img.flatten(),
                # 'transformation': transformation
            }
        else:
            self._set_action(action)
            self.sim.step()
            self._step_callback()
            obs = self._get_obs()
            aug_info = {}
        state_info = self.get_current_info()
        info = {**aug_info, **state_info}

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        self.time_step += 1
        # Episode ends only when the horizon is reached
        done = False
        if self.time_step >= self.horizon:
            done = True
        return obs, reward, done, info

    def get_initial_info(self):
        state_info = self.get_current_info()

        if self.use_auxiliary_loss:
            aug_info = {
                'action_taken': np.zeros(self.action_space.shape),
                # 'transformed_frame': transformed_img.flatten(),
                # 'transformation': transformation
            }
            return {**aug_info, **state_info}
        else:
            return state_info

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    def render(self, mode='human', image_size=None, depth=True, camera_name=None):
        self._render_callback()
        if camera_name is None:
            camera_name = self.default_camera_name
        if image_size is None:
            image_size = self.image_size
        if mode == 'rgb_array':
            # could be a bug code
            # self.sim.render_contexts[0]._set_mujoco_buffers()
            if depth:
                image, depth = self.sim.render(camera_name=camera_name, width=image_size, height=image_size, depth=True)
                # id = self.sim.model.camera_name2id('external_camera_0')
                # print(self.sim.model.cam_fovy)
                rgbd_data = np.dstack([image, depth])
                return rgbd_data[::-1, :, :]
            else:
                image = self.sim.render(camera_name=camera_name, width=image_size, height=image_size, depth=False)
                return image[::-1, :, :]
        elif mode == 'human':
            return self._get_viewer().render()

    # Auxiliary Reward Methods
    # ----------------------------
    @staticmethod
    def random_image_transformation(image, max_translation=10, max_angle=30):
        angle = np.random.uniform(-max_angle, max_angle)
        translation_x = np.random.uniform(-max_translation, max_translation)
        translation_y = np.random.uniform(-max_translation, max_translation)

        width = image.shape[1]
        height = image.shape[0]
        M1 = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

        M2 = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

        transformed_img = cv.warpAffine(image, M1 + M2, (image.shape[1], image.shape[0]))
        return transformed_img, np.asarray([angle, translation_x, translation_y])

    # Helper Functions
    # ----------------------------
    def _get_info_state(self, achieved_goal, desired_goal):
        # Given g, ag in state space and return the distance and success
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return d, (d <= self.distance_threshold).astype(np.float32)

    def _get_info_obs(self, achieved_goal_obs, desired_goal_obs):
        # Given g, ag in state space and return the distance and success
        achieved_goal_obs = achieved_goal_obs.reshape([-1, self.goal_dim])
        desired_goal_obs = desired_goal_obs.reshape([-1, self.goal_dim])
        d = np.linalg.norm(achieved_goal_obs - desired_goal_obs, axis=-1)
        return d, (d <= self.distance_threshold_obs).astype(np.float32)

    def set_camera_location(self, camera_name=None, pos=[0.0, 0.0, 0.0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_pos[id] = pos

    def set_camera_fov(self, camera_name=None, fovy=50.0):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_fovy[id] = fovy

    def set_camera_orientation(self, camera_name=None, orientation_quat=[0, 0, 0, 0]):
        id = self.sim.model.camera_name2id(camera_name)
        self.sim.model.cam_quat[id] = orientation_quat

# Start Adding interface for multiworld environment collection

    def initialize_camera(self, init_fctn):
        # sim = self.sim
        # # viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=self.device_id)
        # viewer = mujoco_py.MjViewer(sim)
        # init_fctn(viewer.cam)
        # sim.add_render_context(viewer)
        pass

    def _get_env_state(self):
        ''' According to multiworld, there is a base class. 
            But this is roughly already the base class along the inheritance chain.
            I put the implementation here.
        '''
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)
    
    def get_env_state(self):
        base_state = self._get_env_state()
        goal = self.goal_state.copy()
        return base_state, goal

    def _set_env_state(self, state):
        raise NotImplementedError

    def set_env_state(self, state):
        base_state, goal = state
        self._set_env_state(base_state) # from the child class
        self.goal_state = goal
        self._reset_sim()

    def sample_goals(self, num_batches):
        ''' Return a dict with each batch of goals.
            The dict includes keys: 'desired_goal', 'state_desired_goal'
        '''
        desired_goal = []
        state_desired_goal = []
        for _ in range(num_batches):
            goal = self._sample_goal_state()
            desired_goal.append(goal)
            state_desired_goal.append(goal)

        # form it as a dict and return
        return dict(
            desired_goal= desired_goal,
            state_desired_goal= state_desired_goal,
        )

# End   Adding interface for multiworld environment collection
