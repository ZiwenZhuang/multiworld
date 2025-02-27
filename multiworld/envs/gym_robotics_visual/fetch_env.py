import numpy as np
import random 
from multiworld.envs.gym_robotics_visual import rotations, robot_env, utils


# Note: may not generalize to goals in the form of matrix
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
      self, model_path, n_substeps, gripper_extra_height, block_gripper,
      has_object, target_in_the_air, target_offset, obj_range, target_range,
      distance_threshold, initial_qpos, reward_type, object_location_same_as_gripper_probability = 1.0, 
      use_true_reward=False,
      use_visual_observation=True,
      use_image_goal=True,
      distance_threshold_obs=0,
      **kwargs
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.distance_threshold_obs = distance_threshold_obs
        if (not use_visual_observation and distance_threshold == 0. and not use_true_reward) or (
            use_visual_observation and distance_threshold_obs == 0. and not use_true_reward):
            self.compute_reward = self.compute_reward_zero
        self.reward_type = reward_type
        self.object_location_same_as_gripper_probability = object_location_same_as_gripper_probability
        self.use_true_reward = use_true_reward
        self.use_visual_observation = use_visual_observation
        self.use_image_goal = use_image_goal
        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos, **kwargs)
    # GoalEnv methods
    # ----------------------------

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
            d_threshold = self.distance_threshold_obs
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
        # print(achieved_goal[:5, :])
        # print(desired_goal[:5, :])
        # print((np.alltrue(np.equal(achieved_goal, desired_goal), axis=-1) - 1.)[:5])
        # print('---')
        return np.alltrue(np.equal(achieved_goal, desired_goal), axis=-1) - 1.

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        info = self.get_current_info()
        if self.use_visual_observation:
            obs = self.get_image_obs(depth=True)
        else:
            obs = info['obs_state']

        if self.use_image_goal:
            assert self.use_visual_observation
            ag = obs.copy()
            g = self.visual_goal
        else:
            ag = info['ag_state']
            g = info['g_state']
        return {
            'observation': obs.copy(),
            'achieved_goal': ag.copy(),
            'desired_goal': g.copy(),
        }

    def get_current_info(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        return {
            'obs_state': obs.copy(),
            'ag_state': achieved_goal.copy(),
            'g_state': self.goal.copy(),
        }


    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    # def _obs_viewer_setup(self):
    #     body_id = self.sim.model.body_name2id('robot0:gripper_link')
    #     lookat = self.sim.data.body_xpos[body_id]
    #     for idx, value in enumerate(lookat):
    #         self.obs_viewer.cam.lookat[idx] = value
    #     self.obs_viewer.cam.distance = 2.5
    #     self.obs_viewer.cam.azimuth = 132.
    #     self.obs_viewer.cam.elevation = -14.

    # TODO what is this
    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        # Randomize start position of object.
        self.sim.set_state(self.initial_state)
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                     size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            # First render the goal position then the initial observation
            object_qpos[:2] = self.goal[:2]
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.sim.forward()
            self.visual_goal = self.get_image_obs(depth=True)
            # TODO: figure out why the first rendering does not have effect
            if random.random() < self.object_location_same_as_gripper_probability :
                object_qpos[:3] = self.sim.data.get_site_xpos('robot0:grip') # making start position of object same as gripper to limit exploration
            else: 
                # TODO use the set_state function
                object_qpos[:2] = object_xpos # randomly initialized object location
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            # TODO: render visual goal for the case of no object!
        else:
            goal_gripper_target = self.goal
            goal_gripper_rotation = np.array([1., 0., 1., 0.])
            self._move_gripper(goal_gripper_target, goal_gripper_rotation)
            self.visual_goal = self.get_image_obs(depth=True)
            # Set back
            self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    # def _is_success(self, achieved_goal, desired_goal):
    #     d = goal_distance(achieved_goal, desired_goal)
    #     return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos(
            'robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self._move_gripper(gripper_target=gripper_target, gripper_rotation=gripper_rotation)
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _move_gripper(self, gripper_target, gripper_rotation):
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def _get_info_state(self, achieved_goal, desired_goal):
        """Given g, ag in state space and return the distance and success
        """
        achieved_goal = achieved_goal.reshape([-1, self.goal_state_dim])
        desired_goal = desired_goal.reshape([-1, self.goal_state_dim])
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return d, (d <=self.distance_threshold).astype(np.float32)

    def _get_info_obs(self, achieved_goal_obs, desired_goal_obs):
        """Given g, ag in state space and return the distance and success
        """
        achieved_goal_obs = achieved_goal_obs.reshape([-1, self.goal_dim])
        desired_goal_obs = desired_goal_obs.reshape([-1, self.goal_dim])
        d = np.linalg.norm(achieved_goal_obs - desired_goal_obs, axis=-1)
        return d, (d <= self.distance_threshold_obs).astype(np.float32)
