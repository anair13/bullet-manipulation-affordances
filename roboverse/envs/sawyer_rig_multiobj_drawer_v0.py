import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.sawyer_base import SawyerBaseEnv
from roboverse.bullet.misc import load_obj, deg_to_quat, quat_to_deg, get_bbox
from bullet_objects import loader, metadata
import os.path as osp
import importlib.util
import random
import pickle
import gym
from roboverse.bullet.drawer_utils import *
from roboverse.bullet.button_utils import *

test_set = ['mug', 'long_sofa', 'camera', 'grill_trash_can', 'beer_bottle']

quat_dict={'mug': [0, -1, 0, 1],'long_sofa': [0, 0, 0, 1],'camera': [-1, 0, 0, 0],
        'grill_trash_can': [0, 0, 1, 1], 'beer_bottle': [0, 0, 1, -1]}

class SawyerRigMultiobjDrawerV0(SawyerBaseEnv):

    def __init__(self,
                 goal_pos=(0.75, 0.2, -0.1),
                 reward_type='shaped',
                 reward_min=-2.5,
                 randomize=False,
                 observation_mode='state',
                 obs_img_dim=48,
                 success_threshold=0.08,
                 transpose_image=False,
                 invisible_robot=False,
                 object_subset='test',
                 use_bounding_box=True,
                 random_color_p=0.5,
                 quat_dict=quat_dict,
                 task='goal_reaching',
                 DoF=3,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param goal_pos: xyz coordinate of desired goal
        :param reward_type: one of 'shaped', 'sparse'
        :param reward_min: minimum possible reward per timestep
        :param randomize: whether to randomize the object position or not
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        :param invisible_robot: the robot arm is invisible when set to True
        """
        assert DoF in [3, 4, 6]
        assert task in ['goal_reaching', 'pickup']
        print("Task Type: " + task)

        is_set = object_subset in ['test', 'train', 'all']
        is_list = type(object_subset) == list
        assert is_set or is_list

        self.goal_pos = np.asarray(goal_pos)
        self.quat_dict = quat_dict
        self._reward_type = reward_type
        self._reward_min = reward_min
        self._randomize = randomize
        self.pickup_eps = -0.3
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.random_color_p = random_color_p
        self.use_bounding_box = use_bounding_box
        self.object_subset = object_subset
        self._ddeg_scale = 5
        self.task = task
        self.DoF = DoF

        self.object_dict, self.scaling = self.get_object_info()
        self.curr_object = None
        self._object_position_low = (0.55,-0.18,-.36)
        self._object_position_high = (0.85,0.18,-0.15)
        self._goal_low = np.array([0.55,-0.18,-.11])
        self._goal_high = np.array([0.8,0.18,-0.11])
        self._fixed_object_position = np.array([.82, -0.125, -.25])
        self._fixed_lego_position = np.array([.775, 0.125, -.25])
        self.start_obj_ind = 4 if (self.DoF == 3) else 8
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self._success_threshold = success_threshold
        self.obs_img_dim = obs_img_dim #+.15
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.25], distance=0.425,
            yaw=90, pitch=-42, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1
        super().__init__(*args, **kwargs)

    def get_object_info(self):
        complete_object_dict, scaling = metadata.obj_path_map, metadata.path_scaling_map
        complete = self.object_subset is None
        train = (self.object_subset == 'train') or (self.object_subset == 'all')
        test = (self.object_subset == 'test') or (self.object_subset == 'all')

        object_dict = {}
        for k in complete_object_dict.keys():
            in_test = (k in test_set)
            in_subset = (k in self.object_subset)
            if in_subset:
                object_dict[k] = complete_object_dict[k]
            if complete:
                object_dict[k] = complete_object_dict[k]
            if train and not in_test:
                object_dict[k] = complete_object_dict[k]
            if test and in_test:
                object_dict[k] = complete_object_dict[k]
        return object_dict, scaling


    def _set_spaces(self):
        act_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 13
        if self.DoF > 3:
            # Add wrist theta
            observation_dim += 4

        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        self.observation_space = Dict([
            ('observation', state_space),
            ('state_observation', state_space),
            ('desired_goal', state_space),
            ('state_desired_goal', state_space),
            ('achieved_goal', state_space),
            ('state_achieved_goal', state_space),
        ])

    def _load_table(self):
        self._objects = {}
        self._sensors = {}

        self._sawyer = bullet.objects.drawer_sawyer()
        self._table = bullet.objects.table()
        self._top_drawer = bullet.objects.drawer()
        self._bottom_drawer = bullet.objects.drawer_no_handle()

        self._objects['button'] = bullet.objects.button()
        self._objects['lego'] = bullet.objects.drawer_lego()
        self.tray = bullet.objects.drawer_tray()
        self.init_button_height = get_button_cylinder_pos(self._objects['button'])[2]
        self.drawer_opened = False

        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')


    def sample_object_location(self):
        if self._randomize:
            return np.random.uniform(
                low=self._object_position_low,
                high=self._object_position_high)
        return self._fixed_object_position

    def sample_object_color(self):
        if np.random.uniform() < self.random_color_p:
            return list(np.random.choice(range(256), size=3) / 255.0) + [1]
        return None

    def sample_quat(self, object_name):
        if object_name in self.quat_dict:
            return self.quat_dict[self.curr_object]
        return deg_to_quat(np.random.randint(0, 360, size=3))

    def add_object(self, change_object=True, object_position=None, quat=None):
        # Pick object if necessary and save information
        if change_object:
            self.curr_object, self.curr_id = random.choice(list(self.object_dict.items()))
            self.curr_color = self.sample_object_color()

        # Generate random object position
        if object_position is None:
            object_position = self.sample_object_location()

        # Generate quaterion if none is given
        if quat is None:
            quat = self.sample_quat(self.curr_object)

        # Spawn object above table
        self._objects['obj'] = loader.load_shapenet_object(
                self.curr_id,
                self.scaling,
                object_position,
                quat=quat,
                rgba=self.curr_color)

        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1)
        for _ in range(100):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(100):
            bullet.step()

    def _format_action(self, *action):
        if self.DoF == 3:
            if len(action) == 1:
                delta_pos, gripper = action[0][:-1], action[0][-1]
            elif len(action) == 2:
                delta_pos, gripper = action[0], action[1]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), gripper
        elif self.DoF == 4:
            if len(action) == 1:
                delta_pos, delta_yaw, gripper = action[0][:3], action[0][3:4], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            
            delta_angle = [0, 0, delta_yaw[0]]
            return np.array(delta_pos), np.array(delta_angle), gripper
        else:
            if len(action) == 1:
                delta_pos, delta_angle, gripper = action[0][:3], action[0][3:6], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_angle, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), np.array(delta_angle), gripper

    def button_pressed(self):
        curr_height = get_button_cylinder_pos(self._objects['button'])[2]
        pressed = (self.init_button_height - curr_height) > 0.01
        if pressed and not self.drawer_opened:
            self.drawer_opened = True
            return True
        return False

    def check_obj_bounding_box(self, obj):
        object_pos = bullet.get_body_info(obj)['pos']
        adjustment = np.array([0.045, 0.04, 0.15])
        low = np.array(self._pos_low) - adjustment
        high = np.array(self._pos_high) + adjustment
        contained = (object_pos > low).all() and (object_pos < high).all()
        return contained

    def enforce_bounding_box(self):
        contained_obj = self.check_obj_bounding_box(self._objects['obj'])
        contained_lego = self.check_obj_bounding_box(self._objects['lego'])
        
        if not contained_obj or not contained_lego:
            bullet.position_control(self._sawyer, self._end_effector,
                np.array(self._pos_init), self.default_theta)
            for i in range(3): self._simulate(np.array(self._pos_init), self.default_theta, -1)

        if not contained_obj:
            p.removeBody(self._objects['obj'])
            self.add_object(change_object=False)

        if not contained_lego:
            p.removeBody(self._objects['lego'])
            obj_pos = self._fixed_lego_position
            self._objects['lego'] = bullet.objects.drawer_lego(pos=obj_pos)

    def step(self, *action):
        # Get positional information
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        curr_angle = bullet.get_link_state(self._sawyer, self._end_effector, 'theta')
        default_angle = quat_to_deg(self.default_theta)
    
        # Keep necesary degrees of theta fixed
        if self.DoF == 3:
            angle = default_angle
        elif self.DoF == 4:
            angle = np.append(default_angle[:2], [curr_angle[2]])
        else:
            angle = curr_angle

        # If angle is part of action, use it
        if self.DoF == 3:
            delta_pos, gripper = self._format_action(*action)
        else:
            delta_pos, delta_angle, gripper = self._format_action(*action)
            angle += delta_angle * self._ddeg_scale

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle)
        self._simulate(pos, theta, gripper)

        # Open box if button is pressed
        if self.button_pressed():
            open_drawer(self._bottom_drawer)

        # Reset if bounding box is violated
        if self.use_bounding_box:
            self.enforce_bounding_box()

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return observation, reward, done, info

    def get_info(self):
        object_pos = np.asarray(self.get_object_midpoint('obj'))
        height = object_pos[2]
        object_goal_distance = np.linalg.norm(object_pos - self.goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(
            object_pos - end_effector_pos)
        gripper_goal_distance = np.linalg.norm(
            self.goal_pos - end_effector_pos)
        object_goal_success = int(object_goal_distance < self._success_threshold)
        picked_up = height > self.pickup_eps

        info = {
            'object_goal_distance': object_goal_distance,
            'object_goal_success': object_goal_success,
            'object_height': height,
            'picked_up': picked_up,
        }

        return info

    def get_contextual_diagnostics(self, paths, contexts):
        from multiworld.envs.env_util import create_stats_ordered_dict
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        eps1, eps2 = [], []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key][self.start_obj_ind:self.start_obj_ind + 3]
            goal = contexts[i][goal_key][self.start_obj_ind:self.start_obj_ind + 3]
            distance = np.linalg.norm(state - goal)

            if self.task == 'pickup':
                values.append(state[2] > self.pickup_eps)
            if self.task == 'goal_reaching':
                values.append(distance)
                eps1.append(distance < 0.05)
                eps2.append(distance < 0.08)

        if self.task == 'pickup':
            diagnostics_key = goal_key + "/final/picked_up"
        if self.task == 'goal_reaching':
            diagnostics_key = goal_key + "/final/distance"
            diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success_0.05", eps1))
            diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success_0.08", eps2))
        diagnostics.update(create_stats_ordered_dict(diagnostics_key, values))

        values = []
        eps1, eps2 = [], []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key][self.start_obj_ind:self.start_obj_ind + 3]
                goal = contexts[i][goal_key][self.start_obj_ind:self.start_obj_ind + 3]
                distance = np.linalg.norm(state - goal)
                
                if self.task == 'pickup':
                    values.append(state[2] > self.pickup_eps)
                if self.task == 'goal_reaching':
                    values.append(distance)
                    eps1.append(distance < 0.05)
                    eps2.append(distance < 0.08)
        
        if self.task == 'pickup':
            diagnostics_key = goal_key + "/picked_up"
        if self.task == 'goal_reaching':
            diagnostics_key = goal_key + "/distance"
            diagnostics.update(create_stats_ordered_dict(goal_key + "/success_0.05", eps1))
            diagnostics.update(create_stats_ordered_dict(goal_key + "/success_0.08", eps2))

        diagnostics.update(create_stats_ordered_dict(diagnostics_key, values))
        return diagnostics

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def set_goal(self, goal):
        self.goal_pos = goal['state_desired_goal'][self.start_obj_ind:self.start_obj_ind + 3]

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info):
        if self.task == 'goal_reaching':
            return info['object_goal_success'] - 1
        elif self.task == 'pickup':
            return info['picked_up'] - 1

    def sample_goals(self):
        self.obj_goal = np.random.uniform(low=self._goal_low, high=self._goal_high)
        self.lego_goal = np.random.uniform(low=self._goal_low, high=self._goal_high)
        self.hand_goal = np.random.uniform(low=self._goal_low, high=self._goal_high)

        ld_pos = self.get_object_pos('bottom_drawer') 
        self.bd_goal = np.random.uniform(low=ld_pos, high=ld_pos + np.array([0.155, 0, 0]))

        td_pos = self.get_object_pos('drawer_handle') 
        self.td_goal = np.random.uniform(low=(td_pos - np.array([0, 0.18, 0])), high=td_pos)

    def reset(self, change_object=True):
        # Load Enviorment
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_table()
        self.add_object(change_object=change_object)
        self._format_state_query()

        # Sample and load starting positions
        init_pos = np.array(self._pos_init)
        self.goal_pos = np.random.uniform(low=self._goal_low, high=self._goal_high)
        self.sample_goals()

        bullet.position_control(self._sawyer, self._end_effector, init_pos, self.default_theta)

        # Move to starting positions
        action = np.array([0 for i in range(self.DoF)] + [-1])
        for _ in range(3):
            self.step(action)
        return self.get_observation()

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def compute_reward_pu(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        height = obj_state[:, 2]
        reward = (height > self.pickup_eps) - 1
        return reward
    
    def compute_reward_gr(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        object_goal_success = object_goal_distance < self._success_threshold
        return object_goal_success - 1

    def compute_reward(self, obs, actions, next_obs, contexts):
        if self.task == 'goal_reaching':
            return self.compute_reward_gr(obs, actions, next_obs, contexts)
        elif self.task == 'pickup':
            return self.compute_reward_pu(obs, actions, next_obs, contexts)

    def get_object_pos(self, obj_name):
        if obj_name in ['obj', 'lego']:
            return np.array(bullet.get_body_info(self._objects[obj_name], quat_to_deg=False)['pos'])
        elif obj_name == 'button':
            return np.array(get_button_cylinder_pos(self._objects['button']))
        elif obj_name == 'bottom_drawer':
            return np.array(get_drawer_bottom_pos(self._bottom_drawer))
        elif obj_name == 'drawer_handle':
            return np.array(get_drawer_handle_pos(self._top_drawer))
        else:
            return 1/0

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_l_finger_joint', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._sawyer, 'right_gripper_r_finger_joint', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        hand_theta = bullet.get_link_state(self._sawyer, self._end_effector,
            'theta', quat_to_deg=False)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        object_info = bullet.get_body_info(self._objects['obj'],
                                           quat_to_deg=False)
        object_pos = object_info['pos']
        object_theta = object_info['theta']

        lego_info = bullet.get_body_info(self._objects['obj'],
                                           quat_to_deg=False)
        lego_pos = lego_info['pos']

        bottom_drawer_pos = [get_drawer_bottom_pos(self._bottom_drawer)[0]]
        top_drawer_pos = [get_drawer_handle_pos(self._top_drawer)[1]]
        button_height = [get_button_cylinder_pos(self._objects['button'])[2]]

        if self.DoF > 3:
            #(hand_pos, hand_theta, gripper, obj_pos, lego_pos, bd_pos, td_pos, b_pos)
            observation = np.concatenate((
                end_effector_pos, hand_theta, gripper_tips_distance,
                object_pos, lego_pos, bottom_drawer_pos, top_drawer_pos, button_height))
            goal_pos = np.concatenate((
                self.goal_pos, hand_theta, gripper_tips_distance,
                self.goal_pos, self.goal_pos, self.goal_pos))
        else:
            #(hand_pos, gripper, obj_pos, lego_pos, bd_pos, td_pos, b_pos)
            observation = np.concatenate((
                end_effector_pos, gripper_tips_distance, object_pos,
                lego_pos, bottom_drawer_pos, top_drawer_pos, button_height))
            goal_pos = np.concatenate((
                self.goal_pos, gripper_tips_distance, self.goal_pos,
                self.goal_pos, self.goal_pos))

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=goal_pos,
            state_desired_goal=goal_pos,
            achieved_goal=observation,
            state_achieved_goal=observation,
            )

        return obs_dict