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

affordance_dict= {'handle_drawer': False, 'button': False, 'drawer': False, 'rand_obj': False,
'tray': False, 'drawer_open': False, 'side_sign': -1.}

top_drawer_dict= {'handle_drawer': True, 'button': False, 'drawer': False, 'rand_obj': False,
'tray': False, 'drawer_open': False, 'side_sign': 1.}

bottom_drawer_dict= {'handle_drawer': False, 'button': True, 'drawer': True, 'rand_obj': False,
'tray': False, 'drawer_open': False, 'side_sign': 1.}

tray_dict= {'handle_drawer': False, 'button': False, 'drawer': False, 'rand_obj': True,
'tray': True, 'drawer_open': False, 'side_sign': 1.}

obj_dict= {'handle_drawer': False, 'button': False, 'drawer': False, 'rand_obj': True,
'tray': False, 'drawer_open': False, 'side_sign': 1.}

# Constants
td_close_pos = 0.0125
td_open_pos = 0.1687

class SawyerRigAffordancesV1(SawyerBaseEnv):

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
                 random_color_p=1.0,
                 spawn_prob=0.75,
                 quat_dict=quat_dict,
                 task='goal_reaching',
                 test_env=False,
                 env_type=None,
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
        assert env_type in ['top_drawer', 'bottom_drawer', 'tray', 'obj', None]

        is_set = object_subset in ['test', 'train', 'all']
        is_list = type(object_subset) == list
        assert is_set or is_list

        self.goal_pos = np.asarray(goal_pos)
        self.quat_dict = quat_dict
        self._reward_type = reward_type
        self._reward_min = reward_min
        self._randomize = randomize
        self.pickup_eps = -0.33
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.random_color_p = random_color_p
        self.use_bounding_box = use_bounding_box
        self.object_subset = object_subset
        self.spawn_prob = spawn_prob
        self._ddeg_scale = 5
        self.task = task
        self.DoF = DoF
        self.test_env = test_env
        self.env_type = env_type

        if self.test_env:
            self.random_color_p = 0.0
            self.object_subset = kwargs.pop('test_object_subset', ['grill_trash_can'])
            #self.object_subset = ['beer_bottle']

        self.obj_thresh = 0.08
        self.drawer_thresh = 0.065
        self.button_thresh = 0.008

        self.object_dict, self.scaling = self.get_object_info()
        self.curr_object = None
        self._object_position_low = (0.55,-0.18,-.36)
        self._object_position_high = (0.85,0.18,-0.15)
        self._goal_low = np.array([0.55,-0.18,-0.11])
        self._goal_high = np.array([0.8,0.18,-0.11])
        self._fixed_object_position = np.array([.8, -0.12, -.25])
        self._reset_lego_position = np.array([.775, 0.125, -.25])
        self.init_lego_pos = np.array([0.59, 0.125, -0.31])
        # self.start_obj_ind = 4 if (self.DoF == 3) else 8
        self.start_obj_ind = 10 if (env_type == 'bottom_drawer') else 11
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self._success_threshold = success_threshold
        self.obs_img_dim = obs_img_dim #+.15
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[.7, 0, -0.25], distance=0.425,
            yaw=90, pitch=-37, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1

        # Reset-free
        self.reset_interval = kwargs.pop('reset_interval', 10)
        self.reset_counter = self.reset_interval-1
        self.expl = kwargs.pop('expl', False)
        self.trajectory_done = False
        self.final_timestep = False
        self.tasks_done = 0
        self.tasks_done_names = []

        # Debug
        self.debug = kwargs.pop('debug', None)

        super().__init__(*args, **kwargs)

        # Need to overwrite in some cases, registration isnt working
        self._max_force = 100
        self._action_scale = 0.05
        self._pos_init = [0.6, -0.15, -0.2]
        self._pos_low = [0.5,-0.2,-.36]
        self._pos_high = [0.85,0.2,-0.1]

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

    def sample_enviorment(self):
        self.env_type == 'top_drawer'
        self.affordance_dict = top_drawer_dict.copy()

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
        self.sample_enviorment()

        s = self.affordance_dict['side_sign']
        self._pos_init = [0.6, s * -0.15, -0.1] #Initialize gripper slighter higher so it doesn't collide with open drawer
        self._reset_lego_position = np.array([.775, s * 0.125, -.25])
        self.init_lego_pos = np.array([0.59, s * 0.125, -0.31])

        self._sawyer = bullet.objects.drawer_sawyer()

        # if np.random.uniform() < self.random_color_p:
        #     color = [np.random.uniform() for i in range(3)] + [1]
        #     self._table = bullet.objects.table(rgba=color)
        # else:
        self._table = bullet.objects.table(rgba=[.92,.85,.7,1])

        # Drawer
        if self.affordance_dict['drawer']:
            self._bottom_drawer = bullet.objects.drawer_no_handle(
                    pos=np.array([0.6, s * 0.125, -.34]), rgba=self.sample_object_color())
            self._objects['lego'] = bullet.objects.drawer_lego(pos=self.init_lego_pos)
            if self.affordance_dict['drawer_open']:
                open_drawer(self._bottom_drawer)
            self.init_drawer_pos = get_drawer_bottom_pos(self._bottom_drawer)[0]

        # Handle drawer
        if self.affordance_dict['handle_drawer']:
            quat = [0, 0, 0, 1] if s == 1 else [0, 0, 1, 0]

            if self.affordance_dict['drawer']:
                self._top_drawer = bullet.objects.drawer(quat=quat, pos=np.array([0.6, s * 0.125, -.22]),
                    rgba=self.sample_object_color())
            else:
                self._top_drawer = bullet.objects.drawer(quat=quat, pos=np.array([0.6, s * 0.125, -.34]),
                    rgba=self.sample_object_color())
                # self._top_drawer = bullet.objects.drawer(quat=quat, pos=np.array([0.6, s * 1.125, -.34]),
                #     rgba=self.sample_object_color())
            
            # randomly initialize how open drawer is
            self.drawer_opening_num_ts = np.random.random_integers(low=1, high=60)
            #print("num_ts: ", self.drawer_opening_num_ts)
            open_drawer(self._top_drawer, num_ts=self.drawer_opening_num_ts)
            self.init_handle_pos = get_drawer_handle_pos(self._top_drawer)[1]

        # Button
        if self.affordance_dict['button']:
            num_drawers = self.affordance_dict['handle_drawer'] + self.affordance_dict['drawer']
            if num_drawers == 2:
                pos = np.array([0.6, s * 0.125, -.14])
            elif num_drawers == 1:
                pos = np.array([0.6, s * 0.125, -.25])
            else:
                pos = np.array([0.6, s * 0.125, -.34])

            self.button_pos = pos
            self.button_color = self.sample_object_color()
            self._objects['button'] = bullet.objects.button(pos=self.button_pos, rgba=self.button_color)
            self.init_button_height = get_button_cylinder_pos(self._objects['button'])[2]
            self.button_used = False

        self.tray_corners = {
            "back_left" : np.array([0.6, s * -0.15, -.35]),
            "back_right" : np.array([0.6, s * 0.15, -.35]),
            "front_left" : np.array([0.79, s * -0.12, -.35]),
            "front_right" : np.array([0.79, s * 0.12, -.35])
        }
        self.rand_obj_corners = {
            "back_left" : np.array([0.6, s * -0.15, -.25]),
            "back_right" : np.array([0.6, s * 0.15, -.25]),
            "front_left" : np.array([.78, s * -0.12, -.25]),
            "front_right" : np.array([.78, s * 0.12, -.25])
        }

        # Tray
        if self.affordance_dict['tray']:
            back_left = self.tray_corners['back_left']
            back_right = self.tray_corners['back_right']
            front_left = self.tray_corners['front_left']
            front_right = self.tray_corners['front_right']

            issue_catch = self.affordance_dict['handle_drawer'] and not self.affordance_dict['drawer']
            if self.test_env and (not self.affordance_dict['drawer']):
                tray_pos = front_right
            elif self.test_env and issue_catch:
                tray_pos = front_left
            elif self.test_env:
                tray_pos = back_left
            elif self.affordance_dict['drawer'] and self.affordance_dict['handle_drawer']:
                tray_pos = front_left
            elif self.affordance_dict['drawer']:
                tray_pos = random.choice([back_left, front_left])
            elif self.affordance_dict['handle_drawer']:
                tray_pos = random.choice([front_left, front_right])
            elif self.affordance_dict['button']:
                tray_pos = random.choice([back_left, front_left, front_right])
            else:
                tray_pos = random.choice([back_left, back_right, front_left, front_right])

            self.tray = bullet.objects.drawer_tray(pos=tray_pos, rgba=self.sample_object_color())
            self.tray_pos = tray_pos

        # Rand obj
        if self.affordance_dict['rand_obj']:
            back_left = self.rand_obj_corners['back_left']
            back_right = self.rand_obj_corners['back_right']
            front_left = self.rand_obj_corners['front_left']
            front_right = self.rand_obj_corners['front_right']

            if self.test_env or self.affordance_dict['drawer'] or self.affordance_dict['handle_drawer']:
                if self.affordance_dict['drawer'] and self.affordance_dict['handle_drawer']:
                    self._fixed_object_position = back_left
                elif self.affordance_dict['tray'] and np.array_equal(self.tray_corners['front_left'], self.tray_pos):
                    self._fixed_object_position = front_right
                else:
                    self._fixed_object_position = front_left
            elif self.affordance_dict['button']:
                opt_keys = ['back_left', 'front_left', 'front_right']
                opts = []
                if self.affordance_dict['tray']:
                    # don't want out of tray goal to be same as tray location
                    opts = [self.rand_obj_corners[key] for key in opt_keys if not np.array_equal(self.tray_corners[key], self.tray_pos)]
                else:
                    opts = [self.rand_obj_corners[key] for key in opt_keys]
                self._fixed_object_position = random.choice(opts)
            else:
                opt_keys = ['back_left', 'back_right', 'front_left', 'front_right']
                opts = []
                if self.affordance_dict['tray']:
                    # don't want out of tray goal to be same as tray location
                    opts = [self.rand_obj_corners[key] for key in opt_keys if not np.array_equal(self.tray_corners[key], self.tray_pos)]
                else:
                    opts = [self.rand_obj_corners[key] for key in opt_keys]
                self._fixed_object_position = random.choice(opts)
            
            if self.affordance_dict['tray']:
                # either initialize object in tray or at out_of_tray goal
                self.out_of_tray_goal = self._fixed_object_position
                if not self.test_env:
                    self._fixed_object_position = random.choice([self._fixed_object_position, self.tray_pos])

            self.add_object(object_position=self._fixed_object_position)

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
        if not self.affordance_dict['button']:
            return False

        curr_height = get_button_cylinder_pos(self._objects['button'])[2]
        pressed = (self.init_button_height - curr_height) > 0.01
        if pressed and not self.button_used:
            self.button_used = True
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
        if self.affordance_dict['rand_obj']:
            contained_obj = self.check_obj_bounding_box(self._objects['obj'])
        else:
            contained_obj = True

        if self.affordance_dict['drawer']:
            contained_lego = self.check_obj_bounding_box(self._objects['lego'])
        else:
            contained_lego = True

        if not contained_obj or not contained_lego:
            bullet.position_control(self._sawyer, self._end_effector,
                np.array(self._pos_init), self.default_theta)
            for i in range(3): self._simulate(np.array(self._pos_init), self.default_theta, -1)

        if not contained_obj:
            p.removeBody(self._objects['obj'])
            self.add_object(change_object=False)

        if not contained_lego:
            p.removeBody(self._objects['lego'])
            self._objects['lego'] = bullet.objects.drawer_lego(pos=self._reset_lego_position)

    def step(self, *action):
        # # TEMP TEMP TEMP #
        # from PIL import Image
        # img = Image.fromarray(np.uint8(self.fancy_render_obs()))
        # self.gif.append(img)
        # # TEMP TEMP TEMP #




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
            is_open = self.affordance_dict['drawer_open']
            sign = 1 if is_open else -1
            if self.affordance_dict['drawer']:
                slide_drawer(self._bottom_drawer, sign)
                self.affordance_dict['drawer_open'] = not is_open

        # Reset if bounding box is violated
        if self.use_bounding_box:
            self.enforce_bounding_box()

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False

        # reset button and resample rand_obj goal every episode
        if self.final_timestep:
            if self.affordance_dict['button']:
                p.removeBody(self._objects['button'])
                self._objects['button'] = bullet.objects.button(pos=self.button_pos, rgba=self.button_color)
                self.button_used = False
            if self.affordance_dict['rand_obj'] and not self.affordance_dict['tray']:
                self.obj_goal = np.random.uniform(low=self._goal_low, high=self._goal_high)

        return observation, reward, done, info

    def get_info(self):
        info = {}
        state_obs = self.get_observation()['state_observation']
        rand_obj_pos = state_obs[4:7]
        lego_pos = state_obs[7:10]
        bottom_drawer_pos = state_obs[10]
        top_drawer_pos = state_obs[11]
        button_pos = state_obs[12]

        if self.affordance_dict['rand_obj']:
            info['rand_obj_picked_up'] = rand_obj_pos[2] > self.pickup_eps

        if self.affordance_dict['drawer']:
            info['lego_picked_up'] = lego_pos[2] > self.pickup_eps

        return info

    def get_success_metric(self, curr_state, goal_state, key=None, success_list=None, present_list=None):
        if key == 'hand':
            i, j, thresh = 0, 3, self.obj_thresh
            is_task = 1
        elif key == 'rand_obj' or key == 'tray':
            i, j, thresh = 4, 7, self.obj_thresh
            is_task = int(curr_state[i] != 0)
        elif key == 'lego':
            i, j, thresh = 7, 10, self.obj_thresh
            is_task = int((curr_state[i] != 0) and (curr_state[12] != 0))
        elif key == 'bottom_drawer':
            i, j, thresh = 10, 11, self.drawer_thresh
            is_task = int((curr_state[i] != 0) and (curr_state[12] != 0))
        elif key == 'top_drawer':
            i, j, thresh = 11, 12, self.drawer_thresh
            is_task = int(curr_state[i] != 0)
        elif key == 'button':
            i, j, thresh = 12, 13, self.button_thresh
            is_task = int(curr_state[i] != 0)
        else:
            print('KEY ERROR')
            return 1/0

        curr_pos = curr_state[i:j]
        goal_pos = goal_state[i:j]
        success = int((np.linalg.norm(curr_pos - goal_pos) < thresh) and is_task)

        if present_list is not None:
            present_list.append(is_task)
        if success_list is not None:
            success_list.append(success)

        return success, is_task

    def get_contextual_diagnostics(self, paths, contexts):
        #from roboverse.utils.diagnostics import create_stats_ordered_dict
        from multiworld.envs.env_util import create_stats_ordered_dict
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"

        hand_success_list = []
        rand_obj_success_list = []
        tray_success_list = []
        lego_success_list = []
        bottom_drawer_success_list = []
        top_drawer_success_list = []
        button_success_list = []

        hand_present_list = []
        rand_obj_present_list = []
        tray_present_list = []
        lego_present_list = []
        bottom_drawer_present_list = []
        top_drawer_present_list = []
        button_present_list = []

        reward_list = []

        for i in range(len(paths)):
            curr_obs = paths[i]["observations"][-1][state_key]
            goal_obs = contexts[i][goal_key]

            hand_success, hand_present = self.get_success_metric(curr_obs, goal_obs,
                key='hand', success_list=hand_success_list, present_list=hand_present_list)
            rand_obj_success, rand_obj_present = self.get_success_metric(curr_obs, goal_obs,
                key='rand_obj', success_list=rand_obj_success_list, present_list=rand_obj_present_list)
            tray_success, tray_present = self.get_success_metric(curr_obs, goal_obs,
                key='tray', success_list=tray_success_list, present_list=tray_present_list)       
            lego_success, lego_present = self.get_success_metric(curr_obs, goal_obs,
                key='lego', success_list=lego_success_list, present_list=lego_present_list)
            bd_success, bd_present = self.get_success_metric(curr_obs, goal_obs,
                key='bottom_drawer', success_list=bottom_drawer_success_list, present_list=bottom_drawer_present_list)
            td_success, td_present = self.get_success_metric(curr_obs, goal_obs,
                key='top_drawer', success_list=top_drawer_success_list, present_list=top_drawer_present_list)
            button_success, button_present = self.get_success_metric(curr_obs, goal_obs,
                key='button', success_list=button_success_list, present_list=button_present_list)

            num_tasks = hand_present + rand_obj_present + tray_present + lego_present + bd_present + td_present + button_present
            num_success = hand_success + rand_obj_success + tray_success + lego_success + bd_success + td_success + button_success
            reward_list.append(num_success / num_tasks)


        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/hand_success", hand_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/rand_obj_success", rand_obj_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/tray_success", tray_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/lego_success", lego_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/bottom_drawer_success", bottom_drawer_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/top_drawer_success", top_drawer_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/button_success", button_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success", reward_list))

        diagnostics.update(create_stats_ordered_dict(goal_key + "/hand_present", hand_present_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/rand_obj_success", rand_obj_present_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/tray_success", tray_present_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/lego_success", lego_present_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/bottom_drawer_success", bottom_drawer_present_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/top_drawer_success", top_drawer_present_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/button_success", button_present_list))


        hand_success_list = []
        rand_obj_success_list = []
        tray_success_list = []
        lego_success_list = []
        bottom_drawer_success_list = []
        top_drawer_success_list = []
        button_success_list = []
        reward_list = []

        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                curr_obs = paths[i]["observations"][j][state_key]
                goal_obs = contexts[i][goal_key]

                hand_success, hand_present = self.get_success_metric(curr_obs, goal_obs,
                    key='hand', success_list=hand_success_list)
                rand_obj_success, rand_obj_present = self.get_success_metric(curr_obs, goal_obs,
                    key='rand_obj', success_list=rand_obj_success_list)
                tray_success, tray_present = self.get_success_metric(curr_obs, goal_obs,
                    key='tray', success_list=tray_success_list)
                lego_success, lego_present = self.get_success_metric(curr_obs, goal_obs,
                    key='lego', success_list=lego_success_list)
                bd_success, bd_present = self.get_success_metric(curr_obs, goal_obs,
                    key='bottom_drawer', success_list=bottom_drawer_success_list)
                td_success, td_present = self.get_success_metric(curr_obs, goal_obs,
                    key='top_drawer', success_list=top_drawer_success_list)
                button_success, button_present = self.get_success_metric(curr_obs, goal_obs,
                    key='button', success_list=button_success_list)

                num_tasks = hand_present + rand_obj_present + tray_present + lego_present + bd_present + td_present + button_present
                num_success = hand_success + rand_obj_success + tray_success + lego_success + bd_success + td_success + button_success
                reward_list.append(num_success / num_tasks)

        diagnostics.update(create_stats_ordered_dict(goal_key + "/hand_success", hand_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/rand_obj_success", rand_obj_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/tray_success", tray_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/lego_success", lego_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/bottom_drawer_success", bottom_drawer_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/top_drawer_success", top_drawer_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/button_success", button_success_list))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/success", reward_list))

        return diagnostics

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def fancy_render_obs(self):
        fancy_obs_dim = 256
        fancy_projection_matrix_obs = bullet.get_projection_matrix(
            fancy_obs_dim, fancy_obs_dim)

        img, depth, segmentation = bullet.render(
            fancy_obs_dim, fancy_obs_dim, self._view_matrix_obs,
            fancy_projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def fancy_get_image(self, width, height):
        image = np.float32(self.fancy_render_obs())
        return image

    def set_goal(self, goal):
        self.goal_pos = goal['state_desired_goal'][self.start_obj_ind:self.start_obj_ind + 3]

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info=None, print_stats=False):
        curr_state = self.get_observation()['state_achieved_goal']
        x = []
        hand_success, hand_present = self.get_success_metric(curr_state, self.goal_state, 'hand')
        rand_obj_success, rand_obj_present = self.get_success_metric(curr_state, self.goal_state, 'rand_obj')
        tray_success, tray_present = self.get_success_metric(curr_state, self.goal_state, 'tray')
        lego_success, lego_present = self.get_success_metric(curr_state, self.goal_state, 'lego')
        bd_success, bd_present = self.get_success_metric(curr_state, self.goal_state, 'bottom_drawer')
        td_success, td_present = self.get_success_metric(curr_state, self.goal_state, 'top_drawer')
        button_success, button_present = self.get_success_metric(curr_state, self.goal_state, 'button')
        if print_stats:
            print('-----------------')
            print('Hand: ', hand_success)
            if self.affordance_dict['rand_obj'] and self.affordance_dict['tray']: print('Tray: ', tray_success)
            elif self.affordance_dict['rand_obj']: print('Rand Obj: ', rand_obj_success)
            if self.affordance_dict['drawer']: print('Lego: ', lego_success)
            if self.affordance_dict['drawer']: print('Bottom Drawer: ', bd_success)
            if self.affordance_dict['handle_drawer']: 
                print('Top Drawer Open: ', td_success)
            if self.affordance_dict['button']: print('Button: ', button_success)
        reward = (rand_obj_success or tray_success) + lego_success + (bd_success or button_success) + td_success + hand_success
        return reward

    def sample_goals(self):
        self.obj_goal = np.random.uniform(low=self._goal_low, high=self._goal_high)
        self.update_tray_goal()
        self.lego_goal = np.random.uniform(low=self._goal_low, high=self._goal_high)
        self.hand_goal = np.random.uniform(low=self._pos_low, high=self._pos_high)
        self.hand_goal[2] = -0.11
        s = self.affordance_dict['side_sign']

        if self.affordance_dict['drawer']:
            ld_pos = self.get_object_pos('bottom_drawer') - self.affordance_dict['drawer_open'] * np.array([0.15, 0, 0])
            is_open = np.random.uniform() < 0.5
            self.bd_goal = (ld_pos + np.array([0.15, 0, 0])) if is_open else ld_pos
        else:
            self.bd_goal = np.zeros(3)

        self.update_drawer_goal()

        if self.affordance_dict['button']:
            press = np.random.uniform() < 0.5
            self.button_goal = (self.init_button_height - 0.01) if press else self.init_button_height
        else:
            self.button_goal = 0

        self.update_goal_state()

    def reset(self):
        if self.expl:
            self.reset_counter += 1
            if self.reset_interval == self.reset_counter:
                self.reset_counter = 0
                self.tasks_done = 0
                self.tasks_done_names = []
            else:
                self.update_tray_goal()
                self.update_drawer_goal()
                return self.get_observation()  
        else:
            self.trajectory_done = False

        # # TEMP TEMP TEMP #
        # try:
        #     import skvideo
        #     rand_num = 6
        #     filepath = '/home/ashvin/data/sasha/fancy_videos/{0}_rollout.mp4'.format(rand_num)
        #     outputdata = np.stack(self.gif)
        #     import ipdb; ipdb.set_trace()
        #     skvideo.io.vwrite(filepath, outputdata)
        #     # self.gif[0].save('/home/ashvin/data/sasha/fancy_videos/{0}_rollout.gif'.format(rand_num),
        #     #            format='GIF', append_images=self.gif[:],
        #     #            save_all=True, duration=100, loop=0)
        # except AttributeError:
        #     self.gif = []
        # # TEMP TEMP TEMP #

        # Load Enviorment
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_table()
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

        # # TEMP TEMP TEMP #
        # self.gif = []
        # # TEMP TEMP TEMP #
        return self.get_observation()

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def compute_reward_gr(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 1]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_obj_ind:self.start_obj_ind + 1]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        object_goal_success = object_goal_distance < self._success_threshold
        return object_goal_success - 1
    # def compute_reward_gr(self, obs, actions, next_obs, contexts):
    #     obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
    #     obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_obj_ind:self.start_obj_ind + 3]
    #     object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
    #     object_goal_success = object_goal_distance < self._success_threshold
    #     return object_goal_success - 1

    def compute_reward(self, obs, actions, next_obs, contexts):
        return self.compute_reward_gr(obs, actions, next_obs, contexts)

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

        if self.affordance_dict['rand_obj']:
            object_pos = bullet.get_body_info(self._objects['obj'], quat_to_deg=False)['pos']
        else:
            object_pos = [0,0,0]

        if self.affordance_dict['drawer']:
            bottom_drawer_pos = [get_drawer_bottom_pos(self._bottom_drawer)[0]]
            lego_pos = bullet.get_body_info(self._objects['lego'], quat_to_deg=False)['pos']
        else:
            bottom_drawer_pos = [0]
            lego_pos = [0,0,0]

        if self.affordance_dict['handle_drawer']:
            top_drawer_pos = [get_drawer_handle_pos(self._top_drawer)[1]]
        else:
            top_drawer_pos = [0]

        if self.affordance_dict['button']:
            button_height = [get_button_cylinder_pos(self._objects['button'])[2]]
        else:
            button_height = [0]

        if self.DoF > 3:
            #(hand_pos, hand_theta, gripper, obj_pos, lego_pos, bd_pos, td_pos, b_pos)
            observation = np.concatenate((
                end_effector_pos, hand_theta, gripper_tips_distance,
                object_pos, lego_pos, bottom_drawer_pos, top_drawer_pos, button_height))
        else:
            #(hand_pos, gripper, obj_pos, lego_pos, bd_pos, td_pos, b_pos)
            observation = np.concatenate((
                end_effector_pos, gripper_tips_distance, object_pos,
                lego_pos, bottom_drawer_pos, top_drawer_pos, button_height))

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=self.goal_state.copy(),
            state_desired_goal=self.goal_state.copy(),
            achieved_goal=observation,
            state_achieved_goal=observation,
            )

        return obs_dict

    ### Helper Functions
    def handle_more_open_than_closed(self):
        return np.abs(self.get_object_pos('drawer_handle')[1]) > (td_close_pos+td_open_pos)/2
    
    def get_drawer_goal(self):
        if self.affordance_dict['handle_drawer']:
            return self.td_goal_close if self.handle_more_open_than_closed() else self.td_goal_open
        else:
            return 0

    def drawer_done(self, goal):
        return np.abs(goal - self.get_object_pos('drawer_handle')[1]) < 0.025

    def update_tray_goal(self):
        # Change tray goal every episode if task accomplished
        if self.affordance_dict['tray'] and self.affordance_dict['rand_obj']:
            obj_pos = self.get_object_pos('obj')
            obj_in_tray = True if np.linalg.norm(obj_pos[:2] - np.array(self.tray_pos)[:2]) < 0.06 else False
            if not obj_in_tray:
                self.obj_goal = self.tray_pos
            else:
                self.obj_goal = self.out_of_tray_goal
    
    def update_drawer_goal(self):
        s = self.affordance_dict['side_sign']
        if self.affordance_dict['handle_drawer']:
            self.td_goal_close = np.random.uniform(low=td_close_pos, high=(td_close_pos+td_open_pos)/2)
            self.td_goal_close *= -s
            
            # we take max so that close and open goals have some separation always
            td_goal_open_low = max(self.td_goal_close + (td_close_pos+td_open_pos)/4, (td_close_pos+td_open_pos)/2)
            self.td_goal_open = np.random.uniform(low=td_goal_open_low, high=td_open_pos)
            self.td_goal_open *= -s

            # if sampled goal is already achieved by drawer initialization, then set goal to be drawer full open/close
            if self.drawer_done(self.td_goal_close):
                self.td_goal_close = -s * td_close_pos
            if self.drawer_done(self.td_goal_open):
                self.td_goal_open = -s * td_open_pos
            self.td_goal = self.get_drawer_goal()

            # self.td_goal_open, self.td_goal_close = 0.10142616384776754, 0.08013844454185876
            # print("self.td_goal_open: ", self.td_goal_open, ", self.td_goal_close: ", self.td_goal_close)
        else:
            self.td_goal = 0
            self.td_goal_close = 0
            self.td_goal_open = 0

    def get_num_tasks(self):
        if self.debug == 'tray':
            num_tasks = (self.affordance_dict['rand_obj'] and self.affordance_dict['tray'])
        else:
            top_drawer = self.affordance_dict['handle_drawer']
            bottom_drawer = self.affordance_dict['button']
            pnp_or_tray = self.affordance_dict['rand_obj']
            num_tasks = top_drawer + bottom_drawer + pnp_or_tray
        return num_tasks

    def update_goal_state(self):
        if self.DoF > 3:
            self.goal_state = np.concatenate([
                self.hand_goal, [0,0,0,0], [0],
                self.obj_goal, self.lego_goal, [self.bd_goal[0]],
                [self.get_drawer_goal()], [self.button_goal]])
        else:
            self.goal_state = np.concatenate([
                self.hand_goal, [0], self.obj_goal, self.lego_goal, [self.bd_goal[0]],
                [self.get_drawer_goal()], [self.button_goal]])

    ### DEMO COLLECTING FUNCTIONS BEYOND THIS POINT ###

    def demo_reset(self):
        self.task_dict = {'drawer': lambda : self.move_drawer(),
                        'button': self.press_button,
                        'hand': self.move_hand,
                        'lego': lambda: self.move_obj('lego', self.lego_goal),
                        'rand_obj': lambda: self.move_obj('obj', self.obj_goal),
                        }
        self.timestep = 0
        self.grip = -1.
        reset_obs = self.reset()
        self.curr_task = self.sample_task()

        # for logging purposes
        self.record_curr_task()
        #print('----Initial----')
        #self.get_reward(print_stats=True)
        return reset_obs
    
    def record_curr_task(self):
        self.recorded_curr_task = self.curr_task
        if self.recorded_curr_task == "rand_obj":
            if self.affordance_dict["tray"]:
                self.recorded_curr_task = "tray"
        elif self.recorded_curr_task == "button":
            if self.affordance_dict["drawer"]:
                self.recorded_curr_task = "button_with_drawer"
            else:
                self.recorded_curr_task = "button_without_drawer"

    def sample_task(self):
        options = []

        if self.debug == 'tray':
            return 'rand_obj'

        if self.affordance_dict['handle_drawer']:
            options.append('top_drawer')
        if self.affordance_dict['button']:
            options.append('bottom_drawer')
        if self.affordance_dict['rand_obj'] or self.affordance_dict['drawer'] and self.affordance_dict['drawer_open']: 
            options.append('pnp')

        # edge case: bottom/top drawer exists, top drawer is open. don't do tray or else gripper will hit handle.
        if self.affordance_dict['drawer'] and self.affordance_dict['handle_drawer'] and self.handle_more_open_than_closed():
            pass
        else:
            if self.affordance_dict['tray'] and self.affordance_dict['rand_obj']:
                options.append('tray')
        
        remaining_tasks = options
        # this shouldn't occur but added in just in case
        if len(remaining_tasks) == 0: remaining_tasks.append('hand')
        sampled_task = random.choice(remaining_tasks)

        subtask = 'hand'
        if sampled_task == 'top_drawer':
            subtask = 'drawer'
        elif sampled_task == 'bottom_drawer':
            subtask = 'button'
        elif sampled_task == 'pnp':
            pnp_options = []
            if self.affordance_dict['rand_obj']:
                pnp_options.append('rand_obj')
            if self.affordance_dict['drawer'] and self.affordance_dict['drawer_open']:
                pnp_options.append('lego')
            subtask = random.choice(pnp_options)
        elif sampled_task == 'tray':
            subtask = 'rand_obj'

        #print('Current Task: ' + subtask)
        return subtask

    def get_demo_action(self, first_timestep=False, final_timestep=False):
        self.final_timestep = final_timestep
        action, done = self.task_dict[self.curr_task]()

        if self.expl:
            if first_timestep:
                self.trajectory_done = False
                action = np.array([0, 0, 1])
            if final_timestep:
                self.trajectory_done = True
                if self.curr_task in self.tasks_done_names:
                    self.tasks_done_names.remove(self.curr_task)
                self.tasks_done_names.append(self.curr_task)
                self.tasks_done += 1

            if done:
                self.trajectory_done = True

            if self.trajectory_done:
                action = np.array([0, 0, 1, -1])
            else:
                action = np.append(action, [self.grip])
                action = np.random.normal(action, 0.1)
        else:
            if done:
                #self.get_reward(print_stats=True)
                self.tasks_done_names.append(self.curr_task)
                self.update_drawer_goal()
                self.update_goal_state()
                self.curr_task = self.sample_task()
                self.tasks_done += 1

                action = np.array([0, 0, 1, -1])
            else:
                action = np.append(action, [self.grip])
                action = np.random.normal(action, 0.1)

        action = np.clip(action, a_min=-1, a_max=1)
        self.timestep += 1

        print(first_timestep, final_timestep)

        return action

    def move_hand(self):
        ee_pos = self.get_end_effector_pos()
        above = ee_pos[2] >= -0.105
        self.grip = -1.
        done = 0

        if not above:
            #print('Stage 1')
            action = np.array([0.,0., 1.])
        else:
            #print('Stage 2')
            action = (self.hand_goal - ee_pos) * 3.0

        return action, done
    
    def move_drawer(self):
        ee_pos = self.get_end_effector_pos()
        s = self.affordance_dict['side_sign']
        target_pos = self.get_object_pos('drawer_handle') + s * np.array([0,0.015,0]) + np.array([0,0,-.0280 if self.affordance_dict['drawer'] else -.0080])
        y_aligned = np.linalg.norm(target_pos[0] - ee_pos[0]) < 0.035
        x_aligned = (s*(target_pos[1] - ee_pos[1])) > 0.0 and (s*(target_pos[1] - ee_pos[1])) < 0.045
        behind = (s*(target_pos[1] - ee_pos[1])) < 0.017
        aligned = y_aligned and x_aligned
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.01
        done = self.drawer_done(self.td_goal)
        above = ee_pos[2] >= -0.105
        self.grip = -1.

        if not aligned and not above:
            #print('Stage 1')
            action = np.array([0.,0., 1.])
        elif (not aligned):# or behind:
            #print('Stage 2')
            action = (target_pos - ee_pos) * 3.0
            if behind: action[1] = -s # Otherwise we hit the drawer
            action[2] = 0.
            action *= 2.0
        elif not self.affordance_dict['drawer'] and ee_pos[2] >= -0.255:
            #print('Stage 2.5')
            action = np.array([0.,0., -1.])
        elif aligned and not enclosed:
            #print('Stage 3')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
        else:
            #print('Stage 4')
            action = np.sign(np.array([0, self.td_goal - ee_pos[1], 0]))

        # print(aligned, above, behind, enclosed, done)
        # print(self.get_object_pos('drawer_handle'))

        if done:
            action = np.array([0,0,1])

        return action, done

    def press_button(self):
        done = self.button_used
        s = self.affordance_dict['side_sign']
        ee_pos = self.get_end_effector_pos()
        target_pos = self.get_object_pos('button') + s * np.array([0, -0.0375, 0])
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.02
        above = ee_pos[2] >= -0.101
        self.grip = -1.

        if not aligned and not above:
            #print('Stage 1')
            action = np.array([0.,0., 1.])
        elif not aligned:
            #print('Stage 2')
            action = (target_pos - ee_pos) * 3.0
            action[2] = 0.
            action *= 3.0
        else:
            #print('Stage 3')
            action = (target_pos - ee_pos)
            action[2] = -1.
            #action = np.array([0., 0., -1.])

        return action, done

    def move_obj(self, obj, goal):
        ee_pos = self.get_end_effector_pos()
        s = self.affordance_dict['side_sign']
        adjustment = np.array([0.00, 0.01, 0]) if obj == 'lego' else np.array([0.00, -0.011, 0])
        target_pos = self.get_object_pos(obj) + s * adjustment
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.055
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
        done_xy = np.linalg.norm(target_pos[:2] - goal[:2]) < 0.05
        done = done_xy and np.linalg.norm(target_pos[2] - goal[2]) < 0.03
        above = ee_pos[2] >= -0.125

        if not aligned and not above:
            #print('Stage 1')
            action = np.array([0.,0., 1.])
            self.grip = -1.
        elif not aligned:
            #print('Stage 2')
            action = (target_pos - ee_pos) * 3.0
            action[2] = 0.
            action *= 2.0
            self.grip = -1.
        elif aligned and not enclosed and self.grip < 1:
            #print('Stage 3')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 1.5
            self.grip = -1.
        elif enclosed and self.grip < 1:
            #print('Stage 4')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            self.grip += 0.5
        elif not above:
            #print('Stage 5')
            action = np.array([0.,0., 1.])
            self.grip = 1.
        elif not done_xy:
            #print('Stage 6')
            action = goal - ee_pos
            action[2] = 0
            action *= 3.0
            self.grip = 1.
        else:
            #print('Stage 7')
            action = np.array([0.,0.,0.])
            self.grip = -1
        
        # print(aligned, above, done, enclosed, self.grip)
        # print(target_pos, ee_pos, goal)

        return action, done
