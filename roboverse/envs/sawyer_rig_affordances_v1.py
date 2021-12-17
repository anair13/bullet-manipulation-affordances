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
from PIL import Image
import pkgutil

test_set = ['mug', 'long_sofa', 'camera', 'grill_trash_can', 'beer_bottle']

quat_dict={'mug': [0, -1, 0, 1],'long_sofa': [0, 0, 0, 1],'camera': [-1, 0, 0, 0],
        'grill_trash_can': [0, 0, 1, 1], 'beer_bottle': [0, 0, 1, -1]}

# Constants
td_close_coeff = 0.13754340000000412
td_open_coeff = 0.29387810000002523
td_offset_coeff = 0.001

gripper_bounding_x = [.46, .84] #[0.4704, 0.8581]
gripper_bounding_y = [-.19, .19] #[-0.1989, 0.2071]

class SawyerRigAffordancesV1(SawyerBaseEnv):

    def __init__(self,
                 reward_type='shaped',
                 reward_min=-2.5,
                 randomize=False,
                 observation_mode='state',
                 obs_img_dim=48,
                 transpose_image=False,
                 invisible_robot=False,
                 object_subset='test',
                 use_bounding_box=True,
                 random_color_p=1.0,
                 spawn_prob=0.75,
                 quat_dict=quat_dict,
                 task='goal_reaching',
                 test_env=False,
                 DoF=4,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param reward_type: one of 'shaped', 'sparse'
        :param reward_min: minimum possible reward per timestep
        :param randomize: whether to randomize the object position or not
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        :param invisible_robot: the robot arm is invisible when set to True
        """
        assert DoF in [4]
        assert task in ['goal_reaching', 'pickup']

        is_set = object_subset in ['test', 'train', 'all']
        is_list = type(object_subset) == list
        assert is_set or is_list

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
        self.object_subset = object_subset
        self.spawn_prob = spawn_prob
        self._ddeg_scale = 5
        self.task = task
        self.DoF = DoF
        self.test_env = test_env
        self.use_multiple_goals = kwargs.pop('use_multiple_goals', False)
        self.test_env_seed = kwargs.pop('test_env_seed', None) if self.test_env else None
        self.test_env_seeds = kwargs.pop('test_env_seeds', None) if self.test_env else None

        if self.test_env:
            self.random_color_p = 0.0
            self.object_subset = kwargs.pop('test_object_subset', ['grill_trash_can'])

        self.drawer_thresh = 0.065
        self.gripper_pos_thresh = 0.08
        self.gripper_rot_thresh = 10

        self.object_dict, self.scaling = self.get_object_info()
        self.curr_object = None
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self.obs_img_dim = obs_img_dim
        self.new_view = kwargs.pop('new_view', False)
        self.close_view = kwargs.pop('close_view', False)
        if self.new_view:
            if self.close_view:
                p = .9
                gripper_bounding_x[1] = ((1-p) * gripper_bounding_x[0] + p * gripper_bounding_x[1])
                self._view_matrix_obs = bullet.get_view_matrix(
                    target_pos=[0.7, 0, -0.25], distance=0.425,
                    yaw=90, pitch=-50, roll=0, up_axis_index=2)
            else:
                self._view_matrix_obs = bullet.get_view_matrix(
                    target_pos=[0.7, 0, -0.4], distance=0.76,
                    yaw=90, pitch=-50, roll=0, up_axis_index=2)
        else:
            assert not self.close_view
            self._view_matrix_obs = bullet.get_view_matrix(
                target_pos=[.7, 0, -0.25], distance=0.425,
                yaw=90, pitch=-37, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1

        self.expert_policy_std = kwargs.pop('expert_policy_std', 0.1)

        # Reset-free
        self.reset_interval = kwargs.pop('reset_interval', 1)
        self.reset_counter = self.reset_interval-1
        self.expl = kwargs.pop('expl', False)
        self.trajectory_done = False
        self.final_timestep = False
        self.drawer_sliding = kwargs.pop('drawer_sliding', False)

        # Drawer
        self.gripper_has_been_above = False
        self.fix_drawer_orientation = kwargs.pop('fix_drawer_orientation', False)
        self.fix_drawer_orientation_semicircle = kwargs.pop('fix_drawer_orientation_semicircle', False)
        assert not (self.fix_drawer_orientation and self.drawer_orientation_semicircle)
        self.red_drawer_base = kwargs.pop('red_drawer_base', False)

        # Anti-aliasing
        self.downsample = kwargs.pop('downsample', False)
        self.env_obs_img_dim = kwargs.pop('env_obs_img_dim', self.obs_img_dim)

        # Debugging
        self.full_open_close_init_and_goal = kwargs.pop('full_open_close_init_and_goal', False)
        if self.full_open_close_init_and_goal:
            self.current_goal_is_open = False

        super().__init__(*args, **kwargs)

        # Need to overwrite in some cases, registration isnt working
        self._max_force = 100
        self._action_scale = 0.05
        self._pos_init = [0.6, -0.15, -0.2]
        self._pos_low = [0.5,-0.2,-.36]
        self._pos_high = [0.85,0.2,-0.1]

        # # Speed up rendering
        # egl = pkgutil.get_loader('eglRenderer')
        # eglPluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

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

        observation_dim = 11
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
        self._table = bullet.objects.table(rgba=[.92,.85,.7,1])
        # self._debug1 = bullet.objects.button(pos=[.46, -.19, -.25])
        # self._debug2 = bullet.objects.button(pos=[.84, .19, -.25])
        if self.new_view:
            self._wall = bullet.objects.wall()

        ## Top Drawer
        if self.test_env and not self.test_env_seed:
            self.drawer_yaw = 180
            drawer_frame_pos = np.array([.6, -.19, -.34])
        else:
            if self.fix_drawer_orientation:
                self.drawer_yaw = 180
            elif self.fix_drawer_orientation_semicircle:
                self.drawer_yaw = random.uniform(0, 180)
            else:
                self.drawer_yaw = random.uniform(0, 360)
   
            while(True):
                drawer_frame_pos = np.array([random.uniform(gripper_bounding_x[0], gripper_bounding_x[1]), random.uniform(gripper_bounding_y[0], gripper_bounding_y[1]), -.34])
                drawer_handle_open_goal_pos = drawer_frame_pos + td_open_coeff * np.array([np.sin(self.drawer_yaw * np.pi / 180) , -np.cos(self.drawer_yaw * np.pi / 180), 0])
                if gripper_bounding_x[0] <= drawer_handle_open_goal_pos[0] <= gripper_bounding_x[1] \
                    and gripper_bounding_y[0] <= drawer_handle_open_goal_pos[1] <= gripper_bounding_y[1]:
                    break
        quat = deg_to_quat([0, 0, self.drawer_yaw])
        
        # For debugging: hardcode drawer_yaw and drawer_frame_pos
        # self.drawer_yaw = 160.10009720998795
        # rot = Rotation.from_euler('xyz', [0, 0, self.drawer_yaw], degrees=True)
        # quat = rot.as_quat()
        # drawer_frame_pos = np.array(list((0.6351875030272269, -0.10053104435094253, -0.34)))
        #drawer_yaw:  160.10009720998795 , drawer_frame_pos:  (0.6351875030272269, -0.10053104435094253, -0.34)

        if self.drawer_sliding:
            if self.red_drawer_base:
                self._top_drawer = bullet.objects.drawer_sliding_red_base(quat=quat, pos=drawer_frame_pos, rgba=self.sample_object_color())
            else:
                self._top_drawer = bullet.objects.drawer_sliding(quat=quat, pos=drawer_frame_pos, rgba=self.sample_object_color())
        else:
            if self.red_drawer_base:
                self._top_drawer = bullet.objects.drawer_red_base(quat=quat, pos=drawer_frame_pos, rgba=self.sample_object_color())
            else:
                self._top_drawer = bullet.objects.drawer(quat=quat, pos=drawer_frame_pos, rgba=self.sample_object_color())
        
        # case: full open/close drawer initialization/goal
        if self.full_open_close_init_and_goal:
            # flip goals/initializations between open/close drawer
            self.current_goal_is_open = not self.current_goal_is_open
            # case: close drawer initialization + open drawer goal
            if self.current_goal_is_open:
                pass
            # case: open drawer initialization + close drawer goal
            else:
                open_drawer(self._top_drawer, num_ts=60)
        # case: uniform random drawer initialization/goal
        else:
            # randomly initialize how open drawer is
            open_drawer(self._top_drawer, num_ts=np.random.random_integers(low=1, high=60))

        self.init_handle_pos = get_drawer_handle_pos(self._top_drawer)[1]

        ## Distractor Objects
        num_objects = 1 if self.test_env else np.random.randint(3)
        for _ in range(0, num_objects):
            self.spawn_object()

        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')
    
    def sample_quat(self, object_name):
        if object_name in self.quat_dict:
            return self.quat_dict[self.curr_object]
        return deg_to_quat(np.random.randint(0, 360, size=3))

    def spawn_object(self, quat=None):
        # Pick object if necessary and save information
        self.curr_object, self.curr_id = random.choice(list(self.object_dict.items()))
        self.curr_color = self.sample_object_color()

        object_position = self.sample_object_location()
        if object_position is None:
            return

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
    
    def sample_object_location(self):
        if self.test_env:
            obj_pos = np.array([.85, -.15, 0])
        else:
            obj_pos = np.array([random.uniform(gripper_bounding_x[0], gripper_bounding_x[1]), random.uniform(gripper_bounding_y[0], gripper_bounding_y[1]), 0])
        return obj_pos

    def _format_action(self, *action):
        if len(action) == 1:
            delta_pos, delta_yaw, gripper = action[0][:3], action[0][3:4], action[0][-1]
        elif len(action) == 3:
            delta_pos, delta_yaw, gripper = action[0], action[1], action[2]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))

        delta_angle = [0, 0, delta_yaw[0]]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def step(self, *action):
        # Get positional information
        pos = bullet.get_link_state(self._sawyer, self._end_effector, 'pos')
        curr_angle = bullet.get_link_state(self._sawyer, self._end_effector, 'theta')
        default_angle = quat_to_deg(self.default_theta)

        # Keep necesary degrees of theta fixed
        angle = np.append(default_angle[:2], [curr_angle[2]])

        # If angle is part of action, use it
        delta_pos, delta_angle, gripper = self._format_action(*action)
        angle += delta_angle * self._ddeg_scale

        # Update position and theta
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)
        theta = deg_to_quat(angle)
        self._simulate(pos, theta, gripper)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False

        return observation, reward, done, info

    def get_info(self):
        return {}

    def get_success_metric(self, curr_state, goal_state, success_list=None, key=None):
        success = 0
        if key == 'top_drawer':
            curr_pos = curr_state[8:11]
            goal_pos = goal_state[8:11]
            success = int(self.drawer_done(curr_pos, goal_pos))
        else:
            pos = curr_state[0:3]
            goal_pos = goal_state[0:3]
            deg = quat_to_deg(curr_state[3:7])
            goal_deg = quat_to_deg(goal_state[3:7])

            if key == 'gripper_position':
                success = int(np.linalg.norm(pos - goal_pos) < self.gripper_pos_thresh)   
            elif key == 'gripper_rotation_roll':
                success = int(self.norm_deg(deg[0], goal_deg[0]) < self.gripper_rot_thresh)   
            elif key == 'gripper_rotation_pitch':
                success = int(self.norm_deg(deg[1], goal_deg[1]) < self.gripper_rot_thresh)   
            elif key == 'gripper_rotation_yaw':
                success = int(self.norm_deg(deg[2], goal_deg[2]) < self.gripper_rot_thresh)   
            elif key == 'gripper_rotation':
                success = int(np.sqrt(self.norm_deg(deg[0], goal_deg[0])**2 + self.norm_deg(deg[1], goal_deg[1])**2 + self.norm_deg(deg[2], goal_deg[2])**2) < self.gripper_rot_thresh)
            elif key == 'gripper':
                success = int(np.linalg.norm(pos - goal_pos) < self.gripper_pos_thresh) and int(np.sqrt(self.norm_deg(deg[0], goal_deg[0])**2 + self.norm_deg(deg[1], goal_deg[1])**2 + self.norm_deg(deg[2], goal_deg[2])**2) < self.gripper_rot_thresh)
        if success_list is not None:
            success_list.append(success)
        return success
    
    def get_distance_metric(self, curr_state, goal_state, distance_list=None, key=None):
        distance = float("inf")
        if key == 'top_drawer':
            curr_pos = curr_state[8:11]
            goal_pos = goal_state[8:11]
            distance = np.linalg.norm(curr_pos-goal_pos)
        else:
            pos = curr_state[0:3]
            goal_pos = goal_state[0:3]
            deg = quat_to_deg(curr_state[3:7])
            goal_deg = quat_to_deg(goal_state[3:7])

            if key == 'gripper_position':
                distance = np.linalg.norm(pos - goal_pos) 
            elif key == 'gripper_rotation_roll':
                distance = self.norm_deg(deg[0], goal_deg[0])
            elif key == 'gripper_rotation_pitch':
                distance = self.norm_deg(deg[1], goal_deg[1])
            elif key == 'gripper_rotation_yaw':
                distance = self.norm_deg(deg[2], goal_deg[2])
            elif key == 'gripper_rotation':
                distance = np.sqrt(self.norm_deg(deg[0], goal_deg[0])**2 + self.norm_deg(deg[1], goal_deg[1])**2 + self.norm_deg(deg[2], goal_deg[2])**2)
        if distance_list is not None:
            distance_list.append(distance)
        return distance

    def norm_deg(self, deg1, deg2):
        return min(np.linalg.norm((360 + deg1 - deg2) % 360), np.linalg.norm((360 + deg2 - deg1) % 360))

    def get_gripper_deg(self, curr_state, roll_list=None, pitch_list=None, yaw_list=None):
        quat = curr_state[3:7]
        deg = quat_to_deg(quat)
        if roll_list is not None:
            roll_list.append(deg[0])
        if pitch_list is not None:
            pitch_list.append(deg[1])
        if yaw_list is not None:
            yaw_list.append(deg[2])

        return deg

    def get_contextual_diagnostics(self, paths, contexts):
        #from roboverse.utils.diagnostics import create_stats_ordered_dict
        from multiworld.envs.env_util import create_stats_ordered_dict
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"

        success_keys = ["top_drawer", "gripper_position", "gripper_rotation_roll", "gripper_rotation_pitch", "gripper_rotation_yaw", "gripper_rotation", "gripper"]
        distance_keys = ["top_drawer", "gripper_position", "gripper_rotation_roll", "gripper_rotation_pitch", "gripper_rotation_yaw", "gripper_rotation"]

        dict_of_success_lists = {}
        for k in success_keys:
            dict_of_success_lists[k] = []
        
        dict_of_distance_lists = {}
        for k in distance_keys:
            dict_of_distance_lists[k] = []
        
        for i in range(len(paths)):
            curr_obs = paths[i]["observations"][-1][state_key]
            goal_obs = contexts[i][goal_key]
            for k in success_keys:
                self.get_success_metric(curr_obs, goal_obs, success_list=dict_of_success_lists[k], key=k)
            for k in distance_keys:
                self.get_distance_metric(curr_obs, goal_obs, distance_list=dict_of_distance_lists[k], key=k)
        for k in success_keys:
            diagnostics.update(create_stats_ordered_dict(goal_key + f"/final/{k}_success", dict_of_success_lists[k]))
        for k in distance_keys:
            diagnostics.update(create_stats_ordered_dict(goal_key + f"/final/{k}_distance", dict_of_distance_lists[k]))
        
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                curr_obs = paths[i]["observations"][j][state_key]
                goal_obs = contexts[i][goal_key]
                for k in success_keys:
                    self.get_success_metric(curr_obs, goal_obs, success_list=dict_of_success_lists[k], key=k)
                for k in distance_keys:
                    self.get_distance_metric(curr_obs, goal_obs, distance_list=dict_of_distance_lists[k], key=k)
        for k in success_keys:
            diagnostics.update(create_stats_ordered_dict(goal_key + f"/{k}_success", dict_of_success_lists[k]))
        for k in distance_keys:
            diagnostics.update(create_stats_ordered_dict(goal_key + f"/{k}_distance", dict_of_distance_lists[k]))

        gripper_rotation_roll_list = []
        gripper_rotation_pitch_list = []
        gripper_rotation_yaw_list = []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                curr_obs = paths[i]["observations"][j][state_key]
                self.get_gripper_deg(curr_obs, roll_list=gripper_rotation_roll_list, pitch_list=gripper_rotation_pitch_list, yaw_list=gripper_rotation_yaw_list)

        diagnostics.update(create_stats_ordered_dict(state_key + "/gripper_rotation_roll", gripper_rotation_roll_list))
        diagnostics.update(create_stats_ordered_dict(state_key + "/gripper_rotation_pitch", gripper_rotation_pitch_list))
        diagnostics.update(create_stats_ordered_dict(state_key + "/gripper_rotation_yaw", gripper_rotation_yaw_list))

        return diagnostics

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.env_obs_img_dim, self.env_obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        
        if self.downsample:
            im = Image.fromarray(np.uint8(img), 'RGB').resize(self.image_shape, resample=Image.ANTIALIAS)
            img = np.array(im)       
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def get_reward(self, info=None, print_stats=False):
        curr_state = self.get_observation()['state_achieved_goal']
        td_success = self.get_success_metric(curr_state, self.goal_state)
        if print_stats:
            print('-----------------')
            print('Top Drawer: ', td_success)
        reward = td_success
        return reward

    def sample_goals(self):
        self.update_drawer_goal()
        self.update_goal_state()

    def reset(self):
        if self.use_multiple_goals:
            self.test_env_seed = np.random.choice(self.test_env_seeds)
        if self.test_env_seed:
            random.seed(self.test_env_seed)
        if self.expl:
            self.reset_counter += 1
            if self.reset_interval == self.reset_counter:
                self.reset_counter = 0
            else:
                self.sample_goals()
                return self.get_observation()  
        else:
            self.trajectory_done = False

        # Load Environment
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_table()
        self._format_state_query()

        # Sample and load starting positions
        init_pos = np.array(self._pos_init)
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

        top_drawer_pos = self.get_td_handle_pos()

        #(hand_pos, hand_theta, gripper, td_pos)
        #(3, 4, 1, 3)
        observation = np.concatenate((end_effector_pos, hand_theta, gripper_tips_distance, top_drawer_pos))

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
    def sample_object_color(self):
        if np.random.uniform() < self.random_color_p:
            return list(np.random.choice(range(256), size=3) / 255.0) + [1]
        return None

    def get_drawer_handle_future_pos(self, coeff):
        drawer_frame_pos = get_drawer_frame_pos(self._top_drawer)
        return drawer_frame_pos + coeff * np.array([np.sin(self.drawer_yaw * np.pi / 180) , -np.cos(self.drawer_yaw * np.pi / 180), 0])

    def handle_more_open_than_closed(self):
        drawer_handle_close_pos = self.get_drawer_handle_future_pos(td_close_coeff)
        drawer_handle_open_pos = self.get_drawer_handle_future_pos(td_open_coeff)
        drawer_handle_pos = self.get_td_handle_pos()
        return np.linalg.norm(drawer_handle_open_pos - drawer_handle_pos) < np.linalg.norm(drawer_handle_close_pos - drawer_handle_pos)

    def drawer_done(self, curr_pos, goal_pos):
        if curr_pos.size == 0 or goal_pos.size == 0:
            return 0
        else:
            return np.linalg.norm(curr_pos - goal_pos) < self.drawer_thresh

    def done_fn(self, curr_state, goal_state):
        curr_pos = curr_state['state_observation'][8:11]
        goal_pos = goal_state['state_desired_goal'][8:11]
        return self.drawer_done(curr_pos, goal_pos)

    def update_drawer_goal(self):
        # case: full open/close drawer initialization/goal
        if self.full_open_close_init_and_goal:
            # case: close drawer initialization + open drawer goal
            if self.current_goal_is_open:
                td_goal_coeff = td_open_coeff
            # case: open drawer initialization + close drawer goal
            else:
                td_goal_coeff = td_close_coeff
        # case: uniform random drawer initialization/goal
        else:
            if self.handle_more_open_than_closed():
                td_goal_coeff = np.random.uniform(low=(td_close_coeff+td_open_coeff)/2, high=td_open_coeff)
            else:
                td_goal_coeff = np.random.uniform(low=td_close_coeff, high=(td_close_coeff+td_open_coeff)/2)
        
        drawer_handle_goal_pos = self.get_drawer_handle_future_pos(td_goal_coeff)
        drawer_handle_pos = self.get_td_handle_pos()

        # if sampled goal is already achieved by drawer initialization, then set goal to be drawer full open/close
        if self.drawer_done(drawer_handle_pos, drawer_handle_goal_pos):
            assert not self.full_open_close_init_and_goal

            td_goal_coeff = td_close_coeff if self.handle_more_open_than_closed() else td_open_coeff
            drawer_handle_goal_pos = self.get_drawer_handle_future_pos(td_goal_coeff)
        
        self.td_goal_coeff = td_goal_coeff
        self.td_goal = drawer_handle_goal_pos 
        
    def update_goal_state(self):
        self.goal_state = np.concatenate([[0 for _ in range(8)], self.td_goal])

    def get_td_handle_pos(self):
        return np.array(get_drawer_handle_pos(self._top_drawer))

    ### DEMO COLLECTING FUNCTIONS BEYOND THIS POINT ###
    def demo_reset(self):
        self.timestep = 0
        self.grip = -1.
        reset_obs = self.reset()

        #print('----Initial----')
        #self.get_reward(print_stats=True)
        return reset_obs

    def get_demo_action(self, first_timestep=False, final_timestep=False):
        self.final_timestep = final_timestep
        
        if self.drawer_sliding:
            self.td_goal = self.get_drawer_handle_future_pos(self.td_goal_coeff) # update goal in case drawer slides

        action, done = self.move_drawer()

        if self.expl:
            if first_timestep:
                self.trajectory_done = False
                self.gripper_has_been_above = False
                action = np.array([0, 0, 1, 0])
            if done or final_timestep:
                self.trajectory_done = True

            if self.trajectory_done:
                action = np.array([0, 0, 1, 0, -1])
            else:
                action = np.append(action, [self.grip])
                action = np.random.normal(action, self.expert_policy_std)
        else:
            if done:
                #self.get_reward(print_stats=True)
                self.sample_goals()

                action = np.array([0, 0, 1, 0, -1])
            else:
                action = np.append(action, [self.grip])
                action = np.random.normal(action, self.expert_policy_std)

        action = np.clip(action, a_min=-1, a_max=1)
        self.timestep += 1

        return action
    
    def move_drawer(self, print_stages=False):
        self.grip = -1
        ee_pos = self.get_end_effector_pos()
        ee_yaw = self.get_end_effector_theta()[2]

        drawer_handle_pos = self.get_td_handle_pos()
        drawer_frame_pos = get_drawer_frame_pos(self._top_drawer)
        ee_early_stage_goal_pos = drawer_handle_pos - td_offset_coeff * np.array([np.sin((self.drawer_yaw+180) * np.pi / 180) , -np.cos((self.drawer_yaw+180) * np.pi / 180), 0])

        if 0 <= self.drawer_yaw < 90:
            goal_ee_yaw = self.drawer_yaw
        elif 90 <= self.drawer_yaw < 270:
            goal_ee_yaw = self.drawer_yaw - 180
        else:
            goal_ee_yaw = self.drawer_yaw - 360
        
        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(ee_early_stage_goal_pos[:2] - ee_pos[:2]) < .035
        gripper_pos_z_aligned = np.linalg.norm(ee_early_stage_goal_pos[2] - ee_pos[2]) < .0375
        gripper_above = ee_pos[2] >= -0.105
        if not self.gripper_has_been_above and gripper_above:
            self.gripper_has_been_above = True
        done = np.linalg.norm(self.td_goal - drawer_handle_pos) < 0.025

        # Stage 1: if gripper is too low, raise it
        if not self.gripper_has_been_above:
            if print_stages: print('Stage 1')
            action = np.array([0, 0, 1, 0])
        # Stage 2: align gripper yaw
        elif gripper_yaw_aligned:
            if print_stages: print('Stage 2')
            if goal_ee_yaw > ee_yaw:
                action = np.array([0, 0, 0, 1])
            else:
                action = np.array([0, 0, 0, -1])
        # Stage 3: align gripper position with handle position
        elif not gripper_pos_xy_aligned:
            if print_stages: print('Stage 3')
            xy_action = (ee_early_stage_goal_pos - ee_pos) * 6
            action = np.array([xy_action[0], xy_action[1], 0, 0])
        # Stage 4: lower gripper around handle
        elif gripper_pos_xy_aligned and not gripper_pos_z_aligned:
            if print_stages: print('Stage 4')
            xy_action = (ee_early_stage_goal_pos - ee_pos) * 6
            action = np.array([xy_action[0], xy_action[1], xy_action[2]*3, 0])
        # Stage 5: open/close drawer
        else:
            if print_stages: print('Stage 5')
            xy_action = self.td_goal - drawer_handle_pos
            action = 6*np.array([xy_action[0], xy_action[1], 0, 0])
  
        if done:
            action = np.array([0, 0, 1, 0])
        
        # if self.timestep < 25:
        #     action = np.array([1, 1, 1, 0])
        # elif self.timestep < 50:
        #     action = np.array([1, -1, 1, 0])
        # elif self.timestep < 100:
        #     action = np.array([-1, -1, 1, 0])
        # elif self.timestep < 200:
        #     action = np.array([-1, 1, 1, 0])
        # else:
        #     pass

        if self.final_timestep and print_stages: print("drawer_yaw: ", self.drawer_yaw, ", drawer_frame_pos: ", get_drawer_frame_pos(self._top_drawer))
        return action, done