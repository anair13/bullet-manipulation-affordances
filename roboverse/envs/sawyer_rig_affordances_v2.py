import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.bullet.control import get_object_position
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

# Constants
td_close_coeff = 0.20567612 #0.13754340000000412 #0.21567452 #0.13754340000000412
td_open_coeff = 0.29387810000002523
td_offset_coeff = 0.001

gripper_bounding_x = [.5, .8] #[.46, .84] #[0.4704, 0.8581]
gripper_bounding_y = [-.17, .17] #[-0.1989, 0.2071]

tasks = [
    "open_drawer", "close_drawer", 
    "move_object_top_to_out", "move_object_out_to_top",
    "move_object_in_to_out", "move_object_out_to_in",
    "move_object_top_to_in", "move_object_in_to_out",
]

class SawyerRigAffordancesV2(SawyerBaseEnv):

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

        self.obj_thresh = 0.08
        self.drawer_thresh = 0.065
        self.gripper_pos_thresh = 0.08
        self.gripper_rot_thresh = 10

        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self.obs_img_dim = obs_img_dim

        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[0.7, 0, -0.25], distance=0.425,
            yaw=90, pitch=-27, roll=0, up_axis_index=2)

        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        self.dt = 0.1

        self.expert_policy_std = kwargs.pop('expert_policy_std', 0.1)

        # Tasks
        self.task_dict = {
            'move_drawer': lambda : self.move_drawer(),
            'move_obj_pnp': lambda: self.move_obj_pnp(),
            'move_obj_slide': lambda: self.move_obj_slide(),
        }
        self.curr_task = 'drawer'

        # Reset-free
        self.reset_interval = kwargs.pop('reset_interval', 1)
        self.reset_counter = self.reset_interval-1
        self.expl = kwargs.pop('expl', False)
        self.trajectory_done = False
        self.final_timestep = False
        self.drawer_sliding = kwargs.pop('drawer_sliding', False)

        # Drawer
        self.gripper_has_been_above = False

        # Anti-aliasing
        self.downsample = kwargs.pop('downsample', False)
        self.env_obs_img_dim = kwargs.pop('env_obs_img_dim', self.obs_img_dim)

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

    def _set_spaces(self):
        act_dim = self.DoF + 1
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 20
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
        self._wall = bullet.objects.wall_narrow(scale=1.0)

        # self._debug1 = bullet.objects.button(pos=[gripper_bounding_x[0], gripper_bounding_y[0], -.35])
        # self._debug2 = bullet.objects.button(pos=[gripper_bounding_x[1], gripper_bounding_y[1], -.35])

        ## Top Drawer
        if self.test_env and not self.test_env_seed:
            self.drawer_yaw = 180
            drawer_frame_pos = np.array([.6, -.19, -.34])
        else:
            self.drawer_yaw = random.uniform(0, 180)
   
            tries = 0
            while(True):
                drawer_frame_pos = np.array([random.uniform(gripper_bounding_x[0], gripper_bounding_x[1]), random.uniform(gripper_bounding_y[0], gripper_bounding_y[1]), -.34])
                drawer_handle_open_goal_pos = drawer_frame_pos + td_open_coeff * np.array([np.sin(self.drawer_yaw * np.pi / 180) , -np.cos(self.drawer_yaw * np.pi / 180), 0])
                if gripper_bounding_x[0] <= drawer_handle_open_goal_pos[0] <= gripper_bounding_x[1] \
                    and gripper_bounding_y[0] <= drawer_handle_open_goal_pos[1] <= gripper_bounding_y[1]:
                    break
                tries += 1
                if (tries > 25):
                    self.drawer_yaw = random.uniform(0, 180)
        quat = deg_to_quat([0, 0, self.drawer_yaw])
        
        # For debugging: hardcode drawer_yaw and drawer_frame_pos
        # self.drawer_yaw = 160.10009720998795
        # rot = Rotation.from_euler('xyz', [0, 0, self.drawer_yaw], degrees=True)
        # quat = rot.as_quat()
        # drawer_frame_pos = np.array(list((0.6351875030272269, -0.10053104435094253, -0.34)))
        #drawer_yaw:  160.10009720998795 , drawer_frame_pos:  (0.6351875030272269, -0.10053104435094253, -0.34)

        # drawer_frame_pos = np.array([gripper_bounding_x[1], gripper_bounding_y[0], -.34])
        # self.drawer_yaw = 180
        # quat = deg_to_quat([0, 0, self.drawer_yaw])

        if self.drawer_sliding:
            self._top_drawer = bullet.objects.drawer_sliding_lightblue_base(quat=quat, pos=drawer_frame_pos, rgba=self.sample_object_color())
        else:
            self._top_drawer = bullet.objects.drawer_lightblue_base(quat=quat, pos=drawer_frame_pos, rgba=self.sample_object_color())
        
        open_drawer(self._top_drawer, 100)

        self.init_handle_pos = get_drawer_handle_pos(self._top_drawer)[1]

        ## Objects
        objects_within_gripper_range = False
        self._objs = []
        tries = 0
        while(not objects_within_gripper_range):
            for obj in self._objs:
                p.removeBody(obj)
            self._objs = []

            self.get_obj_pnp_goals()
            for rgba, pos in zip([[0.93, .294, .169, 1], [.5, 1., 0., 1], [0., .502, .502, 1]], [self.on_top_drawer_goal, self.in_drawer_goal, self.out_of_drawer_goal]):
                if np.random.uniform() < .5:
                    pos = self.out_of_drawer_goal + np.array([0, 0, 0.5])
                    self.get_obj_pnp_goals()
                self._objs.append(self.spawn_object(object_position=pos, rgba=rgba))
            
            objects_within_gripper_range = True
            for obj in self._objs:
                pos, _ = get_object_position(obj)
                if not (gripper_bounding_x[0] - .04 <= pos[0] and pos[0] <= gripper_bounding_x[1] + .04 \
                    and gripper_bounding_y[0] - .04 <= pos[1] and pos[1] <= gripper_bounding_y[1] + .04):
                    objects_within_gripper_range = False
                    break
                    
            tries += 1
            if tries > 10:
                break

        # Tray acts as stopper for drawer closing
        #self.debug1 = bullet.objects.button(pos=self.get_drawer_handle_future_pos(-.05) + np.array([0, 0, .1]))
        tray_pos = self.get_drawer_handle_future_pos(-.01)
        self._tray = bullet.objects.tray(quat=quat, pos=tray_pos, scale=0.0001)
        if np.random.uniform() < .5:
            close_drawer(self._top_drawer, 200)
        #close_drawer(self._top_drawer, np.random.random_integers(low=1, high=80))

        # _debug1_pos = list(get_drawer_frame_pos(self._top_drawer))
        # _debug1_pos[2] += .1
        # _debug1_pos[0] += .2
        # self._debug1 = bullet.objects.button(pos=_debug1_pos)
        
        # _debug2_pos = list(get_drawer_bottom_pos(self._top_drawer))
        # _debug2_pos[2] += 0
        # self._debug2 = bullet.objects.button(pos=_debug2_pos)

        self._workspace = bullet.Sensor(self._sawyer,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._sawyer, 'link_name', 'gripper_site')
    
    def sample_quat(self):
        return deg_to_quat(np.random.randint(0, 360, size=3))

    def spawn_object(self, object_position=None, quat=None, rgba=[0, 1, 0, 1]):
        # Pick object if necessary and save information
        if object_position is None:
            object_position = self.sample_object_location()
        if object_position is None:
            return

        # Generate quaterion if none is given
        if quat is None:
            quat = self.sample_quat()

        obj = bullet.objects.drawer_lego(pos=object_position, quat=quat, rgba=rgba, scale=1.4)

        # Allow the objects to land softly in low gravity
        p.setGravity(0, 0, -1)
        for _ in range(100):
            bullet.step()
        # After landing, bring to stop
        p.setGravity(0, 0, -10)
        for _ in range(100):
            bullet.step()
        
        return obj
    
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
        elif key == 'obj_pnp':
            curr_pos_0 = curr_state[11:14]
            goal_pos_0 = goal_state[11:14]
            curr_pos_1 = curr_state[14:17]
            goal_pos_1 = goal_state[14:17]
            curr_pos_2 = curr_state[17:20]
            goal_pos_2 = goal_state[17:20]
            success = int(self.obj_pnp_done(curr_pos_0, goal_pos_0)) \
                and int(self.obj_pnp_done(curr_pos_1, goal_pos_1)) \
                and int(self.obj_pnp_done(curr_pos_2, goal_pos_2))
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
        elif key == 'obj_pnp':
            curr_pos_0 = curr_state[11:14]
            goal_pos_0 = goal_state[11:14]
            curr_pos_1 = curr_state[14:17]
            goal_pos_1 = goal_state[14:17]
            curr_pos_2 = curr_state[17:20]
            goal_pos_2 = goal_state[17:20]
            success = np.linalg.norm(curr_pos_0 - goal_pos_0) \
                and np.linalg.norm(curr_pos_1 - goal_pos_1) \
                and np.linalg.norm(curr_pos_2 - goal_pos_2)
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

        success_keys = ["top_drawer", "obj_pnp", "gripper_position", "gripper_rotation_roll", "gripper_rotation_pitch", "gripper_rotation_yaw", "gripper_rotation", "gripper"]
        distance_keys = ["top_drawer", "obj_pnp", "gripper_position", "gripper_rotation_roll", "gripper_rotation_pitch", "gripper_rotation_yaw", "gripper_rotation"]

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
        r = np.random.uniform()
        if r < 1/2:
            self.update_obj_pnp_goal()
            task = 'move_obj_pnp'
        # elif r < 2/3:
        #     self.update_obj_slide_goal()
        #     task = 'move_obj_slide'
        else:
            self.update_drawer_goal()
            task = 'move_drawer'
        
        #self.update_goal_state()
        self.goal_state = np.ones((11,))
        return task
        

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
                self.curr_task = self.sample_goals()
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
        self.curr_task = self.sample_goals()

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

        obj0_pos = self.get_position_of_object_idx(0)
        obj1_pos = self.get_position_of_object_idx(1)
        obj2_pos = self.get_position_of_object_idx(2)

        #(hand_pos, hand_theta, gripper, td_pos, obj0_pos, obj1_pos, obj2_pos)
        #(3, 4, 1, 3, 3, 3, 3)
        observation = np.concatenate((
            end_effector_pos, hand_theta, gripper_tips_distance, 
            top_drawer_pos,
            obj0_pos, obj1_pos, obj2_pos,
        ))

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
    
    def obj_pnp_done(self, curr_pos, goal_pos):
        if curr_pos.size == 0 or goal_pos.size == 0:
            return 0
        else:
            return np.linalg.norm(curr_pos - goal_pos) < self.obj_thresh

    def done_fn(self, curr_state, goal_state):
        curr_pos = curr_state['state_observation'][8:11]
        goal_pos = goal_state['state_desired_goal'][8:11]
        return self.drawer_done(curr_pos, goal_pos)

    def get_obj_pnp_goals(self):
        self.on_top_drawer_goal = np.array(list(get_drawer_frame_pos(self._top_drawer)))
        self.on_top_drawer_goal[2] += .1
        self.in_drawer_goal = np.array(list(get_drawer_bottom_pos(self._top_drawer)))

        self.out_of_drawer_goal = None
        while self.out_of_drawer_goal is None:
            offset = 0.0
            out_of_drawer_goal = np.array([random.uniform(gripper_bounding_x[0] + offset, gripper_bounding_x[1] - offset), random.uniform(gripper_bounding_y[0] + offset, gripper_bounding_y[1] - offset), -0.34])
            drawer_frame_far = np.linalg.norm(out_of_drawer_goal[:2] - self.on_top_drawer_goal[:2]) > 0.1
            drawer_base_far = np.linalg.norm(out_of_drawer_goal[:2] - self.in_drawer_goal[:2]) > 0.2
            #self._debug1 = bullet.objects.button(pos=self.on_top_drawer_goal + np.array([0, .1, 0]))
            if drawer_frame_far and drawer_base_far:
                self.out_of_drawer_goal = out_of_drawer_goal

    ## Outdated
    def get_obj_slide_goals(self):
        self.get_obj_pnp_goals()

        self.obj_slide_goal = None
        while self.obj_slide_goal is None:
            offset = .025
            obj_slide_goal = np.array([random.uniform(gripper_bounding_x[0] + offset, gripper_bounding_x[1] - offset), random.uniform(gripper_bounding_y[0] + offset, gripper_bounding_y[1] - offset), -0.34])
            drawer_frame_far = np.linalg.norm(obj_slide_goal[:2] - self.on_top_drawer_goal[:2]) > 0
            drawer_base_far = np.linalg.norm(obj_slide_goal[:2] - self.in_drawer_goal[:2]) > .2
            obj_far = np.linalg.norm(obj_slide_goal[:2] - self.get_object_pos(self.obj_slide)[:2]) > .1
            if drawer_frame_far and drawer_base_far and obj_far:
                self.obj_slide_goal = obj_slide_goal

    def get_drawer_objs(self):
        obj_in_drawer = None
        for obj in self._objs:
            obj_pos = self.get_object_pos(obj)
            if np.linalg.norm(self.in_drawer_goal - obj_pos) < self.obj_thresh:
                obj_in_drawer = obj

        obj_on_drawer = None
        for obj in self._objs:
            obj_pos = self.get_object_pos(obj)
            if np.linalg.norm(self.on_top_drawer_goal - obj_pos) < self.obj_thresh:
                obj_on_drawer = obj
        
        return obj_in_drawer, obj_on_drawer

    def update_obj_pnp_goal(self):
        self.get_obj_pnp_goals()

        obj_in_drawer, obj_on_drawer = self.get_drawer_objs()

        obj_to_be_in_drawer = set()
        obj_to_be_on_drawer = set()
        obj_to_be_out_of_drawer = set()
        
        if obj_on_drawer:
            obj_to_be_out_of_drawer.add(obj_on_drawer)
        else:
            obj_to_be_on_drawer = set(self._objs)
        
        if self.handle_more_open_than_closed(): #np.linalg.norm(self.get_drawer_handle_future_pos(td_open_coeff) - self.get_td_handle_pos()) < .1:
            if obj_in_drawer:
                obj_to_be_out_of_drawer.add(obj_in_drawer)
            else:
                obj_to_be_in_drawer = set(self._objs)
        else:
            obj_to_be_on_drawer.discard(obj_in_drawer)
        
        possible_goals = [
            (self.in_drawer_goal, list(obj_to_be_in_drawer)),
            (self.on_top_drawer_goal, list(obj_to_be_on_drawer)),
            (self.out_of_drawer_goal, list(obj_to_be_out_of_drawer)),
        ]
        np.random.shuffle(possible_goals)

        for (goal, can_interact_objs) in possible_goals:
            if len(can_interact_objs) != 0:
                self.obj_pnp = np.random.choice(can_interact_objs)
                self.obj_pnp_goal = goal
        
        self.goal_ee_yaw = np.random.uniform(-90, 90)
    
    def update_obj_slide_goal(self):
        obj_in_drawer, obj_on_drawer = self.get_drawer_objs()
        can_interact_objs = set(self._objs)
        can_interact_objs.discard(obj_in_drawer)
        can_interact_objs.discard(obj_on_drawer)
        
        self.obj_slide = np.random.choice(list(can_interact_objs))
        self.get_obj_slide_goals()

        #self._debug1 = bullet.objects.button(pos=self.obj_slide_goal + np.array([0, 0, .2]))

    def update_drawer_goal(self):
        # if self.handle_more_open_than_closed():
        #     td_goal_coeff = np.random.uniform(low=td_close_coeff, high=(td_close_coeff+td_open_coeff)/2)
        # else:
        #     td_goal_coeff = np.random.uniform(low=(td_close_coeff+td_open_coeff)/2, high=td_open_coeff)
        
        # drawer_handle_goal_pos = self.get_drawer_handle_future_pos(td_goal_coeff)
        # drawer_handle_pos = self.get_td_handle_pos()

        # # if sampled goal is already achieved by drawer initialization, then set goal to be drawer full open/close
        # if self.drawer_done(drawer_handle_pos, drawer_handle_goal_pos):
        #     td_goal_coeff = td_close_coeff if self.handle_more_open_than_closed() else td_open_coeff
        #     drawer_handle_goal_pos = self.get_drawer_handle_future_pos(td_goal_coeff)

        td_goal_coeff = td_close_coeff if self.handle_more_open_than_closed() else td_open_coeff
        drawer_handle_goal_pos = self.get_drawer_handle_future_pos(td_goal_coeff)
        
        self.td_goal_coeff = td_goal_coeff
        self.td_goal = drawer_handle_goal_pos 
        
    def update_goal_state(self):
        self.goal_state = np.concatenate([[0 for _ in range(8)], self.td_goal])

    def get_td_handle_pos(self):
        return np.array(get_drawer_handle_pos(self._top_drawer))
    
    def get_position_of_object_idx(self, idx):
        pos, _ = get_object_position(self._objs[idx])
        return pos

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

        action, done = self.task_dict[self.curr_task]()

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
        #print((drawer_handle_pos - drawer_frame_pos)/np.array([np.sin((self.drawer_yaw+180) * np.pi / 180) , -np.cos((self.drawer_yaw+180) * np.pi / 180), 0]))
        ee_early_stage_goal_pos = drawer_handle_pos - td_offset_coeff * np.array([np.sin((self.drawer_yaw+180) * np.pi / 180) , -np.cos((self.drawer_yaw+180) * np.pi / 180), 0])

        if 0 <= self.drawer_yaw < 90:
            goal_ee_yaw = self.drawer_yaw
        elif 90 <= self.drawer_yaw < 270:
            goal_ee_yaw = self.drawer_yaw - 180
        else:
            goal_ee_yaw = self.drawer_yaw - 360
        
        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(ee_early_stage_goal_pos[:2] - ee_pos[:2]) < .02
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

    def move_obj_pnp(self, print_stages=True):
        ee_pos = self.get_end_effector_pos()
        target_pos = self.get_object_pos(self.obj_pnp)
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.025
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
        done_xy = np.linalg.norm(target_pos[:2] - self.obj_pnp_goal[:2]) < 0.05
        done = done_xy and np.linalg.norm(target_pos[2] - self.obj_pnp_goal[2]) < 0.03
        above = ee_pos[2] >= -0.125

        ee_yaw = self.get_end_effector_theta()[2]
        goal_ee_yaw = self.goal_ee_yaw
        # if 0 <= self.drawer_yaw < 90:
        #     goal_ee_yaw = self.drawer_yaw - 90
        # elif 90 <= self.drawer_yaw < 270:
        #     goal_ee_yaw = self.drawer_yaw - 90
        # else:
        #     goal_ee_yaw = self.drawer_yaw - 360 + 90
        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 5

        if not aligned and not above:
            if print_stages: print('Stage 1')
            action = np.array([0.,0., 1., 0.])
            self.grip = -1.
        elif gripper_yaw_aligned:
            if print_stages: print('Stage 2')
            if goal_ee_yaw > ee_yaw:
                action = np.array([0, 0, 0, 1])
            else:
                action = np.array([0, 0, 0, -1])
        elif not aligned:
            if print_stages: print('Stage 3')
            diff = (target_pos - ee_pos) * 3.0
            action = np.array([diff[0], diff[1], 0., 0.])
            action *= 2.0
            self.grip = -1.
        elif aligned and not enclosed and self.grip < 1:
            if print_stages: print('Stage 4')
            diff = target_pos - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 1.5
            self.grip = -1.
        elif enclosed and self.grip < 1:
            if print_stages: print('Stage 5')
            diff = target_pos - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            self.grip += 0.5
        elif not above:
            if print_stages: print('Stage 6')
            action = np.array([0.,0., 1., 0.])
            self.grip = 1.
        elif not done_xy:
            if print_stages: print('Stage 7')
            diff = self.obj_pnp_goal - ee_pos
            action = np.array([diff[0], diff[1], diff[2], 0.])
            action[2] = 0
            action *= 3.0
            self.grip = 1.
        else:
            if print_stages: print('Stage 8')
            action = np.array([0.,0.,0., 0.])
            self.grip = -1
        
        # print(aligned, above, done, enclosed, self.grip)
        # print(target_pos, ee_pos, goal)

        return action, done

    def move_obj_slide(self, print_stages=False):
        ee_pos = self.get_end_effector_pos()
        ee_yaw = self.get_end_effector_theta()[2]
        obj_pos = self.get_object_pos(self.obj_slide)
        goal_pos = self.obj_slide_goal

        #import pdb; pdb.set_trace()
        vec = goal_pos[:2] - obj_pos[:2]
        direction = (np.arctan2(vec[1], vec[0]) * 180 / np.pi + 360 + 90) % 360
        if 0 <= direction < 90:
            goal_ee_yaw = direction
        elif 90 <= direction < 270:
            goal_ee_yaw = direction - 180
        else:
            goal_ee_yaw = direction - 360
        ee_early_stage_goal_pos = obj_pos - 0.1 * np.array([np.sin(direction * np.pi / 180) , -np.cos(direction * np.pi / 180), 0])

        gripper_yaw_aligned = np.linalg.norm(goal_ee_yaw - ee_yaw) > 5
        gripper_pos_xy_aligned = np.linalg.norm(ee_early_stage_goal_pos[:2] - ee_pos[:2]) < .035
        gripper_pos_z_aligned = np.linalg.norm(ee_early_stage_goal_pos[2] - ee_pos[2]) < .0375
        gripper_above = ee_pos[2] >= -0.105
        if not self.gripper_has_been_above and gripper_above:
            self.gripper_has_been_above = True

        done_xy = np.linalg.norm(obj_pos[:2] - goal_pos[:2]) < 0.05
        done = done_xy and np.linalg.norm(obj_pos[2] - goal_pos[2]) < 0.03

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
            xy_action = goal_pos - obj_pos
            action = 6*np.array([xy_action[0], xy_action[1], 0, 0])
  
        if done:
            action = np.array([0, 0, 1, 0])

        return action, done

    def get_object_pos(self, obj):
        return np.array(bullet.get_body_info(obj, quat_to_deg=False)['pos'])