import roboverse.bullet as bullet
import numpy as np
import roboverse.utils as utils
from roboverse.envs.widow200_grasp import Widow200GraspEnv
import gym
from roboverse.bullet.misc import load_obj
# from roboverse.utils.shapenet_utils import (
#     load_shapenet_object, import_shapenet_metadata,
#     load_sapien_object, import_sapien_metadata,
# )
import os.path as osp

REWARD_NEGATIVE = -1.0
REWARD_POSITIVE = 10.0

# shapenet_obj_path_map, shapenet_path_scaling_map = import_shapenet_metadata()
# sapien_obj_path_map, sapien_path_scaling_map = import_sapien_metadata()
# shapenet_obj_path_map = dict: object str names --> Shapenet Paths ({class_id}/{object_id})
# shapenet_path_scaling_map = dict: Shapenet Paths ({class_id}/{object_id}) --> scaling factor


class Widow200GraspV2Env(Widow200GraspEnv):
    def __init__(self,
                 *args,
                 observation_mode='state',
                 transpose_image=False,
                 reward_height_threshold=-0.25,
                 num_objects=1,
                 object_names=('beer_bottle',),
                 scaling_local_list=[0.5]*10,
                 reward_type=False,  # Not actually used in grasping envs
                 randomize=True,  # Not actually used
                 target_object=None,
                 **kwargs):

        self._object_position_high = (.82, .075, -.20)
        self._object_position_low = (.78, -.125, -.20)
        self._num_objects = num_objects
        self.object_names = list(object_names)
        self._scaling_local_list = scaling_local_list # converted into dict below.
        self.set_scaling_dicts()
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._reward_type = reward_type
        self.target_object = target_object

        super().__init__(*args, **kwargs)

        self._env_name = 'Widow200GraspV2Env'
        self._height_threshold = -0.31
        self._reward_height_thresh = reward_height_threshold
        self.scripted_traj_len = 25
        self._gripper_open = True

    def set_scaling_dicts(self):
        return
        # assert isinstance(self._scaling_local_list, list), (
        #     "self._scaling_local_list not a list")
        # obj_path_map, path_scaling_map = dict(shapenet_obj_path_map), dict(shapenet_path_scaling_map)
        # obj_path_map.update(sapien_obj_path_map)
        # path_scaling_map.update(sapien_path_scaling_map)

        # self.object_path_dict = dict(
        #     [(obj, path) for obj, path in obj_path_map.items() if obj in self.object_names])
        # self.scaling = dict(
        #     [(path, path_scaling_map[path]) for _, path in self.object_path_dict.items()])
        # self._scaling_local = dict(
        #     [(obj, self._scaling_local_list[i]) for i, obj in enumerate(self.object_names)])

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def _set_action_space(self):
        act_dim = 4
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_spaces(self):
        self._set_action_space()
        # obs = self.reset()
        if self._observation_mode == 'state':
            observation_dim = 7 + 1 + 7*self._num_objects
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        elif self._observation_mode == 'pixels' or self._observation_mode == 'pixels_debug':
            img_space = gym.spaces.Box(0, 1, (self.image_length,), dtype=np.float32)
            if self._observation_mode == 'pixels':
                observation_dim = 7
            elif self._observation_mode == 'pixels_debug':
                observation_dim = 7 + 1 + 7*self._num_objects
            obs_bound = 100
            obs_high = np.ones(observation_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def _load_meshes(self):
        super()._load_meshes()
        if "Drawer" in self._env_name:
            # pass
            self._tray = bullet.objects.widow200_hidden_tray()
            # tray is underneath table so we can get its center.
        else:
            self._tray = bullet.objects.widow200_tray()

        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._robot_id,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        object_positions = self._generate_object_positions()
        self._load_objects(object_positions)

        if "Drawer" in self._env_name:
            self._drawer = bullet.objects.drawer()
            bullet.open_drawer(self._drawer, noisy_open=self.noisily_open_drawer)

            if self.close_drawer_on_reset:
                bullet.close_drawer(self._drawer)

    def _generate_object_positions(self):
        import scipy.spatial
        min_distance_threshold = 0.07
        object_positions = np.random.uniform(
            low=self._object_position_low, high=self._object_position_high)
        object_positions = np.reshape(object_positions, (1,3))
        max_attempts = 100
        i = 0
        while object_positions.shape[0] < self._num_objects:
            i += 1
            object_position_candidate = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
            object_position_candidate = np.reshape(
                object_position_candidate, (1,3))
            min_distance = scipy.spatial.distance.cdist(
                object_position_candidate, object_positions)
            if (min_distance > min_distance_threshold).any():
                object_positions = np.concatenate(
                    (object_positions, object_position_candidate), axis=0)

            if i > max_attempts:
                ValueError('Min distance could not be assured')

        return object_positions

    def _load_object(self, object_name, idx, object_positions):
        return
        # if object_name in shapenet_obj_path_map:
        #     obj = load_shapenet_object(
        #         shapenet_obj_path_map[object_name], self.scaling,
        #         object_positions[idx], scale_local=self._scaling_local[object_name])
        # elif object_name in sapien_obj_path_map:
        #     obj = load_sapien_object(
        #         sapien_obj_path_map[object_name], self.scaling,
        #         object_positions[idx], scale_local=self._scaling_local[object_name])
        # else:
        #     raise NotImplementedError("No such object: {}".format(object_name))
        # return obj


    def _load_objects(self, object_positions):
        assert len(self.object_names) >= self._num_objects
        import random
        indexes = list(range(self._num_objects))
        random.shuffle(indexes)

        for idx in indexes:
            object_name = self.object_names[idx]
            self._objects[object_name] = self._load_object(
                object_name, idx, object_positions)
            for _ in range(10):
                bullet.step()

    def step(self, action):
        action = np.asarray(action)
        pos = list(bullet.get_link_state(self._robot_id, self._end_effector,
                                         'pos'))
        delta_pos = action[:3]
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        theta = list(bullet.get_link_state(self._robot_id, self._end_effector,
                                           'theta'))
        target_theta = theta

        delta_theta = action[3]
        target_theta = np.clip(target_theta, [0, 85, 137], [180, 85, 137])
        target_theta = bullet.deg_to_quat(target_theta)
        gripper = -0.8

        self._simulate(pos, target_theta, gripper, delta_theta=delta_theta)

        pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        if pos[2] < self._height_threshold:
            gripper = 0.8
            for _ in range(5):
                self._simulate(pos, target_theta, gripper, delta_theta=0)
            for _ in range(5):
                pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
                pos = list(pos)
                pos = np.clip(pos, self._pos_low, self._pos_high)
                pos[2] += 0.05
                self._simulate(pos, target_theta, gripper, delta_theta=0)
            done = True
            reward = self.get_reward({})
            for obj_name in self._objects.keys():
                object_info = bullet.get_body_info(self._objects[obj_name],
                                                   quat_to_deg=False)
                object_pos = np.asarray(object_info['pos'])
                object_height = object_pos[2]
                print('-------------------')
                print('{} height: {}'.format(obj_name, object_height))
            if reward > 0:
                info = {'grasp_success': 1.0}
            else:
                info = {'grasp_success': 0.0}
        else:
            done = False
            reward = REWARD_NEGATIVE
            info = {'grasp_success': 0.0}

        observation = self.get_observation()
        self._prev_pos = bullet.get_link_state(self._robot_id, self._end_effector, 'pos')
        return observation, reward, done, info

    def get_obj_obs_array(self):
        obj_obs = None
        object_list = self._objects.keys()
        for object_name in object_list:
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = object_info['pos']
            object_theta = object_info['theta']
            if obj_obs is None:
                obj_obs = np.concatenate((object_pos, object_theta))
            else:
                obj_obs = np.concatenate(
                    (obj_obs, object_pos, object_theta))
        return obj_obs

    def get_gripper_tips_distance(self):
        left_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[0], keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._robot_id, self._gripper_joint_name[1], keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)

        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]

        return gripper_tips_distance

    def get_observation(self):
        gripper_tips_distance = self.get_gripper_tips_distance()
        end_effector_pos = self.get_end_effector_pos()
        end_effector_theta = bullet.get_link_state(
            self._robot_id, self._end_effector, 'theta', quat_to_deg=False)

        if self._observation_mode == 'state':
            observation = np.concatenate(
                (end_effector_pos, end_effector_theta, gripper_tips_distance))

            observation = np.concatenate(observation, self.get_obj_obs_array())

        elif self._observation_mode == 'pixels':
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            # image_observation = np.zeros((48, 48, 3), dtype=np.uint8)
            observation = {
                'state': np.concatenate(
                    (end_effector_pos, gripper_tips_distance)),
                'image': image_observation
            }
        elif self._observation_mode == 'pixels_debug':
            # This mode passes in all the true state information + images
            image_observation = self.render_obs()
            image_observation = np.float32(image_observation.flatten())/255.0
            state_observation = np.concatenate(
                (end_effector_pos, end_effector_theta, gripper_tips_distance))

            state_observation = np.concatenate(
                (state_observation, self.get_obj_obs_array()))

            observation = {
                'state': state_observation,
                'image': image_observation,
            }
        else:
            raise NotImplementedError

        return observation

    def get_info(self):
        info = {'end_effector_pos': self.get_end_effector_pos()}
        object_list = self._objects.keys()
        for object_name in object_list:
             object_info = bullet.get_body_info(self._objects[object_name],
                                                    quat_to_deg=False)
             object_pos = object_info['pos']
             info["object" + str(object_name)] = object_pos

        return info

    def get_reward(self, info):
        object_list = self._objects.keys()
        reward = REWARD_NEGATIVE
        for object_name in object_list:
            object_info = bullet.get_body_info(self._objects[object_name],
                                               quat_to_deg=False)
            object_pos = np.asarray(object_info['pos'])
            object_height = object_pos[2]
            if object_height > self._reward_height_thresh:
                end_effector_pos = np.asarray(self.get_end_effector_pos())
                object_gripper_distance = np.linalg.norm(
                    object_pos - end_effector_pos)
                if object_gripper_distance < 0.1:
                    reward = REWARD_POSITIVE
        return reward


if __name__ == "__main__":
    import roboverse
    import time

    save_video = True
    images = []

    num_objects_x = 1
    env = roboverse.make("Widow200GraspV2-v0",
                         gui=True, observation_mode='pixels_debug')
    obs = env.reset()
    # object_ind = np.random.randint(0, env._num_objects)
    object_ind = num_objects_x - 1
    i = 0
    xy_dist_thresh = 0.02
    action = env.action_space.sample()
    for _ in range(500):
        time.sleep(0.1)
        object_pos = obs['state'][8: 8 + 3]
        # print("obs", obs)
        # print("object_pos", object_pos)
        ee_pos = obs['state'][:3]

        xyz_delta = object_pos - ee_pos
        xy_diff = np.linalg.norm(xyz_delta[:2])
        action = xyz_delta
        if xy_diff > xy_dist_thresh:
            action[2] = 0 # prevent downward motion if too far.
        action = action*7.0
        action += np.random.normal(scale=0.1, size=(3,))

        # action = np.array([0, 0, 0])
        theta_action = +0.0
        action = np.concatenate((action, np.asarray([theta_action])))
        print('action', action)
        obs, rew, done, info = env.step(action)

        img = env.render()
        if save_video:
            images.append(img)

        # print("info", env.get_info())
        i+=1
        if done or i > env.scripted_traj_len:
            # object_ind = np.random.randint(0, env._num_objects)
            object_ind = num_objects_x - 1
            obs = env.reset()
            i = 0
            print('Reward: {}'.format(rew))
        # print(obs)