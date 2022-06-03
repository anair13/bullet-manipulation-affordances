from roboverse.envs.sawyer_grasp import SawyerGraspOneEnv
import numpy as np
import roboverse.bullet as bullet

class SawyerReachEnv(SawyerGraspOneEnv):

    def __init__(
        self, 
        *args, 
        traj_ratio=1.0, 
        touch_goal=False, 
        success_threshold=0.05, 
        randomize_reach_z_on_fail=False,
        **kwargs
    ):
        """
        traj_ratio: probability to sample steps towards the true goal or to a false goal during get_demo_action
        touch_goal: reward is based on xyz-distance from object, else xy-distance
        """
        super().__init__(*args, **kwargs)
        self._traj_ratio = traj_ratio
        self._touch_goal = touch_goal
        if touch_goal:
            self._demo_z_offset = np.array([0, 0, 0.05])
        else:
            self._demo_z_offset = np.array([0, 0, 0.1])
        self._success_threshold = success_threshold
        self._pos_low = [0.6, -0.2, -.31]
        self._pos_high = [0.8, 0.2, -0.2]
        self._reset_low = [.65, -0.1, -.31]
        self._reset_high = [.75, 0.1, -.2]
        self._randomize_z_on_fail = randomize_reach_z_on_fail

    def get_reward(self, info):
        if self._reward_type == 'sparse':
            reward = info['reach_success'] - 1.0
        elif self._reward_type == 'shaped':
            reward = -info['reach_distance']
        else:
            raise NotImplementedError

        return reward
    
    def reset(self):
        obs = super().reset()

        obj_pnp = self._objects["lego"]
        self._goal_pos = self.get_object_pos(obj_pnp)
        self._demo_goal = self._goal_pos + self._demo_z_offset
        self._use_false_goal = np.random.random() > self._traj_ratio
        if self._randomize_z_on_fail:
            self._false_goal = np.random.uniform(
                low=self._reset_low, high=self._reset_high
            )
        else:
            self._false_goal = np.random.uniform(
                low=self._object_position_low, high=self._object_position_high
            ) + self._demo_z_offset

        return obs

    def step(self, action):
        observation, reward, done, info = super().step(action)

        obj_pnp = self._objects["lego"]
        self._goal_pos = self.get_object_pos(obj_pnp)

        return observation, reward, done, info
    
    def get_demo_action(self):
        ee_pos = self.get_end_effector_pos()
        if self._use_false_goal:
            target_pos = self._false_goal
        else:
            target_pos = self._demo_goal

        diff = (target_pos - ee_pos) * 3.0 * 2.0
        action = np.zeros((4,))
        action[0] = diff[0]
        action[1] = diff[1]
        action[2] = diff[2]
        self.grip = -1

        return action


    def get_object_pos(self, obj):
        return np.array(bullet.get_body_info(obj, quat_to_deg=False, physicsClientId=self._uid)['pos'])


    def get_info(self):
        info = super().get_info()
        ee_pos = self.get_end_effector_pos()
        target_pos = self._goal_pos + self._demo_z_offset

        diff = (target_pos - ee_pos) * 3.0 * 2.0
        if self._touch_goal:
            dist = np.linalg.norm(diff)
        else:
            dist = np.linalg.norm(diff[:2])
        # import ipdb; ipdb.set_trace()
        info.update({
            'reach_distance': dist,
            'reach_success': int(dist < self._success_threshold)
        })

        return info