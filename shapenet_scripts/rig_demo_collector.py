import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=1000)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()
data_save_path = "/home/ashvin/data/sasha/demos/" + args.name + ".pkl"
video_save_path = "/home/ashvin/data/sasha/demos/videos"

#state_env = roboverse.make('SawyerRigMultiobj-v0', DoF=3, object_subset='train')
state_env = roboverse.make('SawyerRigMultiobj-v0', DoF=3, object_subset='train', use_bounding_box=True)

imsize = state_env.obs_img_dim

renderer_kwargs=dict(
        create_image_format='HWC',
        output_image_format='CWH',
        width=imsize,
        height=imsize,
        flatten_image=True,)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)

object_name = 'obj'
num_grasps = 0
image_data = []
act_dim = env.action_space.shape[0]

if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    os.makedirs(video_save_path)

imlength = env.obs_img_dim * env.obs_img_dim * 3

dataset = []

for i in tqdm(range(args.num_trajectories)):
    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float),
        'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
        'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'object_name': env.curr_object,
    }

    env.reset()
    images = []
    for j in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        images.append(Image.fromarray(img))


        ee_pos = env.get_end_effector_pos()
        target_pos = env.get_object_midpoint('obj')
        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.04
        enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
        above = ee_pos[2] > -0.3

        if not aligned and not above:
            #print('Stage 1')
            action = (target_pos - ee_pos) * 3.0
            action[2] = 1
            grip = -1.
        elif not aligned:
            #print('Stage 2')
            action = (target_pos - ee_pos) * 3.0
            action[2] = 0.
            action *= 3.0
            grip = -1.
        elif aligned and not enclosed:
            #print('Stage 3')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            grip = -1.
        elif enclosed and grip < 1:
            #print('Stage 4')
            action = target_pos - ee_pos
            action[2] -= 0.03
            action *= 3.0
            action[2] *= 2.0
            grip += 0.5
        else:
            #print('Stage 5')
            action = np.zeros((3,))
            action[2] = 1.0
            grip = 1.

        action = np.append(action, [grip])
        action = np.random.normal(action, 0.1)
        action = np.clip(action, a_min=-1, a_max=1)

        # ee_pos = env.get_end_effector_pos()
        # target_pos = env.get_object_midpoint(object_name)

        # if j < 25:
        #     action = target_pos - ee_pos
        #     action[2] = 0.
        #     action *= 3.0
        #     grip = -1.
        # elif j < 35:
        #     action = target_pos - ee_pos
        #     action[2] -= 0.03
        #     action *= 3.0
        #     action[2] *= 2.0
        #     grip = -1.
        # elif j < 42:
        #     action = np.zeros((3,))
        #     grip = 0.5
        # else:
        #     action = np.zeros((3,))
        #     action[2] = 1.0
        #     grip = 1.

        # action = np.append(action, [grip])
        # action = np.random.normal(action, 0.1)


        observation = env.get_observation()
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][j, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][j] = reward
    
    num_grasps += info['picked_up']

    if args.video_save_frequency > 0 and i % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(video_save_path, i),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

    dataset.append(trajectory)

print('Success Rate: {}'.format(num_grasps / args.num_trajectories))
file = open(data_save_path, 'wb')
pkl.dump(dataset, file)
file.close()

