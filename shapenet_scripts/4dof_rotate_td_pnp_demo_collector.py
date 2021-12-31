import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg
import os
from PIL import Image
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=8000)
parser.add_argument("--num_timesteps", type=int, default=100)
parser.add_argument("--reset_interval", type=int, default=10)
parser.add_argument("--downsample", action='store_true')
parser.add_argument("--drawer_sliding", action='store_true')
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--subset", type=str, default='train')
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")

args = parser.parse_args()
prefix = f"/2tb/home/patrickhaoy/data/affordances/data/{args.name}/"
#prefix = "/2tb/home/patrickhaoy/data/test/" #"/2tb/home/patrickhaoy/data/affordances/combined_new/" #prefix = "/home/ashvin/data/sasha/demos"

if not os.path.exists(prefix):
  os.makedirs(prefix)

# prefix = "/home/ashvin/data/rail-khazatsky/sasha/affordances/combined/"
demo_data_save_path = prefix + args.name + "_demos"
recon_data_save_path = prefix + args.name + "_images"
video_save_path = prefix + args.name + "_video"

kwargs = {
    'drawer_sliding': args.drawer_sliding,
}
if args.downsample:
    kwargs['downsample'] = True
    kwargs['env_obs_img_dim'] = 196

state_env = roboverse.make('SawyerRigAffordances-v2', random_color_p=0.0, expl=True, reset_interval=args.reset_interval, **kwargs)

# FOR TESTING, TURN COLORS OFF
imsize = state_env.obs_img_dim

renderer_kwargs=dict(
        create_image_format='HWC',
        output_image_format='CWH',
        width=imsize,
        height=imsize,
        flatten_image=True,)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)
imlength = env.obs_img_dim * env.obs_img_dim * 3

success = 0
returns = 0
act_dim = env.action_space.shape[0]
num_datasets = 0
demo_dataset = []
recon_dataset = {
    'observations': np.zeros((args.num_trajectories, args.num_timesteps, imlength), dtype=np.uint8),
    'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
}

for j in tqdm(range(args.num_trajectories)):
    env.demo_reset()
    recon_dataset['env'][j, :] = np.uint8(env.render_obs().transpose()).flatten()
    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float),
        'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
        'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
    }
    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        recon_dataset['observations'][j, i, :] = img.transpose().flatten()

        observation = env.get_observation()

        action = env.get_demo_action(first_timestep=(i == 0), final_timestep=(i == args.num_timesteps-1))
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][i, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][i] = reward

    demo_dataset.append(trajectory)

    if ((j + 1) % 500) == 0:
        curr_name = demo_data_save_path + '_{0}.pkl'.format(num_datasets+args.offset)
        file = open(curr_name, 'wb')
        pkl.dump(demo_dataset, file)
        file.close()

        num_datasets += 1
        demo_dataset = []

np.save(recon_data_save_path + '_{0}.npy'.format(args.offset), recon_dataset)
