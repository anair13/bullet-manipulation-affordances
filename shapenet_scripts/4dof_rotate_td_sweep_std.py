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
parser.add_argument("--num_timesteps", type=int, default=75)
parser.add_argument("--reset_interval", type=int, default=10)
parser.add_argument("--downsample", action='store_true')
parser.add_argument("--drawer_sliding", action='store_true')

args = parser.parse_args()
#prefix = "/2tb/home/patrickhaoy/data/affordances/data/antialias_reset_free_v5_rotated_top_drawer/"
prefix = "/2tb/home/patrickhaoy/data/test/" #"/2tb/home/patrickhaoy/data/affordances/combined_new/" #prefix = "/home/ashvin/data/sasha/demos"
video_save_path = prefix + args.name + "_video"

for std in [0.6, 0.7, 0.8, 0.9]:
    kwargs = {}
    if args.downsample:
        kwargs['downsample'] = True
        kwargs['env_obs_img_dim'] = 196
    if args.drawer_sliding:
        kwargs['drawer_sliding'] = True
    kwargs['expert_policy_std'] = std
    state_env = roboverse.make('SawyerRigAffordances-v1', random_color_p=0.0, expl=True, reset_interval=args.reset_interval, **kwargs)

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
        'observations': np.zeros((args.num_trajectories, args.num_timesteps, 48, 48, 3), dtype=np.uint8),
    }

    for j in tqdm(range(args.num_trajectories)):
        env.demo_reset()
        for i in range(args.num_timesteps):
            img = np.uint8(env.render_obs())
            recon_dataset['observations'][j, i, :] = img

            observation = env.get_observation()

            action = env.get_demo_action(first_timestep=(i == 0), final_timestep=(i == args.num_timesteps-1))
            next_observation, reward, done, info = env.step(action)

    recon_dataset['observations'] = recon_dataset['observations'].reshape((args.num_trajectories*args.num_timesteps, 48, 48, 3))
    images = [Image.fromarray(img) for img in recon_dataset['observations']]
    fpath = '{}/{}.gif'.format(video_save_path, std)
    images[0].save(fpath,
                format='GIF', append_images=images[1:],
                save_all=True, duration=10, loop=0)
    print("saved", fpath)
