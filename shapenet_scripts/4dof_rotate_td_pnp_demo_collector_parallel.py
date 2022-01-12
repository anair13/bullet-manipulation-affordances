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
from multiprocess import Pool
import gc

def collect(id):
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
        'observations': np.zeros((args.num_trajectories_per_demo, args.num_timesteps, imlength), dtype=np.uint8),
        'env': np.zeros((args.num_trajectories_per_demo, imlength), dtype=np.uint8),
        'skill_id': np.zeros((args.num_trajectories_per_demo, ), dtype=np.uint8)
    }

    for j in tqdm(range(args.num_trajectories_per_demo)):
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
            'skill_id': 0,
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
            trajectory['skill_id'] = info['skill_id']
            recon_dataset['skill_id'][j] = info['skill_id']

        demo_dataset.append(trajectory)

        if ((j + 1) % args.num_trajectories_per_demo) == 0:
            curr_name = demo_data_save_path + '_{0}.pkl'.format(id+args.demo_offset)
            file = open(curr_name, 'wb')
            pkl.dump(demo_dataset, file)
            file.close()

            del demo_dataset
            gc.collect()
            demo_dataset = []

            np.save(prefix + args.name + "_images_{0}.npy".format(id + args.demo_offset), recon_dataset)
            del recon_dataset
            gc.collect()
            recon_dataset = []

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--num_trajectories", type=int, default=4000)
    parser.add_argument("--num_trajectories_per_demo", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--num_timesteps", type=int, default=150)
    parser.add_argument("--reset_interval", type=int, default=5)
    parser.add_argument("--downsample", action='store_true')
    parser.add_argument("--drawer_sliding", action='store_true')
    parser.add_argument("--demo_offset", type=int, default=0)
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
        'expert_policy_std': 0.05,
        # 'fixed_drawer_yaw': 24.18556394023222,
        # 'fixed_drawer_pos': np.array([0.50850424, 0.11416014, -0.34]),
        'use_single_obj_idx': 1,
        'demo_num_ts': args.num_timesteps
    }
    if args.downsample:
        kwargs['downsample'] = True
        kwargs['env_obs_img_dim'] = 196

    skill_to_id: {
        'open_drawer': 0,
        'close_drawer': 1,
        'pnp_to_top_of_drawer': 2,
        'pnp_to_inside_drawer': 3,
        'pnp_to_outside_drawer': 4,
    }

    assert args.num_trajectories % args.num_trajectories_per_demo == 0

    pool = Pool(args.num_threads)
    ids = [id for id in range(args.num_trajectories // args.num_trajectories_per_demo)]
    results = pool.map(collect, ids)
    pool.close()
    pool.join()