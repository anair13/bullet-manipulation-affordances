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
from roboverse.envs.sawyer_affordances_meta_v0 import SawyerAffordancesMetaV0

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=4000)
parser.add_argument("--num_timesteps", type=int, default=75)
parser.add_argument("--subset", type=str, default='train')
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")

args = parser.parse_args()
prefix = "/media/ashvin/data2/s3doodad/ssmrl/bullet/state/v3/"
if not os.path.exists(prefix):
    os.makedirs(prefix)

tasks_file = "/media/ashvin/data2/s3doodad/ssmrl/bullet/state/v2/bullet_100_tasks.pkl"
tasks = pkl.load(open(tasks_file, "rb"))

for task_id in range(len(tasks)):
    # prefix = "/home/ashvin/data/rail-khazatsky/sasha/affordances/combined/"
    demo_data_save_path = prefix + args.name + "_demos_%d.pkl" % task_id
    recon_data_save_path = prefix + args.name + "_images_%d.npy" % task_id
    video_save_path = prefix + args.name + "_video_%d" % task_id
    if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
        os.makedirs(video_save_path)
    print("creating", demo_data_save_path)

    # state_env = roboverse.make('SawyerRigAffordances-v0', random_color_p=0.0)
    state_env = SawyerAffordancesMetaV0(fixed_tasks=tasks)
    state_env.reset_task(task_id)
    print(tasks[task_id])

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
        'object': [],
        'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
    }

    avg_tasks_done = 0
    for j in tqdm(range(args.num_trajectories)):
        env.demo_reset()
        recon_dataset['env'][j, :] = np.uint8(env.render_obs().transpose()).flatten()
        recon_dataset['object'].append(env.curr_object)
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

        images = []
        for i in range(args.num_timesteps):
            img = np.uint8(env.render_obs())
            images.append(Image.fromarray(img))
            recon_dataset['observations'][j, i, :] = img.transpose().flatten()

            observation = env.get_observation()

            action = env.get_demo_action()
            next_observation, reward, done, info = env.step(action)

            trajectory['observations'].append(observation)
            trajectory['actions'][i, :] = action
            trajectory['next_observations'].append(next_observation)
            trajectory['rewards'][i] = reward

        demo_dataset.append(trajectory)
        avg_tasks_done += env.tasks_done

        if args.video_save_frequency > 0 and j % args.video_save_frequency == 0:
            fpath = '{}/{}.gif'.format(video_save_path, j)
            images[0].save(fpath,
                           format='GIF', append_images=images[1:],
                           save_all=True, duration=100, loop=0)

    file = open(demo_data_save_path, 'wb')
    pkl.dump(demo_dataset, file)
    file.close()

    num_datasets += 1
    demo_dataset = []


    print('Success Rate: {}'.format(avg_tasks_done / args.num_trajectories))
    np.save(recon_data_save_path, recon_dataset)
