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
parser.add_argument("--num_trajectories", type=int, default=200)
parser.add_argument("--num_timesteps", type=int, default=75)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")

args = parser.parse_args()

prefix = "/2tb/home/patrickhaoy/data/affordances/combined_new/" #"/global/scratch/users/patrickhaoy/s3doodad/affordances/combined_new"
demo_data_save_path = prefix + args.name + "_demos"
recon_data_save_path = prefix + args.name + "_images.npy"

state_env = roboverse.make('SawyerRigMultiobj-v0', object_subset='test')

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

    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        recon_dataset['observations'][j, i, :] = img.transpose().flatten()

        observation = env.get_observation()

        action = env.get_demo_action()
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][i, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][i] = reward

    demo_dataset.append(trajectory)
    avg_tasks_done += env.done

    if ((j + 1) % 500) == 0:
        curr_name = demo_data_save_path + '_{0}.pkl'.format(num_datasets)
        file = open(curr_name, 'wb')
        pkl.dump(demo_dataset, file)
        file.close()

        num_datasets += 1
        demo_dataset = []


print('Success Rate: {}'.format(avg_tasks_done / args.num_trajectories))
np.save(recon_data_save_path, recon_dataset)








# import roboverse
# import numpy as np
# import pickle as pkl
# from tqdm import tqdm
# from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
# from roboverse.bullet.misc import quat_to_deg 
# import os
# from PIL import Image
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--name", type=str)
# parser.add_argument("--num_trajectories", type=int, default=1000)
# parser.add_argument("--num_timesteps", type=int, default=50)
# parser.add_argument("--subset", type=str, default='train')
# parser.add_argument("--video_save_frequency", type=int,
#                     default=0, help="Set to zero for no video saving")

# args = parser.parse_args()
# demo_data_save_path = "/home/ashvin/data/sasha/demos/gr_" + args.name + "_demos"
# recon_data_save_path = "/home/ashvin/data/sasha/demos/gr_" + args.name + "_images.npy"
# video_save_path = "/home/ashvin/data/sasha/demos/videos"

# state_env = roboverse.make('SawyerRigMultiobj-v0', DoF=3, object_subset=args.subset)
# imsize = state_env.obs_img_dim

# renderer_kwargs=dict(
#         create_image_format='HWC',
#         output_image_format='CWH',
#         width=imsize,
#         height=imsize,
#         flatten_image=True,)

# renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
# env = InsertImageEnv(state_env, renderer=renderer)
# imlength = env.obs_img_dim * env.obs_img_dim * 3

# success = 0
# returns = 0
# act_dim = env.action_space.shape[0]

# if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
#     os.makedirs(video_save_path)

# demo_dataset = []
# recon_dataset = {
#     'observations': np.zeros((args.num_trajectories, args.num_timesteps, imlength), dtype=np.uint8),
#     'object': [],
#     'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
# }

# for i in tqdm(range(args.num_trajectories)):
#     env.reset()
#     trajectory = {
#         'observations': [],
#         'next_observations': [],
#         'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float),
#         'rewards': np.zeros((args.num_timesteps), dtype=np.float),
#         'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
#         'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
#         'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
#         'object_name': env.curr_object,
#     }
#     images = []
#     #recon_dataset['env'][i, :] = np.uint8(env.render_obs().transpose()).flatten()
#     #recon_dataset['object'].append(env.curr_object)
#     for j in range(args.num_timesteps):
#         img = np.uint8(env.render_obs())
#         #recon_dataset['observations'][i, j, :] = img.transpose().flatten()
#         images.append(Image.fromarray(img))

#         ee_pos = env.get_end_effector_pos()
#         target_pos = env.get_object_midpoint('obj')
#         aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.04
#         enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
#         above = ee_pos[2] > -0.3

#         if not aligned and not above:
#             #print('Stage 1')
#             action = (target_pos - ee_pos) * 3.0
#             action[2] = 1
#             grip = -1.
#         elif not aligned:
#             #print('Stage 2')
#             action = (target_pos - ee_pos) * 3.0
#             action[2] = 0.
#             action *= 3.0
#             grip = -1.
#         elif aligned and not enclosed:
#             #print('Stage 3')
#             action = target_pos - ee_pos
#             action[2] -= 0.03
#             action *= 3.0
#             action[2] *= 2.0
#             grip = -1.
#         elif enclosed and grip < 1:
#             #print('Stage 4')
#             action = target_pos - ee_pos
#             action[2] -= 0.03
#             action *= 3.0
#             action[2] *= 2.0
#             grip += 0.5
#         else:
#             #print('Stage 5')
#             action = env.goal_pos - ee_pos
#             action *= 3.0
#             grip = 1.

#         action = np.append(action, [grip])
#         action = np.random.normal(action, 0.1)
#         action = np.clip(action, a_min=-1, a_max=1)

#         observation = env.get_observation()
#         next_observation, reward, done, info = env.step(action)

#         trajectory['observations'].append(observation)
#         trajectory['actions'][j, :] = action
#         trajectory['next_observations'].append(next_observation)
#         trajectory['rewards'][j] = reward

#         returns += reward
    
#     success += info['picked_up']

#     if args.video_save_frequency > 0 and i % args.video_save_frequency == 0:
#         images[0].save('{}/{}.gif'.format(video_save_path, i),
#                        format='GIF', append_images=images[1:],
#                        save_all=True, duration=100, loop=0)

#     demo_dataset.append(trajectory)

# print('Success Rate: {}'.format(success / args.num_trajectories))
# print('Returns: {}'.format(returns / args.num_trajectories))



# #np.save(recon_data_save_path, recon_dataset)
# step_size = 1000
# for i in range(args.num_trajectories // step_size):
#     curr_name = demo_data_save_path + '_{0}.pkl'.format(i)
#     start_ind, end_ind = i*step_size, (i+1)*step_size
#     curr_data = demo_dataset[start_ind:end_ind]
#     file = open(curr_name, 'wb')
#     pkl.dump(curr_data, file)
#     file.close()
