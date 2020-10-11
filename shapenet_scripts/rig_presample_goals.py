import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rlkit.envs.images import EnvRenderer, InsertImageEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.encoder_wrappers import VQVAEWrappedEnv
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--num_timesteps", type=int, default=50)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()
data_save_path = "/home/ashvin/data/sasha/demos/" + args.name + ".pkl"
video_save_path = "/home/ashvin/data/sasha/demos/videos"

env = roboverse.make('SawyerRigMultiobj-v0', DoF=3, object_subset=['grill_trash_can'], random_color_p=0)

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pkl.load(open(local_path, "rb"))
    print("loaded", local_path)
    vae.to("cpu")
    return vae

vae_path = "/home/ashvin/data/rail-khazatsky/sasha/complex_obj/best_vae.pkl"
model = load_vae(vae_path)

object_name = 'obj'
num_grasps = 0
success = 0
returns = 0
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.spaces['state_achieved_goal'].low.size

if not os.path.exists(video_save_path) and args.video_save_frequency > 0:
    os.makedirs(video_save_path)


imlength = env.obs_img_dim * env.obs_img_dim * 3

dataset = {
        'initial_latent_state': np.zeros((args.num_trajectories, model.representation_size), dtype=np.float),
        'latent_desired_goal': np.zeros((args.num_trajectories * args.num_timesteps,
            model.representation_size), dtype=np.float),
        'state_desired_goal': np.zeros((args.num_trajectories * args.num_timesteps,
            obs_dim), dtype=np.float),
        'image_desired_goal': np.zeros((args.num_trajectories * args.num_timesteps, imlength), dtype=np.float),
        }

for i in tqdm(range(args.num_trajectories)):
    env.reset()
    target_pos = env.get_object_midpoint(object_name)
    images = []

    init_img = np.uint8(env.render_obs()).transpose() / 255.0
    dataset['initial_latent_state'][i] = ptu.get_numpy(model.encode(ptu.from_numpy(init_img))).flatten()
    
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
            action = env.goal_pos - ee_pos
            action *= 3.0
            grip = 1.

        action = np.append(action, [grip])
        action = np.random.normal(action, 0.1)
        action = np.clip(action, a_min=-1, a_max=1)

        obs, reward, done, info = env.step(action)

        img = np.uint8(env.render_obs()).transpose() / 255.0
        img = ptu.from_numpy(img)

        # # temp
        # img = np.zeros_like(img)
        # img = ptu.from_numpy(img)
        # # temp\

        latent_obs = ptu.get_numpy(model.encode(img)).flatten()
        img = ptu.get_numpy(img)

        dataset['latent_desired_goal'][i * args.num_timesteps + j] = latent_obs
        dataset['state_desired_goal'][i * args.num_timesteps + j] = obs['state_achieved_goal']
        dataset['image_desired_goal'][i * args.num_timesteps + j] = img.flatten()
        dataset['initial_image_observation'][i * args.num_timesteps + j] = init_img.flatten()

        returns += reward
    
    success += info['object_goal_success']
    num_grasps += info['picked_up']

    if args.video_save_frequency > 0 and i % args.video_save_frequency == 0:
        images[0].save('{}/{}.gif'.format(video_save_path, i),
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)

print('Success Rate: {}'.format(success / args.num_trajectories))
print('Picked Up: {}'.format(num_grasps / args.num_trajectories))
print('Returns: {}'.format(returns / args.num_trajectories))
file = open(data_save_path, 'wb')
pkl.dump(dataset, file)
file.close()

