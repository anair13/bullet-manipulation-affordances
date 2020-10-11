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

args = parser.parse_args()
data_save_path = "/home/ashvin/data/sasha/demos/" + args.name + ".pkl"
env = roboverse.make('SawyerRigMultiobj-v0', DoF=3, object_subset=['camera'], random_color_p=0)

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pkl.load(open(local_path, "rb"))
    print("loaded", local_path)
    vae.to("cpu")
    return vae

vae_path = "/home/ashvin/data/rail-khazatsky/sasha/complex_obj/pixelcnn_vqvae.pkl"
model = load_vae(vae_path)

dataset = []

for i in tqdm(range(args.num_trajectories)):
    env.reset()

    init_obs = ptu.from_numpy(np.uint8(env.render_obs()).transpose() / 255.0)
    init_z = ptu.get_numpy(model.encode(init_obs))

    sampled_z = model.sample_prior(args.num_timesteps, cond=init_z)
    dataset.append(sampled_z)

dataset = np.concatenate(dataset, axis=0)
print(dataset.shape)
    
file = open(data_save_path, 'wb')
pkl.dump(dataset, file)
file.close()