import roboverse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rlkit.envs.images import EnvRenderer, InsertImageEnv
import rlkit.torch.pytorch_util as ptu
#from rlkit.envs.encoder_wrappers import VQVAEWrappedEnv
import os
from PIL import Image
import argparse

from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands
from experiments.kuanfang.iql.drawer_pnp_single_obj_commands import drawer_pnp_single_obj_commands
from experiments.kuanfang.iql.drawer_pnp_push_commands import drawer_pnp_push_commands

########################################
# Args.
########################################
parser = argparse.ArgumentParser()
# parser.add_argument("--name", type=str)
parser.add_argument("--num_trajectories", type=int, default=25)
parser.add_argument("--num_timesteps", type=int, default=150)
parser.add_argument("--save_last_k_steps", type=int, default=65)
parser.add_argument("--no_save_last_w_steps", type=int, default=35)
parser.add_argument("--downsample", action='store_true')
parser.add_argument("--drawer_sliding", action='store_true')
parser.add_argument("--test_env_seeds", nargs='+', type=int)
parser.add_argument("--video_save_frequency", type=int,
                    default=0, help="Set to zero for no video saving")
parser.add_argument("--gui", dest="gui", action="store_true", default=False)

args = parser.parse_args()

num_timesteps = args.num_timesteps
num_trajectories = args.num_trajectories
save_last_k_steps = args.save_last_k_steps
no_save_last_w_steps = args.no_save_last_w_steps
ROOT_PATH = "/2tb/home/patrickhaoy/data/affordances/data/td_pnp_push_v4/"

for test_env_seed in args.test_env_seeds:
    data_save_path = ROOT_PATH + "td_pnp_push_goals_seed{}.pkl".format(str(test_env_seed))
    command = drawer_pnp_push_commands[test_env_seed]

    ########################################
    # Environment.
    ########################################
    kwargs = {
        'drawer_sliding': True if args.drawer_sliding else False,
        'test_env_command': command,
        'version': 4,
        # 'gui': True
    }
    if args.downsample:
        kwargs['downsample'] = True
        kwargs['env_obs_img_dim'] = 196
    env = roboverse.make('SawyerRigAffordances-v3', test_env=True, expl=True, use_single_obj_idx=1, **kwargs)

    ########################################
    # Rollout in Environment and Collect Data.
    ########################################
    obs_dim = env.observation_space.spaces['state_achieved_goal'].low.size
    imlength = env.obs_img_dim * env.obs_img_dim * 3

    num_transitions = num_trajectories * (save_last_k_steps - no_save_last_w_steps) 
    dataset = {
        'initial_latent_state': np.zeros((num_transitions, 720), dtype=np.float),
        'latent_desired_goal': np.zeros((num_transitions, 720), dtype=np.float),
        'state_desired_goal': np.zeros((num_transitions, obs_dim), dtype=np.float),
        'image_desired_goal': np.zeros((num_transitions, imlength), dtype=np.float),
        'initial_image_observation': np.zeros((num_transitions, imlength), dtype=np.float),
    }

    for i in tqdm(range(num_trajectories)):
        is_done = False
        ## If task isn't done, scripted policy failed, so recollect samples.
        while not is_done:
            is_done = True

            env.demo_reset()
            init_img = np.uint8(env.render_obs()).transpose() / 255.0

            ## All but final skill
            for j0 in range(len(command['command_sequence']) - 1):
                skill_is_done = False
                for j1 in range(num_timesteps):
                    action, done = env.get_demo_action(first_timestep=(j1 == 0), final_timestep=(j1 == num_timesteps - 1), return_done=True)
                    obs, reward, _, info = env.step(action)
                    skill_is_done = skill_is_done or done
                is_done = is_done and skill_is_done
                env.demo_reset()
            
            ## Final skill
            skill_is_done = False
            for j in range(num_timesteps):
                action, done = env.get_demo_action(first_timestep=(j == 0), final_timestep=(j == num_timesteps - 1), return_done=True)
                obs, reward, _, info = env.step(action)
                skill_is_done = skill_is_done or done

                if j + save_last_k_steps >= num_timesteps and j < num_timesteps - no_save_last_w_steps:
                    img = np.uint8(env.render_obs()).transpose() / 255.0

                    j_o = j - (num_timesteps - save_last_k_steps)
                    dataset['state_desired_goal'][i * (save_last_k_steps - no_save_last_w_steps) + j_o] = obs['state_achieved_goal']
                    dataset['image_desired_goal'][i * (save_last_k_steps - no_save_last_w_steps) + j_o] = img.flatten()
                    dataset['initial_image_observation'][i * (save_last_k_steps - no_save_last_w_steps) + j_o] = init_img.flatten()
            is_done = is_done and skill_is_done

    file = open(data_save_path, 'wb')
    pkl.dump(dataset, file)
    file.close()