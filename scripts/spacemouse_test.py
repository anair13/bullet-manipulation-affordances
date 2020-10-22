import roboverse as rv
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg 
import os
from PIL import Image
import argparse


#spacemouse = rv.devices.SpaceMouse(DoF=3)
env = rv.make('SawyerRigAffordances-v0')
#env.reset()
images = []
for j in range(10):
    env.demo_reset()
    for i in range(50):
        img = np.uint8(env.render_obs())
        images.append(Image.fromarray(img))

        action = env.get_demo_action()
        next_observation, reward, done, info = env.step(action)

# env.demo_reset()
# i = 0
# while True:
#     i += 1
#     if i % 50 == 0:
#         env.reset()
#     img = np.uint8(env.render_obs())
#     action = spacemouse.get_action()
#     next_obs, rew, term, info = env.step(action)
#     #print(rew)
#     if term: break

# images[0].save('/home/ashvin/data/sasha/drawer_test/rollout.gif',
#                        format='GIF', append_images=images[1:],
#                        save_all=True, duration=100, loop=0)
images[0].save('/Users/sasha/Desktop/rollout.gif',
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)