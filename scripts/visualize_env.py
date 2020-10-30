import roboverse as rv
import numpy as np
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg 
import os
from PIL import Image
import argparse



images = []
env = rv.make('SawyerRigAffordances-v0', random_color_p=0.0, gui=True)
# env = rv.make('SawyerRigAffordances-v0', gui=True)
#env = rv.make('SawyerRigMultiobjTray-v0', gui=True)
for j in range(5):
    print('Iteration: ', j)
    env.demo_reset()

    for i in range(75):
        img = Image.fromarray(np.uint8(env.render_obs()))
        images.append(img)

        action = env.get_demo_action()
        next_observation, reward, done, info = env.step(action)