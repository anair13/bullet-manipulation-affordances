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

#env = rv.make('SawyerRigMultiobjTray-v0', gui=True)
#env = rv.make('SawyerRigAffordances-v0', gui=True)
env = rv.make('SawyerRigMultiobj-v0', gui=True, test_env=True)

for j in range(5):
	env.demo_reset()
	#env.reset()
	for i in range(75):
		img = Image.fromarray(np.uint8(env.render_obs()))
		#action = spacemouse.get_action()
		action = env.get_demo_action()
		next_observation, reward, done, info = env.step(action)
	#img.save('/Users/sasha/Desktop/ENV2/{}.jpeg'.format(j))

# env = rv.make('SawyerRigAffordances-v0', gui=True)
#env = rv.make('SawyerRigMultiobjTray-v0', gui=True)
# for j in range(5):
#     print('Iteration: ', j)
#     env.reset()
#     #env.demo_reset()

#     for i in range(75):
#         img = Image.fromarray(np.uint8(env.render_obs()))
#         images.append(img)

#         action = env.get_demo_action()
#         next_observation, reward, done, info = env.step(action)