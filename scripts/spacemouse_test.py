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
env = rv.make('SawyerRigMultiobjDrawer-v0', gui=True)
env.reset()

# while True:
# 	action = spacemouse.get_action()
# 	next_obs, rew, term, info = env.step(action)
# 	print(rew)
# 	if term: break

def move_hand(grip):
    ee_pos = env.get_end_effector_pos()
    done = np.linalg.norm(ee_pos - env.hand_goal) < 0.04
    above = ee_pos[2] >= -0.11
    grip = -1.

    if not above:
        #print('Stage 1')
        action = np.array([0,0,1])
    else:
        #print('Stage 2')
        action = (env.hand_goal - ee_pos) * 3.0

    return action, grip, done

def move_drawer(grip):
    ee_pos = env.get_end_effector_pos()
    target_pos = env.get_object_pos('drawer_handle') + np.array([0,0.025,0])
    aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.075
    enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.01
    reached = np.linalg.norm(env.td_goal - target_pos) < 0.04
    above = ee_pos[2] >= -0.105
    done = reached and above
    grip = -1.

    if not aligned and not above and not reached:
        #print('Stage 1')
        action = np.array([0,0,1])
    elif not aligned and not reached:
        #print('Stage 2')
        action = (target_pos - ee_pos) * 3.0
        action[2] = 0.
    elif aligned and not enclosed and not reached:
        #print('Stage 3')
        action = target_pos - ee_pos
        action[2] -= 0.03
        action *= 3.0
        action[2] *= 2.0
    elif not reached:
        #print('Stage 4')
        action = np.sign(np.array([0, env.td_goal[1] - ee_pos[1], 0]))
    else:
    	#print('Stage 5')
    	action = np.array([0, 0, 1])
 	
    return action, grip, reached

def press_button(grip):
    pressed = env.drawer_opened
    ee_pos = env.get_end_effector_pos()
    target_pos = env.get_object_pos('button')
    aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.05
    enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
    above = ee_pos[2] >= -0.105
    done = pressed and above
    grip = -1.

    if (not aligned) and (not above) and (not pressed):
        #print('Stage 1')
        action = np.array([0,0,1])
    elif not aligned and not pressed:
        #print('Stage 2')
        action = (target_pos - ee_pos) * 3.0
        action[2] = 0.
    elif not pressed:
        #print('Stage 3')
        action = np.array([0, 0, -1.])
    else:
        #print('Stage 4')
        action = np.array([0, 0, 1.])

    return action, grip, pressed

def move_obj(grip, obj, goal):
    ee_pos = env.get_end_effector_pos()
    adjustment = 0 if obj == 'lego' else np.array([0.0, -0.018, 0])
    target_pos = env.get_object_pos(obj) + adjustment
    aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.05
    enclosed = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.025
    reached = np.linalg.norm(target_pos[:2] - goal[:2]) < 0.04
    done = reached and (grip == -1.)
    above = ee_pos[2] >= -0.125

    if not aligned and not above and not reached:
        #print('Stage 1')
        action = np.array([0,0,1])
        grip = -1.
    elif not aligned and not reached:
        #print('Stage 2')
        action = (target_pos - ee_pos) * 3.0
        action[2] = 0.
        grip = -1.
    elif aligned and not enclosed and not reached:
        #print('Stage 3')
        action = target_pos - ee_pos
        action[2] -= 0.03
        action *= 3.0
        action[2] *= 2.0
        grip = -1.
    elif enclosed and grip < 1 and not reached:
        #print('Stage 4')
        action = target_pos - ee_pos
        action[2] -= 0.03
        action *= 3.0
        action[2] *= 2.0
        grip += 0.5
    elif not above and not reached:
        #print('Stage 5')
        action = np.array([0, 0, 1])
        grip = 1.
    elif not reached:
        #print('Stage 6')
        action = goal - ee_pos
        action[2] = 0
        action *= 3.0
        grip = 1.
    elif reached and grip > -1.:
        print('Stage 7')
        action = np.array([0,0,0])
        grip -= 0.25
    else:
        print('Stage 8')
        action = np.array([0,0,0])
        grip = -1.

    return action, grip, done

grip = -1.
stages_left = [1,1,1,1,1]
images = []

for i in range(150):
    img = np.uint8(env.render_obs())
    images.append(Image.fromarray(img))

    if stages_left[0]:
        action, grip, done = move_drawer(grip)
        if i >= 40 or done:
            stages_left[0] = 0

    elif stages_left[1]:
        action, grip, done = press_button(grip)
        if done:
            stages_left[1] = 0
        elif i >= 60:
        	stages_left[1] = 0
        	stages_left[2] = 0
  
    elif stages_left[2]:
        action, grip, done = move_obj(grip, 'lego', env.lego_goal)
        if i >= 100 or done:
            stages_left[2] = 0

    elif stages_left[3]:
        action, grip, done = move_obj(grip, 'obj', env.obj_goal)
        if done:
            stages_left[3] = 0
    elif stages_left[4]:
        action, grip, done = move_hand(grip)
        if done:
            stages_left[4] = 0
    else:
        action = np.random.normal(size=(3,), scale=0.25)
        grip = np.random.normal()

    print('Stages Left: ', stages_left)
    action = np.append(action, [grip])
    action = np.random.normal(action, 0.1)
    action = np.clip(action, a_min=-1, a_max=1)

    observation = env.get_observation()
    next_observation, reward, done, info = env.step(action)



images[0].save('/Users/sasha/Desktop/scipted_policy.gif',
                       format='GIF', append_images=images[1:],
                       save_all=True, duration=100, loop=0)