import os
import pdb
import numpy as np
import pybullet as p
import pybullet_data as pdata
import math

from roboverse.bullet.misc import (
  load_urdf,
  load_urdf_randomize_color,
  deg_to_quat,
)

def loader_randomize_color(*filepath, **defaults):
    filepath = os.path.join(*filepath)
    def fn(*args, **kwargs):
        defaults.update(kwargs)

        if 'deg' in defaults:
          assert 'quat' not in defaults
          defaults['quat'] = deg_to_quat(defaults['deg'])
          del defaults['deg']
        return load_urdf_randomize_color(filepath, **defaults)
    return fn

def loader(*filepath, **defaults):
    filepath = os.path.join(*filepath)
    def fn(*args, **kwargs):
        defaults.update(kwargs)

        if 'deg' in defaults:
          assert 'quat' not in defaults
          defaults['quat'] = deg_to_quat(defaults['deg'])
          del defaults['deg']
        return load_urdf(filepath, **defaults)
    return fn

cur_path = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(cur_path, '../envs/assets')
PDATA_PATH = pdata.getDataPath()
obj_dir = "bullet-objects"

## robots

sawyer = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro.urdf')
sawyer_invisible = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_invisible.urdf')
sawyer_finger_visual_only = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/sawyer_xacro_finger_visual_only.urdf')
sawyer_hand_visual_only = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/simple_sawyer_xacro_finger_visual_only.urdf')
drawer_sawyer = loader(ASSET_PATH, 'sawyer_robot/sawyer_description/urdf/drawer_sawyer.urdf')

widowx_200 = loader(
  ASSET_PATH,
  'interbotix_descriptions/urdf/wx200.urdf',
  pos=[0.6, 0, -0.4],
  deg=[math.pi, math.pi, math.pi],
  scale=1
)


## pybullet_data objects
table = loader(PDATA_PATH, 'table/table.urdf',
               pos=[.75, -.2, -1],
               quat=[0, 0, 0.707107, 0.707107],
               scale=1.0)

duck = loader(PDATA_PATH, 'duck_vhacd.urdf',
              pos=[.75, -.4, -.3],
              quat=[0, 0, 1, 0],
              #deg=[0,0,0],
              scale=0.8)

lego = loader(PDATA_PATH, 'lego/lego.urdf',
              pos=[.75, .2, -.3],
              quat=[0, 0, 1, 0],
              rgba=[1, 0, 0, 1],
              scale=1.3)

traybox = loader(PDATA_PATH, 'tray/traybox.urdf',
              pos=[.65, 0, -.3],
              quat=[0, 0, 1, 0],
              rgba=[1, .898, .706, 1],
              scale=2.0)

## custom objects

bowl = loader(ASSET_PATH, os.path.join(obj_dir, "bowl", "bowl.urdf"),
              pos=[.75, 0, -.3],
              scale=0.1)

lid = loader(ASSET_PATH, os.path.join(obj_dir, "bowl", "lid.urdf"),
              pos=[.75, 0, -.3],
              scale=0.1) #0.25

cube = loader(ASSET_PATH, os.path.join(obj_dir, "cube", "cube.urdf"),
              pos=[.75, -.4, -.3],
              quat=[0, 0, 0, 1],
              scale=0.03) #0.05

spam = loader(ASSET_PATH, os.path.join(obj_dir, "spam", "spam.urdf"),
              pos=[.75, -.4, -.3],
              #deg=[90,0,0], #90,0,-90
              quat=[0, 0, 0, 1],
              scale=0.015) #0.0175 #0.25
## tray
tray = loader(ASSET_PATH, os.path.join(obj_dir, "box_open_top", "box_open_top.urdf"),
              pos=[.6, -0.2, -.35],
              rgba=[1, 1, 1, 1],
              deg=[0, 0, 0],
              scale=0.175)

wall = loader(ASSET_PATH, os.path.join(obj_dir, "wall", "wall.urdf"),
              pos=[.68, 0, -.3],
              rgba=[1, 1, 1, 1],
              deg=[0, 0, 0],
              scale=0.8)

box = loader(ASSET_PATH, os.path.join(obj_dir, "box", "box.urdf"),
                # pos=[0.85, 0, -.35],
                pos=[0.8, 0.075, -.35],
                scale=0.125)

widow200_tray = loader(ASSET_PATH, os.path.join(obj_dir, "tray", "tray.urdf"),
              pos=[0.8, -0.05, -0.36],
              deg=[0, 0, 0],
              scale=0.5)
bowl_sliding = loader(ASSET_PATH, 'objects/bowl_sliding/bowl.urdf',
              pos=[.75, 0, -.3],
              scale=0.25)


# Drawer (WARNING, DO NOT TOUCH THESE URDF FILES, IT WILL BREAK THINGS AND BE VERY HARD TO CATCH!!!!!)
drawer_pos = np.array([0.6, 0.125, -.34])
drawer = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.12]),
              scale=0.125)
drawer_sliding = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_sliding.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.12]),
              scale=0.125)
drawer_red_base = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_red_base.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.12]),
              scale=0.125)
drawer_sliding_red_base = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_sliding_red_base.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.12]),
              scale=0.125)
drawer_no_handle = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_no_handle.urdf"),
              pos=drawer_pos,
              deg=[0,0,90],
              scale=0.125)
button = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "button", "button.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.2]),
              scale=0.25)

drawer_no_randomize = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.12]),
              scale=0.125)
drawer_no_handle_no_randomize = loader(ASSET_PATH, os.path.join(obj_dir, "drawer", "drawer_no_handle.urdf"),
              pos=drawer_pos,
              deg=[0,0,90],
              scale=0.125)
button_no_randomize = loader(ASSET_PATH, os.path.join(obj_dir, "button", "button.urdf"),
              pos=drawer_pos + np.array([0, 0, 0.2]),
              scale=0.25)

drawer_lego = loader(PDATA_PATH, 'lego/lego.urdf',
              pos=drawer_pos + np.array([-0.01, 0, 0.03]),
              quat=[0, 0, 1, 0],
              rgba=[0, 0, 1, 1],
              scale=1.4)
drawer_tray = loader_randomize_color(ASSET_PATH, os.path.join(obj_dir, "box_open_top", "box_open_top.urdf"),
              pos=[0.6, -0.15, -.35],
              deg=[0, 0, 0],
              scale=0.175)
