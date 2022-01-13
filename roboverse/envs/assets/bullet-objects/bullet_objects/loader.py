import numpy as np
from roboverse.bullet.misc import load_obj, deg_to_quat
import os.path as osp
from pathlib import Path

path = Path(__file__).parent.absolute()

SHAPENET_ASSET_PATH = str(path.parent.joinpath("ShapeNetCore"))
print("shapenet asset path:", SHAPENET_ASSET_PATH)

def load_shapenet_object(object_path, scaling, object_position, scale_local=0.5, rgba=None, quat=None, physicsClientId=0):
    path = object_path.split('/')
    dir_name, object_name, = path[-2], path[-1]

    # Randomize initial theta
    if quat is None:
        quat = deg_to_quat(np.random.randint(0, 360, size=3))

    obj = load_obj(
        SHAPENET_ASSET_PATH + '/ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name, object_name),
        SHAPENET_ASSET_PATH + '/ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(dir_name, object_name),
        object_position, quat=quat, rgba=rgba, scale=scale_local*scaling['{0}/{1}'.format(dir_name, object_name)], 
        physicsClientId=physicsClientId)

    return obj
