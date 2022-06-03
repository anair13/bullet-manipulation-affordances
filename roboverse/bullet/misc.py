from roboverse.utils.serialization import make_dir
import pybullet_data as pdata
import pybullet as p
import pdb
import cv2
import os
import random
import numpy as np
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in os.sys.path:
    os.sys.path.remove(ros_path)
os.sys.path.append(ros_path)


#########################
#### setup functions ####
#########################

def connect():
    clid = p.connect(p.SHARED_MEMORY)
    if (clid < 0):
        clid = p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=clid)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=clid)

    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-40,
                                 cameraTargetPosition=[.7, 0, -0.3], physicsClientId=clid)
    #p.resetDebugVisualizerCamera(0.8, 90, -20, [0.75, -.2, 0])
    p.setAdditionalSearchPath(pdata.getDataPath(), physicsClientId=clid)

    return clid


def connect_headless(gui=False):
    if gui:
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            cid = p.connect(p.GUI)
    else:
        cid = p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=90, cameraPitch=-40,
                                 cameraTargetPosition=[.7, 0, -0.3], physicsClientId=cid)
    #p.resetDebugVisualizerCamera(0.8, 90, -20, [0.75, -.2, 0])
    p.setAdditionalSearchPath(pdata.getDataPath(), physicsClientId=cid)

    return cid


def setup(real_time=True, gravity=-10, physicsClientId=0):
    '''
        sets parameters for running pybullet
        interactively
    '''
    p.setRealTimeSimulation(real_time, physicsClientId=physicsClientId)
    p.setGravity(0, 0, gravity, physicsClientId=physicsClientId)
    p.stepSimulation(physicsClientId=physicsClientId)


def setup_headless(timestep=1./240, solver_iterations=150, gravity=-10, physicsClientId=0):
    '''
        sets parameters for running pybullet
        in a headless environment
    '''
    p.setPhysicsEngineParameter(
        numSolverIterations=solver_iterations, physicsClientId=physicsClientId)
    p.setTimeStep(timestep, physicsClientId=physicsClientId)
    p.setGravity(0, 0, gravity, physicsClientId=physicsClientId)
    p.stepSimulation(physicsClientId=physicsClientId)


def reset(physicsClientId=0):
    p.resetSimulation(physicsClientId=physicsClientId)


def replace_line(filename, line_num, text):
    lines = open(filename, 'r').readlines()
    lines[line_num] = text
    out = open(filename, 'w')
    out.writelines(lines)
    out.close()


def load_urdf_randomize_color(filepath, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None, physicsClientId=0):
    from shutil import copyfile

    if rgba is not None:
        rand_filepath = filepath[:-5] + \
            '_rand_color_{0}.urdf'.format(np.random.uniform())
        copyfile(filepath, rand_filepath)
        color_line = '    <color rgba="{0} {1} {2} {3}"/>'.format(
            rgba[0], rgba[1], rgba[2], rgba[3])
        replace_line(rand_filepath, 3, color_line)
        try:
            body = p.loadURDF(rand_filepath, globalScaling=scale)
            p.changeVisualShape(body, -1, rgbaColor=rgba,
                                physicsClientId=physicsClientId)
        finally:
            pass
            # os.remove(rand_filepath)
    else:
        body = p.loadURDF(filepath, globalScaling=scale,
                          physicsClientId=physicsClientId)

    p.resetBasePositionAndOrientation(
        body, pos, quat, physicsClientId=physicsClientId)
    return body


def load_urdf(filepath, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None, physicsClientId=0):
    body = p.loadURDF(filepath, globalScaling=scale,
                      physicsClientId=physicsClientId)
    p.resetBasePositionAndOrientation(
        body, pos, quat, physicsClientId=physicsClientId)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba,
                            physicsClientId=physicsClientId)

    return body


def load_obj(filepathcollision, filepathvisual, pos=[0, 0, 0], quat=[0, 0, 0, 1], scale=1, rgba=None, physicsClientId=0):
    collisionid = p.createCollisionShape(p.GEOM_MESH, fileName=filepathcollision,
                                         meshScale=scale * np.array([1, 1, 1]), physicsClientId=physicsClientId)
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepathvisual,
                                   meshScale=scale * np.array([1, 1, 1]), physicsClientId=physicsClientId)
    body = p.createMultiBody(0.05, collisionid, visualid,
                             physicsClientId=physicsClientId)
    if rgba is not None:
        p.changeVisualShape(body, -1, rgbaColor=rgba,
                            physicsClientId=physicsClientId)
    p.resetBasePositionAndOrientation(
        body, pos, quat, physicsClientId=physicsClientId)
    return body


def save_state(*savepath, physicsClientId=0):
    if len(savepath) > 0:
        savepath = os.path.join(*savepath)
        make_dir(os.path.dirname(savepath))
        p.saveBullet(savepath, physicsClientId=physicsClientId)
        state_id = None
    else:
        state_id = p.saveState(physicsClientId=physicsClientId)
    return state_id


def load_state(*loadpath, physicsClientId=0):
    loadpath = os.path.join(*loadpath)
    p.restoreState(fileName=loadpath, physicsClientId=physicsClientId)

#############################
#### rendering functions ####
#############################


def get_view_matrix(target_pos=[.75, -.2, 0], distance=0.9,
                    yaw=90, pitch=-20, roll=0, up_axis_index=2, physicsClientId=0):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        target_pos, distance, yaw, pitch, roll, up_axis_index, physicsClientId=physicsClientId)
    return view_matrix


def get_projection_matrix(height, width, fov=60, near_plane=0.1, far_plane=2, physicsClientId=0):
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(
        fov, aspect, near_plane, far_plane, physicsClientId=physicsClientId)
    return projection_matrix


def render(height, width, view_matrix, projection_matrix,
           shadow=1, light_direction=[1, 1, 1], renderer=p.ER_BULLET_HARDWARE_OPENGL, gaussian_width=5, physicsClientId=0):
    ## ER_BULLET_HARDWARE_OPENGL
    img_tuple = p.getCameraImage(width,
                                 height,
                                 view_matrix,
                                 projection_matrix,
                                 shadow=shadow,
                                 lightDirection=light_direction,
                                 renderer=renderer,
                                 physicsClientId=physicsClientId)
    _, _, img, depth, segmentation = img_tuple
    # import ipdb; ipdb.set_trace()
    # Here, if I do len(img), I get 9216.
    img = np.reshape(np.array(img), (width, height, 4))
    img = img[:, :, :-1]
    if gaussian_width > 0:
        img = cv2.GaussianBlur(img, (gaussian_width, gaussian_width), 0)
    return img, depth, segmentation

############################
#### rotation functions ####
############################


def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])


def quat_to_deg(quat, physicsClientId=0):
    euler_rad = p.getEulerFromQuaternion(quat, physicsClientId=physicsClientId)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg


def quat_to_deg_batch(quat_batch, physicsClientId=0):
    euler_deg_batch = np.zeros((quat_batch.shape[0], 3))
    for i in range(quat_batch.shape[0]):
        euler_rad = p.getEulerFromQuaternion(
            quat_batch[i], physicsClientId=physicsClientId)
        euler_deg = rad_to_deg(euler_rad)
        euler_deg_batch[i] = euler_deg
    return euler_deg_batch


def deg_to_quat(deg, physicsClientId=0):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad, physicsClientId=physicsClientId)
    return quat


#######################
#### miscellaneous ####
#######################

def step(physicsClientId=0):
    p.stepSimulation(physicsClientId=physicsClientId)


def l2_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b, 2)


def rot_diff_deg(a, b):
    '''
        a, b : orientations in degrees
        returns ||a-b||_1, taking into account that multiple
        [r_x, r_y, r_z] vectors correspond to the same orientation
    '''
    diff = (a - b) % 360
    diff = np.minimum(diff, 360-diff)
    return np.linalg.norm(diff, 1)


def add_debug_line(x, y, rgb=[1, 0, 0], duration=5, physicsClientId=0):
    p.addUserDebugLine(x, y, rgb, duration, physicsClientId=physicsClientId)

# def is_contacting(body_1, body_2, link_1=-1, link_2=-1, physicsClientId=0):
#     points = p.getContactPoints(body_1, body_2, link_1, link_2, physicsClientId=physicsClientId)
#     return len(points) > 0


def is_contacting(body_1, body_2, link_1=-1, link_2=-1, threshold=.005):
    dist = get_link_dist(body_1, body_2, link_1=link_1, link_2=link_2)
    return dist < threshold


def get_link_dist(body_1, body_2, link_1=-1, link_2=-1, threshold=1, physicsClientId=0):
    points = p.getClosestPoints(
        body_1, body_2, threshold, link_1, link_2, physicsClientId=physicsClientId)
    distances = [point[8] for point in points] + [np.float('inf')]
    return min(distances)


def get_bbox(body, draw=False, physicsClientId=0):
    xyz_min, xyz_max = p.getAABB(body, physicsClientId=physicsClientId)
    if draw:
        draw_bbox(xyz_min, xyz_max)
    return np.array(xyz_min), np.array(xyz_max)


def bbox_intersecting(bbox_1, bbox_2):
    min_1, max_1 = bbox_1
    min_2, max_2 = bbox_2
    # print(min_1, max_1, min_2, max_2)
    intersecting = (min_1 <= max_2).all() and (min_2 <= max_1).all()
    return intersecting


def get_midpoint(body, weights=[.5, .5, .5], physicsClientId=0):
    weights = np.array(weights)
    xyz_min, xyz_max = get_bbox(body, physicsClientId=physicsClientId)
    midpoint = xyz_max * weights + xyz_min * (1 - weights)
    return midpoint


def draw_bbox(aabbMin, aabbMax, physicsClientId=0):
    '''
        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getAABB.py
    '''
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 0, 0], physicsClientId=physicsClientId)
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [0, 1, 0], physicsClientId=physicsClientId)
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [0, 0, 1], physicsClientId=physicsClientId)

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)

    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)

    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)

    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1.0, 0.5, 0.5], physicsClientId=physicsClientId)
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1], physicsClientId=physicsClientId)
