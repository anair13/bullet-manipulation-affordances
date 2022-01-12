import pybullet as p
import numpy as np


def get_joint_states(body_id, joint_indices, physicsClientId=0):
    all_joint_states = p.getJointStates(body_id, joint_indices, physicsClientId=physicsClientId)
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities)


def get_movable_joints(body_id, physicsClientId=0):
    num_joints = p.getNumJoints(body_id, physicsClientId=physicsClientId)
    movable_joints = []
    for i in range(num_joints):
        if p.getJointInfo(body_id, i, physicsClientId=physicsClientId)[2] != p.JOINT_FIXED:
            movable_joints.append(i)
    return movable_joints


def get_link_state(body_id, link_index, physicsClientId=0):
    position, orientation, _, _, _, _ = p.getLinkState(body_id, link_index, physicsClientId=physicsClientId)
    return np.asarray(position), np.asarray(orientation)


def get_joint_info(body_id, joint_id, key, physicsClientId=0):
    keys = ["jointIndex", "jointName", "jointType", "qIndex", "uIndex",
            "flags", "jointDamping", "jointFriction", "jointLowerLimit",
            "jointUpperLimit", "jointMaxForce", "jointMaxVelocity", "linkName",
            "jointAxis", "parentFramePos", "parentFrameOrn", "parentIndex"]
    value = p.getJointInfo(body_id, joint_id, physicsClientId=physicsClientId)[keys.index(key)]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return value


def apply_action_ik(target_ee_pos, target_ee_quat, target_gripper_state,
                    robot_id, end_effector_index, movable_joints,
                    lower_limit, upper_limit, rest_pose, joint_range,
                    num_sim_steps=5, physicsClientId=0):
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               lowerLimits=lower_limit,
                                               upperLimits=upper_limit,
                                               jointRanges=joint_range,
                                               restPoses=rest_pose,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.01, 
                                               physicsClientId=physicsClientId)

    p.setJointMotorControlArray(robot_id,
                                movable_joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                # targetVelocity=0,
                                forces=[500] * len(movable_joints),
                                positionGains=[0.03] * len(movable_joints),
                                # velocityGain=1, 
                                physicsClientId=physicsClientId
                                )
    # set gripper action
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.03, 
                            physicsClientId=physicsClientId)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.03, 
                            physicsClientId=physicsClientId)

    for _ in range(num_sim_steps):
        p.stepSimulation(physicsClientId=physicsClientId)


def reset_robot(robot_id, reset_joint_indices, reset_joint_values, physicsClientId=0):
    assert len(reset_joint_indices) == len(reset_joint_values)
    for i, value in zip(reset_joint_indices, reset_joint_values):
        p.resetJointState(robot_id, i, value, physicsClientId=physicsClientId)


def reset_object(body_id, position, orientation, physicsClientId=0):
    p.resetBasePositionAndOrientation(body_id,
                                      position,
                                      orientation, 
                                      physicsClientId=physicsClientId)


def get_object_position(body_id, physicsClientId=0):
    object_position, object_orientation = \
        p.getBasePositionAndOrientation(body_id, physicsClientId=physicsClientId)
    return np.asarray(object_position), np.asarray(object_orientation)


def step_simulation(num_sim_steps, physicsClientId=0):
    for _ in range(num_sim_steps):
        p.stepSimulation(physicsClientId=physicsClientId)


def quat_to_deg(quat, physicsClientId=0):
    euler_rad = p.getEulerFromQuaternion(quat, physicsClientId=physicsClientId)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg


def deg_to_quat(deg, physicsClientId=0):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad, physicsClientId=physicsClientId)
    return quat


def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])
