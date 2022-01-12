import pybullet as p
import roboverse.bullet as bullet
import roboverse.bullet.control as control
import numpy as np


def pop_up_button(button, physicsClientId=0):
    return slide_button(button, 1, physicsClientId=physicsClientId)


def push_down_button(button, physicsClientId=0):
    return slide_button(button, -1, physicsClientId=physicsClientId)


def get_button_cylinder_pos(button, physicsClientId=0):
    button_cylinder_pos = bullet.get_link_state(
        button, get_button_cylinder_joint(button, physicsClientId=physicsClientId), physicsClientId=physicsClientId)
    return button_cylinder_pos['pos']


def get_button_cylinder_joint(button, physicsClientId=0):
    joint_names = [control.get_joint_info(button, j, 'jointName', physicsClientId=physicsClientId)
                   for j in range(p.getNumJoints(button, physicsClientId=physicsClientId))]
    button_cylinder_joint_idx = joint_names.index('base_button_joint')
    return button_cylinder_joint_idx


def slide_button(button, direction, physicsClientId=0):
    assert direction in [-1, 1]
    # -1 = push down; 1 = pop up
    button_cylinder_joint_idx = get_button_cylinder_joint(button, physicsClientId=physicsClientId)
    num_ts = 20
    command = 10 * direction

    p.setJointMotorControl2(
        button,
        button_cylinder_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=command,
        force=10, 
        physicsClientId=physicsClientId
    )

    control.step_simulation(num_ts, physicsClientId=physicsClientId)

    button_pos = get_button_cylinder_pos(button, physicsClientId=physicsClientId)

    p.setJointMotorControl2(
        button,
        button_cylinder_joint_idx,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=10, 
        physicsClientId=physicsClientId
    )

    return button_pos
