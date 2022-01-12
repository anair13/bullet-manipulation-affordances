import pybullet as p

def set_body_state(body, pos, quat, physicsClientId=0):
	p.resetBasePositionAndOrientation(body, pos, quat, physicsClientId=physicsClientId)