import roboverse as rv

env = rv.make("SawyerRigAffordances-v1", gui=True, expl=True, reset_interval=10, drawer_sliding=True)
ts = 100

for i in range(100):
    env.demo_reset()
    for t in range(ts):
        action = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1))
        next_observation, reward, done, info = env.step(action)