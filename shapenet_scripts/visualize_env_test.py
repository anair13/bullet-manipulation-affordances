import roboverse as rv
import numpy as np
import skvideo.io

#obs_img_dim=196, 
env = rv.make("SawyerRigAffordances-v1", new_view=True, close_view=True, gui=False, expl=False, reset_interval=10, drawer_sliding=True, fix_drawer_orientation_semicircle=True, test_env=True, test_env_seed=1, env_obs_img_dim=196, red_drawer_base=True) #down_sample=True
ts = 75
num_traj = 100

save_video = True

if save_video:
    video_save_path = '/2tb/home/patrickhaoy/data/test/'
    num_traj = 1
    observations = np.zeros((ts, 196, 196, 3))

for i in range(num_traj):
    env.demo_reset()
    for t in range(ts):
        if save_video:
            img = np.uint8(env.render_obs())
            observations[t, :] = img
        action = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1))
        next_observation, reward, done, info = env.step(action)

if save_video:
    writer = skvideo.io.FFmpegWriter(video_save_path + "debug.mp4")
    for i in range(ts):
            writer.writeFrame(observations[i, :, :, :])
    writer.close()
