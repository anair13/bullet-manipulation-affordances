import roboverse as rv
import numpy as np
import skvideo.io

#obs_img_dim=196, 
env = rv.make(
    "SawyerRigAffordances-v2", 
    gui=True, 
    expl=True, 
    reset_interval=5, 
    drawer_sliding=False, 
    env_obs_img_dim=196, 
    random_color_p=0.0, 
    # test_env=True, 
    # test_env_seed=6
)#, downsample=True)  
ts = 100
num_traj = 100

save_video = True

if save_video:
    video_save_path = '/2tb/home/patrickhaoy/data/test/'
    num_traj = 3
    observations = np.zeros((num_traj*ts, 196, 196, 3))

for i in range(num_traj):
    env.demo_reset()
    for t in range(ts):
        if save_video:
            img = np.uint8(env.render_obs())
            observations[i*ts + t, :] = img
        action = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1))
        next_observation, reward, done, info = env.step(action)

if save_video:
    writer = skvideo.io.FFmpegWriter(video_save_path + "debug.mp4")
    for i in range(num_traj*ts):
            writer.writeFrame(observations[i, :, :, :])
    writer.close()
