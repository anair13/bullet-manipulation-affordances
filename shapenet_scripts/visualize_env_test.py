import roboverse as rv
import numpy as np
import skvideo.io

from experiments.kuanfang.iql.drawer_pnp_commands import drawer_pnp_commands

#obs_img_dim=196, 
env = rv.make(
    "SawyerRigAffordances-v2", 
    gui=True, 
    expl=True, 
    reset_interval=10, 
    drawer_sliding=False, 
    env_obs_img_dim=196, 
    random_color_p=0.0, 
    test_env=True, 
    test_env_command=drawer_pnp_commands[1],
    # expert_policy_std=0.05,
)#, downsample=True)  
ts = 100
num_traj = 100

save_video = False

if save_video:
    video_save_path = '/2tb/home/patrickhaoy/data/test/'
    num_traj = 3
    observations = np.zeros((num_traj*ts, 196, 196, 3))

tasks_success = dict()
tasks_count = dict()
for i in range(num_traj):
    env.demo_reset()
    curr_task = env.curr_task
    is_done = False
    for t in range(ts):
        if save_video:
            img = np.uint8(env.render_obs())
            observations[i*ts + t, :] = img
        action, done = env.get_demo_action(first_timestep=(t == 0), final_timestep=(t == ts - 1), return_done=True)
        next_observation, reward, _, info = env.step(action)
        if done and not is_done:
            is_done = True 
            
            if curr_task not in tasks_success.keys():
                tasks_success[curr_task] = 1
            else:
                tasks_success[curr_task] += 1
    
    if curr_task not in tasks_count.keys():
        tasks_count[curr_task] = 1
    else:
        tasks_count[curr_task] += 1

print()
total_successes = 0

for task in tasks_count.keys():
    num_successes = tasks_success.get(task, 0)
    num_tries = tasks_count.get(task, 0)
    print(f"{task} | success rate: {num_successes/num_tries}, count: {num_tries} \n")
    total_successes += num_successes

print(f"Overall success rate: {total_successes/num_traj}, count: {num_traj} \n")


if save_video:
    writer = skvideo.io.FFmpegWriter(video_save_path + "debug.mp4")
    for i in range(num_traj*ts):
            writer.writeFrame(observations[i, :, :, :])
    writer.close()
