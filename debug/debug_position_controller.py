from VisFly.envs.HoverEnv import HoverEnv
import torch as th
import cv2
import numpy as np


env = HoverEnv(
    num_scene=1,
    num_agent_per_scene=4,
    visual=True,
    dynamics_kwargs={
        "action_type": "position",
    },
    random_kwargs={
        "state_generator": {
            "type": "uniform",
            "kwargs": [
                {
                    "position": {
                        "half": [1.0, 1.0, 0.1],
                        "mean": [2.0, 2.0, 1.2]
                    },
                }
            ]
        }
    },
    scene_kwargs={
        "path": "VisFly/datasets/visfly-beta/configs/scenes/box15_wall_empty",
        "render_settings":{
            "mode": "fix",
            "view": "custom",
            "resolution": [1920, 1080],
            "position":[[6., 5.5, 8.5], [6, 6, 1.5]],
            "trajectory": True,
            "axes": True
        }
    }

)
env.reset()

points = th.tensor((
    [[4,0,0],
     [4,4,0],
     [0,4,0],
     [0,0,0],]
)) + th.tensor([[2, 2, 1.2]])

num_agent = env.num_agent


total_steps = 1000
current_target_index = th.zeros((num_agent,), dtype=int)

current_target = points[current_target_index]

pos_action = current_target / 10
action = th.hstack([th.zeros((num_agent, 1)),pos_action])

pos = []
t = []
for i in range(total_steps):
    obs, reward, done, info = env.step(action)
    img = env.render()
    for j in range(num_agent):
        reach_target = th.norm(env.position[j] - current_target[j]) < 0.3
        if reach_target:
            current_target_index[j] = (current_target_index[j] + 1) % points.shape[0]
            current_target[j] = points[current_target_index[j]]
    pos_action = current_target / 10
    action = th.hstack([th.zeros((num_agent, 1)),pos_action])

    pos.append(env.position.clone())
    t.append(env.envs.t.clone())
    cvt_img = cv2.cvtColor(img[0].clip(max=255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Environment Render", cvt_img)
    cv2.waitKey(1)

    if done.any():
        print("Episode finished.")
        break

pos = th.stack(pos, dim=0)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
axeses = plt.subplots(3, 1)
t = th.stack(t, dim=0)[:,0]
plt.subplot(3, 1, 1)
plt.plot(t, pos[..., 0].cpu().numpy(), label='X Position')
plt.xlabel('Time')
plt.ylabel('X Position')
plt.legend()
# plt.subplot(3, 1, 2)
# plt.plot(t, pos[..., 1].cpu().numpy(), label='Y Position')
# plt.xlabel('Time')
# plt.ylabel('Y Position')
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.plot(t, pos[..., 2].cpu().numpy(), label='Z Position')
# plt.xlabel('Time')
# plt.ylabel('Z Position')
# plt.legend()
plt.tight_layout()
plt.show()





