import os.path
from abc import ABC, abstractmethod
import torch as th
from matplotlib import pyplot as plt
from typing import List, Union
import cv2
import sys
import numpy as np
import copy
from .FigFashion.FigFashion import FigFon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio  # 用于生成 GIF 数据流
from io import BytesIO  # 用于内存操作

FigFon.set_fashion("IEEE")


def render_fig(fig):
    # set low dpi
    fig.set_dpi(50)
    canvas = FigureCanvas(fig)
    canvas.draw()  # 渲染图形
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 转换为 (height, width, 3) 形状

    return image_array

# def create_gif(image_list):
#     gif_buffer = BytesIO()  # 创建一个内存缓冲区
#     with imageio.get_writer(gif_buffer, mode='I', format='gif', duration=0.5) as writer:
#         for image in image_list:
#             writer.append_data(image)  # 将每张图片写入 GIF 数据流
#
#     # 2. 将 GIF 数据流转换为 NumPy 数组
#     gif_buffer.seek(0)  # 将指针移回缓冲区开头
#     gif_array = imageio.mimread(gif_buffer)  # 读取 GIF 数据流为 NumPy 数组列表
#     gif_array = np.stack(gif_array)
#     return gif_array

# def create_movement_image(images):
#     """
#     create movement images by merging images
#     """
#     image = []
#     for i in range(len(images)):
#         if i == 0:
#             image = images[i]
#         else:
#             image = np.hstack((image, images[i]))
#     return image

class TestBase:
    def __init__(
            self,
            env=None,
            model=None,
            name: Union[List[str], None] = None,
            save_path: str = None,

    ):
        self.save_path = os.path.join(save_path, name) if save_path is not None else os.path.dirname(os.path.abspath(sys.argv[0])) + "/saved/test/" + name
        if self.save_path.endswith((".zip", ".rar", ".pth")):
            self.save_path = self.save_path[:-4]
        self.model = model
        self.env = env
        self.name = name

        self.obs_all = []
        self.state_all = []
        self.info_all = []
        self.action_all = []
        self.collision_all = []
        self.render_image_all = []
        self.reward_all = []
        self.t = []

    def test(
            self,
            policy=None,
            world=None,
            # model=None,
            is_fig: bool = False,
            is_video: bool = False,
            is_sub_video: bool = False,
            is_fig_save: bool = False,
            is_video_save: bool = False,
            render_kwargs={},
            
    ):
        if is_fig_save:
            if not is_fig:
                raise ValueError("is_fig_save must be True if is_fig is True")
        if is_video_save:
            if not is_video:
                raise ValueError("is_video_save must be True if is_video is True")
        if policy is None:
            policy = self.model.policy
        if self.env is not None:
            env = self.env
        else:
            env = self.model.env
        # done_all = th.full((env.num_envs,), False)
        obs = env.reset(is_test=True)
        self._img_names = [name for name in obs.keys() if (("color" in name) or ("depth" in name) or ("semantic" in name))]
        self.obs_all.append(obs)
        self.state_all.append(env.state)
        self.info_all.append([{} for _ in range(env.num_envs)])
        self.t.append(env.t.clone())
        self.collision_all.append({"col_dis": env.collision_dis,
                                   "is_col": env.is_collision,
                                   "col_pt": env.collision_point})
        agent_index = [i for i in range(env.num_agent)]
        self.eq_r = []
        self.eq_l = []

        while True:
            with th.no_grad():
                action = policy.predict(obs)
                if isinstance(action, tuple):
                    action = action[0]
                # obs, reward, done, info = env.step(action, is_test=True)
                if world is not None:
                    obs, reward, done, info = env.step(action, is_test=True, latent_func=world.step)
                else:
                    obs, reward, done, info = env.step(action, is_test=True)
                 # = env.get_observation(), env.reward, env.done, env.info
                col_dis, is_col, col_pt = env.collision_dis, env.is_collision, env.collision_point
                state = env.state
                self.collision_all.append({"col_dis": col_dis, "is_col": is_col, "col_pt": col_pt})

            self.reward_all.append(reward)
            self.action_all.append(action)
            self.state_all.append(state)
            self.obs_all.append(obs)
            self.info_all.append(copy.deepcopy(info))
            self.t.append(env.t.clone())
            if env.visual:
                render_image = cv2.cvtColor(env.render(**render_kwargs)[0], cv2.COLOR_RGBA2RGB)
                self.render_image_all.append(render_image)
            # done_all[done] = True

            for i in reversed(agent_index):
                if done[i]:
                    self.eq_r.append(info[i]['episode']['r'].item())
                    self.eq_l.append(info[i]['episode']['l'].item())
                    agent_index.remove(i)

            if len(agent_index) == 0:
                break

        mean_r = th.as_tensor(self.eq_r,dtype=th.float32).mean().item()
        mean_l = th.as_tensor(self.eq_l,dtype=th.float32).mean().item()
        # print(f"Average Rewards:{mean_r}, Average Length:{mean_l}")

        if is_fig:
            figs = self.draw()
            if is_fig_save:
                for i, fig in enumerate(figs):
                    self.save_fig(fig, c=i)
        else:
            figs = []
        if is_video:
            self.play(is_sub_video=is_sub_video)
            if is_video_save:
                self.save_video()

        render_video = th.as_tensor(np.stack(self.render_image_all, axis=0)).unsqueeze(0) if len(self.render_image_all) > 0 else None
        return figs, render_video, mean_r, mean_l



    @abstractmethod
    def draw(self, names: Union[List[str], None] = "video") -> plt.Figure:
        raise NotImplementedError

    # @abstractmethod
    def play(self, render_name: Union[List[str], None] = "video",is_sub_video=False):
        """
        how to play the video
        """
        """
        how to draw the figures
        """
        for image, t, obs in zip(self.render_image_all, self.t, self.obs_all):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(winname=render_name, mat=image)
            if is_sub_video:
                for name in self._img_names:
                    # Convert CUDA tensor to CPU before numpy operations
                    obs_data = obs[name]
                    if hasattr(obs_data, 'cpu'):
                        obs_data = obs_data.cpu()
                    if hasattr(obs_data, 'numpy'):
                        obs_data = obs_data.numpy()
                    
                    cv2.imshow(winname=name,
                               mat=np.hstack(np.transpose(obs_data, (0,2,3,1) ))
                               )
            cv2.waitKey(int(self.env.envs.dynamics.ctrl_dt * 1000))

    def save_fig(self, fig, path=None, c=""):
        path = path if path is not None else self.save_path
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(f"{path}/{c}.png")
        print(f"fig saved in {path}/{c}.png")

    def save_video(self, is_sub_video=False):
        height, width, layers = self.render_image_all[0].shape
        names = self.name if self.name is not None else "video"

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(f"{self.save_path}/cache"):
            os.makedirs(f"{self.save_path}/cache")

        # render video
        path = f"{self.save_path}/video.mp4"
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), int(1/self.env.envs.dynamics.dt), (width, height))
        # obs video
        path_obs = []
        video_obs = []
        if is_sub_video:
            for name in self._img_names:
                path_obs.append(f"{self.save_path}/{name}.mp4")
                width, height = self.obs_all[0][name].shape[3]*self.obs_all[0][name].shape[0], self.obs_all[0][name].shape[2]
                video_obs.append(cv2.VideoWriter(path_obs[-1], cv2.VideoWriter_fourcc(*'mp4v'), int(1/self.env.dynamics.dt), (width, height)))

        # 将图片写入视频
        for index, (image, t, obs) in enumerate(zip(self.render_image_all, self.t, self.obs_all)):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image)
            if is_sub_video:
                for i, name in enumerate(self._img_names):
                    # Convert CUDA tensor to CPU before numpy operations
                    obs_data = obs[name]
                    if hasattr(obs_data, 'cpu'):
                        obs_data = obs_data.cpu()
                    if hasattr(obs_data, 'numpy'):
                        obs_data = obs_data.numpy()
                    
                    if "depth" in name:
                        max_depth = 10
                        img = np.clip(np.hstack(np.transpose(obs_data, (0, 2, 3, 1))),None, max_depth)
                        img = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)*255/max_depth).astype(np.uint8)
                        video_obs[i].write(img)
                    elif "color" in name:
                        img = np.hstack(np.transpose(obs_data, (0, 2, 3, 1)))
                        img = img.astype(np.uint8)
                        video_obs[i].write(img)
                        # img = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).astype(np.uint8)
                        video_obs[i].write(img)

            # save image in cache
            # if index % 4 == 0:
            #     cv2.imwrite(f"{self.save_path}/cache/raw_{index}.jpg", image)

        video.release()
        if is_sub_video:
            for i in range(len(video_obs)):
                video_obs[i].release()

        print(f"video saved in {path}")
        # raise NotImplementedError
