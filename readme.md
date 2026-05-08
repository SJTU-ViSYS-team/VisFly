# Introduction

VisFly is a versatile quadrotor simulator for vision-based flight with support for differentiable dynamics.
Vision-based drone learning is often limited by the cost of rendering large-scale visual data; VisFly reduces that bottleneck by combining high-throughput simulation with configurable visual environments.

Built on Habitat-Sim, VisFly supports fast RGB/depth rendering in rich indoor scenes and object layouts. On an NVIDIA RTX 4090 it can reach around `1e4 Hz` for `64×64` RGB observations in our benchmark setting. Detailed results are available in our paper: https://arxiv.org/abs/2407.14783.

The simulator exposes Gym-style interfaces and provides flexible configuration hooks for environments, dynamics, observations, rewards, and training pipelines. It also provides differentiable dynamics models that enable gradient-based policy optimization and planning.

# Notice

VisFly is under active development; public interfaces and example layouts may change over time. Please report issues via GitHub Issues (preferred) or by email so we can address them in future updates.

# Installation

## Clone the repository

Create a project folder and clone this repository:

```bash
mkdir project_root && cd project_root
git clone --recurse-submodules https://github.com/SJTU-ViSYS-team/VisFly
```

## Create a Conda environment

Change into the `VisFly` directory and create the environment:

```bash
cd VisFly
conda env create -f environment.yml
```

## Download the demo datasets

The demo datasets are built from ReplicaCAD and are hosted on Hugging Face. Large files in the dataset are stored with Git LFS.

You will need a Hugging Face account and an access token. To download the demo dataset:

```bash
git lfs install
cd datasets
git clone https://YourUsername:YourAccessToken@huggingface.co/datasets/LiFanxing/visfly-beta.git
cd visfly-beta
git lfs pull
```

After cloning, the dataset should be available at `VisFly/datasets/visfly-beta`.

## Install Rendering Engine

### CGAL dependencies

Install the Computational Geometry Algorithms Library (CGAL) system dependency:

```bash
sudo apt-get install libcgal-dev
```

If you run into issues, see the CGAL installation guide: https://www.cgal.org/download/linux.html.

### Install the modified habitat-sim

Clone the modified `habitat-sim` repository into a separate folder and follow the project's build instructions:

```bash
cd ~/Downloads
git clone https://github.com/Fanxing-LI/habitat-sim
cd habitat-sim
# Follow the "Build from Source" section in BUILD_FROM_SOURCE.md
```

Follow the steps in the repository's BUILD_FROM_SOURCE.md to build `habitat-sim` from source.


# Projects and Papers Using VisFly

- [VisFly-Lab: Unified Differentiable Framework for First-Order Reinforcement Learning of Quadrotor Control](https://github.com/Fanxing-LI/APG)
- [StableTracker: Learning to Stably Track Target via Differentiable Simulation](https://github.com/Fanxing-LI/obj_track)
- [Simple but Stable, Fast and Safe: Achieve End-to-end Control by High-Fidelity Differentiable Simulation](https://github.com/Fanxing-LI/avoidance)
- [Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles](https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing)

# Quick Start

## Run an example

The example runner is `VisFly/exps/examples/run.py`.

Activate the environment:

```bash
conda activate visfly
```

From the project root, train a PPO policy for `cluttered_flight`:

```bash
python VisFly/exps/examples/run.py -t 1 -e cluttered_flight
```

Evaluate a trained checkpoint:

```bash
python VisFly/exps/examples/run.py -t 0 -e cluttered_flight -w PPO_std_1
```

Other examples use the same entry point:

```bash
python VisFly/exps/examples/run.py -t 1 -e crossing
python VisFly/exps/examples/run.py -t 1 -e landing
```

Training outputs are saved under `VisFly/exps/examples/saved/<env_name>/`. Environment settings live in `VisFly/exps/examples/env_cfgs/<env_name>.yaml` and algorithm settings in `VisFly/exps/examples/alg_cfgs/<env_name>/PPO.yaml`. Rendering options for evaluation can be adjusted in the `eval_env.scene_kwargs.render_settings` section of the environment config.

# Citation

If VisFly is useful for your research, please consider citing the paper:

```
@misc{li2024visflyefficientversatilesimulator,
    title={VisFly: An Efficient and Versatile Simulator for Training Vision-based Flight},
    author={Fanxing Li and Fangyu Sun and Tianbao Zhang and Danping Zou},
    year={2024},
    eprint={2407.14783},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2407.14783},
}
```

# Notes and API

Note: not all interfaces may be fully complete — VisFly is maintained with limited resources. For details and usage examples, refer to the projects and papers that build on VisFly (see "Projects and Papers Using VisFly").

## Complete Environment Definition

This is a complete definition of the environment of NavigationEnv.

```python
from envs.NavigationEnv import NavigationEnv
import habitat_sim
import torch as th
from utils.type import Normal, Uniform

env = NavigationEnv(
    num_scene=1,  # num of scenes
    num_agent_per_scene=1,  # num of agents in each scene
    visual=True,  # whether to render the camera
    device="cuda",
    max_episode_steps=256,
    # requires_grad=False,    # interface for further update
    seed=42,

    dynamics_kwargs={
        "action_type": "bodyrate",  # assert action_type in ["bodyrate", "thrust", "velocity", "position"]
        "ori_output_type": "quaternion",  # assert ori_output_type in ["quaternion", "euler"]
        "dt": 0.0025,  # simulation interval
        "ctrl_dt": 0.02,  # control interval
        "integrator": "euler",  # assert Integrator in ["euler", "rk4"]
        "ctrl_delay": True, # whether considering the control delay over time
    },
    scene_kwargs={
        "path": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",  # datasets path
        "scene_type": "glb",  # assert scene_type in ["glb", "json"]

        # **use render_kwargs only when you need to output the rendered video or image.**
        "render_settings": {
            "object_path": "datasets/visfly-beta/configs/agents/DJI_Mavic_Mini_2.object_config.json",  # drone model path, or replace it with your own
            "mode": "fix",  # assert mode in ["fix", "follow"] 
            "view": "custom",  # assert view in ["custom", "top", "near", "side", "back] when mode is "fix", ["near", "back"] when mode is "follow"
            "resolution": [1080, 1920],
            "sensor_type": habitat_sim.SensorType.COLOR,  # assert sensor_type in ["COLOR", "DEPTH"]
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),  # Initial position and focus position of the camera when view is "custom"
            "line_width": 6.,  # line width of all the lines drawn in the image
            "axes": False,  # whether to draw the axes
            "trajectory": True,  # whether to draw the trajectory of drones
        }
    },
    sensor_kwargs=[
        {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "uuid": "depth",  # uuid should include the string of type ["depth", "color", "semantic"]
            "resolution": [64, 64],
            "position": [0, 0, 0],
            "orientation": [0, 0, 0],
        },
        # {
        #     "sensor_type": habitat_sim.SensorType.COLOR,
        #     "uuid": "color_0",
        #     "resolution": [64, 64],
        #     "position": [0, 0, 0],
        #     "orientation": [0, 0, 0],
        # },
        # etc.. add more sensors if needed
    ],
    random_kwargs={
        "noise_kwargs": {
            # please refer to the image noise setting in habitat-sim: https://aihabitat.org/docs/habitat-sim/habitat_sim.sensors.noise_models.html
            "depth": {
                "model": "RedwoodDepthNoiseModel",
                "kwargs": {}
            },
            # "color": {
            #     "model": "GaussianNoiseModel",
            #     "kwargs": {}
            # },
            # "IMU": {
            #     "model": "GaussianNoiseModel", # assert model in ["GaussianNoiseModel", "UniformNoiseModel"]
            #     "kwargs": {
            #         "mean": {0,0,...}, # same dim with state
            #         "std": {0,0,...},
            #     }
            # }
        },
        "state_generator": {
            "class": "Uniform",  # assert in ["Uniform", "Normal", "Union"]
            # if this env is a single agent env, the length of state_generator_kwargs should be 1 or equal to num_env or equal to num_env*num_agent_per_env.
            "kwargs": [
                {
                    "position": {"mean": [1., 0., 1.5], "half": [0.0, 2., 1.]},
                    "orientation": {"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                    "velocity": {"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                    "angular_velocity": {"mean": [0., 0., 0.], "half": [0., 0., 0.]},
                },
            ]
        }
        # etc.. add more state generators if needed, the length of multiple state_generator_kwargs should be equal to num_env.
    },

    # Other self-defined variables
    target=th.as_tensor([[9, 0., 1]]),  # target position

)
```

## Create Scenes

Please refer to the official documentation of habitat-sim: https://aihabitat.org/docs/habitat-sim/attributesJSON.html.
All the scenes used in the demo are created via objects in ReplicaCAD datasets.

## Customize Your Own Environment

We make `NavigationEnv` as an example to show how to customize your own environment.

```python
from envs.droneGymEnv import DroneGymEnvsBase
from typing import Dict, Optional
import numpy as np
import torch as th
from gym import spaces


class NavigationEnv(DroneGymEnvsBase):
    def __init__(
            self,
            num_agent_per_scene: int = 1,
            num_scene: int = 1,
            seed: int = 42,
            visual: bool = True,
            requires_grad: bool = False,
            random_kwargs: dict = {},
            dynamics_kwargs: dict = {},
            scene_kwargs: dict = {},
            sensor_kwargs: list = [],
            device: str = "cpu",
            target: Optional[th.Tensor] = None,
            max_episode_steps: int = 256,
            # latent_dim=None,
    ):
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            # latent_dim=latent_dim,
        )

        # define new variables. In this example, we define a target position for each agent.
        self.target = th.ones((self.num_envs, 1)) @ th.tensor([[9, 0., 1]]) if target is None else target

        # add observation space. Any new observation should be added here.
        self.observation_space["target"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # other variables
        self.success_radius = 0.5

    # define observation function
    def get_observation(
            self,
            indices=None
    ) -> Dict:
        # you can pre-process the observation here, like encoding the image using well-trained net or backbone.
        # ...

        return {
            "state": self.sensor_obs["IMU"].cpu().clone().numpy(),
            "depth": self.sensor_obs["depth"],
            "target": self.target.cpu().clone().numpy(),
        }

    # define success condition
    def get_success(self) -> th.Tensor:
        return (self.position - self.target).norm(dim=1) <= self.success_radius

    # define reward function
    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1 / 9
        reward = (
                base_r +
                (self.position - self.target).norm(dim=1) * pos_factor +
                (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
                (self.velocity - 0).norm(dim=1) * -0.002 +
                (self.angular_velocity - 0).norm(dim=1) * -0.002
                + self._success * (self.max_episode_steps - self._step_count) * base_r
            # + 1 / (self.collision_dis + 0.2) * -0.01

        )
        return reward

    # reset agents by indices
    def reset_agent_by_id(self, indices=None, state=None, reset_obs=None):
        super().reset_agent_by_id(agent_indices=None, state=None, reset_obs=None)
        # if your agent have any new private observation that should be reset. Please add here. RacingEnv.py provides an example.
  
    # reset scenes by indices
    def reset_env_by_id(self, scene_indices=None):
        super().reset_env_by_id(scene_indices=scene_indices)
  
    # reset all the agents and scenes, noting that here if you have a scene dataset, VisFly will automatically load to other data in this dataset.
    def reset(self, state=None, obs=None):
        super().reset(state=state, obs=obs)

```

Except from pose(position, linear_velocity, orientation, angular_velocity), the closest obstacle is also recorded as properties, including `self.collision_point`, `self.collision_vector`, `self.collision_dis`.
It probably will be helpful to train the capacity of avoiding obstacles.

## Customize Your Own Policy

Once the environment is defined, we could employ a reinforcement learning algorithm to train the agent. Here we directly use the PPO algorithm in the stable-baselines3 (SB3) library.
Different from SB3, considering requirement of any innovative attempts, we provide a more flexible interface to customize the policy network.

```python
from stable_baselines3.ppo import PPO
import torch
import utils.policies.extractors as extractors

model = PPO(
    policy="CustomMultiInputPolicy",
    policy_kwargs=dict(
        # --- Define the Feature Extractor ---
        # You can define the feature extractor here of both the actor and the critic by only input features_extractor_class and features_extractor_kwargs.
        features_extractor_class=extractors.StateTargetImageExtractor,  
        # assert in ["StateExtractor", "StateTargetExtractor", "StateTargetImageExtractor", "StateImageExtractor", "ImageExtractor", "TargetExtractor"]
        features_extractor_kwargs={
            "net_arch": {
                # Note that the image extractor structure is different from the state extractor structure.
                # if use backbone
                "depth": {  # this name is same with the sensor uuid defined in the environment
                    "backbone": "resnet18",  # assert in ["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_l", "efficientnet_m", "efficientnet_s", "mobilenet_l", "mobilenet_m", "mobilenet_s"]
                    "layer": [128], # the MLP layer after the backbone
                },
        
                # if not use backbone
                "depth":{
                    "channels": [6,12,18],  # the channels of each CNN layer
                    "kernel_size": [5,3,3], # the kernel size of each CNN layer
                    "padding": [0,0,0],     # the padding of each CNN layer
                    "stride": [1,1,1],      # the stride of each CNN layer
                    "cnn_bn": False,        # Whether to use batch normalization in the CNN
                    "layer": [128],     # the MLP layer after the CNN
                    "bn": False,  # Union[List[Bool], Bool]. The length of list should be equal to the length of layer if bool. 
                    "ln": False,  # All the bn and ln input format are consistent with this one.
                },
        
                "state": {
                    "layer": [128, 64],
                    "bn": False,
                    "ln": False,
                },
                "target": {
                    "layer": [128, 64],
                    "bn": False,
                    "ln": False,
                },
                # "recurrent":{     # reserved interface for recurrent networks
                #     "class": "GRU",
                #     "kwargs":{
                #         "hidden_size": latent_dim,
                #     }
                # }
            },
            "activation_fn": torch.nn.ReLU,
        },
        share_features_extractor=True,  # whether to share the feature extractor between the actor and the critic

        # ** If two extractors are different, separately define pi_features_extractor_class, pi_features_extractor_kwargs, **
        # ** vf_features_extractor_class, vf_features_extractor_kwargs, while both features_extractor_class and            **
        # ** features_extractor_kwargs should be None.                                                                     **
        # pi_features_extractor_class = None,
        # pi_features_extractor_kwargs = None,
        # vf_features_extractor_class = None,
        # vf_features_extractor_kwargs = None,

        # --- Define the MLP Extractor ---
        net_arch={
            "pi": [64, 64],
            "vf": [64, 64],
            "pi_bn": False,  # Union[List[Bool], Bool]. The length of list should be equal to the length of net_arch["pi"] if bool.
            "vf_bn": False,
            "pi_ln": False,  # Union[List[Bool], Bool]. The length of list should be equal to the length of net_arch["pi"] if bool.
            "vf_ln": False,
        },

        # --- Define the Activation Function ---
        activation_fn=torch.nn.ReLU,

        # --- Define the Optimizer ---
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={
            "weight_decay": 1e-5
        },
    ),
    env=env,
    # etc... other hyperparameters
)

```

### Customize New Feature Extractor

Assuming that we put forward or just want to try an innovation framework, where five previous actions should be taken into observation.
However, this simulator has not built-in such a feature extractor that takes it as input.
So we need to define a new feature extractor in `utils/policies/extractor.py`. Open this file and add the following script at the end:

```python
from utils.policies.extractors import StateTargetImageExtractor, set_mlp_feature_extractor, set_cnn_feature_extractor
import torch as th

class ActionStateTargetImageExtractor(StateTargetImageExtractor):
    def __init__(self, observation_space, net_arch, activation_fn):
        assert "action" in observation_space.spaces, "The action space should be included in the observation space."
        super(ActionStateTargetImageExtractor, self).__init__(observation_space, net_arch, activation_fn)
  
    def _build(self, observation_space, net_arch, activation_fn):
        super()._build(observation_space, net_arch, activation_fn)
        _action_features_dim = set_mlp_feature_extractor(   # This function will create a 
            self,
            "action",   # extractor name
            observation_space["action"],  
            net_arch["action"], 
            activation_fn
        )
        self._features_dim += _action_features_dim

    def extract(self, observation) -> th.Tensor:
        action_features = self.action_extractor(observation["action"])
        super_features = super().extract(observation)
        return th.cat([super_features, action_features], dim=1)
```

At this time, we can use this new feature extractor in the policy network.
**DO NOT** forget to create new environment and complement the observation space.

```python
from envs.NavigationEnv import NavigationEnv

# Create a TensorStack to record past actions, 
# This class is just an example, which is not verified.
class TensorStack: 
    def __init__(self, length):
        self.length = length
        self.xx = []
  
    def push(self, data):
        pass
  
    def pop(self):
        pass
  
    @property
    def output(self):
        return th.cat(self.xx)

  
class Observe_Action_NavigationEnv(NavigationEnv):
    def __init__(self, *args, **kwargs):
        super(Observe_Action_NavigationEnv, self).__init__(*args, **kwargs)
        self.observation_space["action"] = spaces.Box(low=-1, high=1, shape=(5, 4), dtype=np.float32)  # shape:horizon length 5, action length 4
  
        # save the receding horizon of actions
        self._past_actions = TensorStack(5)
  
    def get_observation(self, indices=None, predicted_obs=None):
        return {
            "state": self.state.cpu().clone().numpy(),
            "depth": self.sensor_obs["depth"],
            "target": self.target.cpu().clone().numpy(),
            "action": self._past_actions.output,
        }
  
    def step(self, _action, is_test=False):
        super().step(_action, is_test)
        self._past_actions.push(_action)
```

## Arguments

In `exps/examples/run.py`, the most commonly used arguments are:

```
-t: 1 for training and 0 for testing.
-e: the example environment name, such as cluttered_flight, crossing, or landing.
-w: the checkpoint name to load.
    If -t is 1, the checkpoint is used as the initial policy.
    If -t is 0, the checkpoint is used for evaluation.
-c: an optional comment added to the saved checkpoint and log directory name.
```

If you create a custom runner, use `exps/examples/run.py` as the reference implementation for loading configs, resolving aliases, and saving checkpoints.

The following examples show typical training schedules.
If the first training curve is not satisfactory, you may adjust the network structure in `exps/examples/alg_cfgs/cluttered_flight/PPO.yaml` and retrain with a different comment:

```bash
python VisFly/exps/examples/run.py -t 1 -e cluttered_flight -c "second_structure"
```

Training logs can be monitored with TensorBoard, and checkpoints are saved under `VisFly/exps/examples/cluttered_flight/saved/`.
If `PPO_second_structure_1.zip` already exists, VisFly increments the final index to avoid overwriting existing files.

After training, evaluate the checkpoint with:

```bash
python VisFly/exps/examples/run.py -t 0 -e cluttered_flight -w "PPO_second_structure_1"
```

Some policies are difficult to train in a single stage.
In that case, curriculum learning can split training into several phases.
For example, first train the drone to hover steadily by defining `get_reward()` as:

```python
def get_reward(self) -> th.Tensor:
    base_r = 0.1
    pos_factor = -0.1 * 1 / 9
    reward = (
            base_r +
            (self.position - self.target).norm(dim=1) * pos_factor +
            (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
            (self.const_velocity - 0).norm(dim=1) * -0.002 +
            (self.angular_velocity - 0).norm(dim=1) * -0.002
            + self._success * (self.max_episode_steps - self._step_count) * base_r
    )
    return reward
```

If vision input is not needed during this stage, set `visual: False` in `exps/examples/env_cfgs/cluttered_flight.yaml` to speed up training.
Run in bash:

```bash
python VisFly/exps/examples/run.py -t 1 -e cluttered_flight -c "first_train"
```

After the agent learns basic flight, train it to avoid obstacles.
Update `get_reward()` and set a cluttered scene path in `exps/examples/env_cfgs/cluttered_flight.yaml`:

```python
# in exps/env_cfgs/cluttered_flight.yaml
scene_kwargs:
  path: VisFly/datasets/visfly-beta/configs/scenes/garage_simple_l_medium


# in NavigationEnv.py
def get_reward(self) -> th.Tensor:
    # precise and stable target flight
    base_r = 0.1
    thrd_perce = th.pi / 18
    reward = base_r * 0 +
             ((self.const_velocity * (self.target - self.position)).sum(dim=1) / (1e-6 + (self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01 +
             (((self.direction * self.const_velocity).sum(dim=1) / (1e-6 + self.const_velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce) - thrd_perce) * -0.01 +
             (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
             (self.const_velocity - 0).norm(dim=1) * -0.002 +
             (self.angular_velocity - 0).norm(dim=1) * -0.002 +
             1 / (self.collision_dis + 0.2) * -0.01 +
             (1 - self.collision_dis).relu() * ((self.collision_vector * (self.const_velocity - 0)).sum(dim=1) / (1e-6 + self.collision_dis)).relu() * -0.005 +
             self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2 + 0.8 / (1 + 1 * self.const_velocity.norm(dim=1)))
    return reward
```

Run in bash:

```bash
python VisFly/exps/examples/run.py -t 1 -e cluttered_flight -c "second_train" -w "first_train_1"
```

The trained model will be saved as `second_train_1.zip`.
You can then transfer the model for validation in another simulator or on real hardware.
