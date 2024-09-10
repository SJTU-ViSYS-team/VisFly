# Introduction
VisFly is a versatile quadrotor simulator specialized for **visual-based** flight. 
As we all know, because of the high requirement of hardware for rendering the vision, the shortage of data is one significant limitations that hinder the development of intelligent drone. 
And this simulator is proposed to handle this issue. 

Based on habitat-sim, it has almost the highest FPS (1e4Hz, 64*64, RGB, Nvidia RTX 4090) as well as the abundant real-world scenes/objects. 
The detailed test result could be found in https://arxiv.org/abs/2407.14783.
Additionally, it is wrapped following the gym standards, integrated with really flexable interfaces. 
We hope you can find anything you want to customize your own environment with minor effort as less as possible.

This simulator contains differentiable dynamics modelling, 
which is considered as a promising research direction in the future. 
We have reserved the interface and will release it in our following research.

We will keep updating this project for more usages.

# Installation
## Clone the repository
Clone the repository to local.
```bash
git clone --recurse-submodule https://github.com/SJTU-ViSYS-team/VisFly
```
## Create an conda environment.
Open the terminal at the root of the project and run the following command:
```bash
conda env create -f environment.yml
```
## Install The Computational Geometry Algorithms Library (CAGL) dependencies.
Run the following command:
```bash
sudo apt-get install libcgal-dev
```
If you encounter any issues, please refer to the official installation steps on the website of [CAGL](https://www.cgal.org/download/linux.html). 
## Install modified habitat-sim.
Clone the modified habitat-sim source code:
```
git clone https://github.com/Fanxing-LI/habitat-sim
cd habitat-sim
```
Then please follow the steps in Section. **Build from Source** in [habitat-sim installation manual](https://github.com/Fanxing-LI/habitat-sim/blob/main/BUILD_FROM_SOURCE.md).

# Quick Start
## Download the Datasets for Demo
Down the dataset from [hugging face](https://huggingface.co/datasets/LiFanxing/VisFly). 
This demonstration dataset is created based on [Replica](https://github.com/facebookresearch/Replica-Dataset).
Here you need to register a hugging face acount and export your [access token](https://huggingface.co/blog/password-git-deprecation). 
```bash
cd datasets
git clone https://YourUsername:YourAccessToken@huggingface.co/datasets/LiFanxing/VisFly.git
mv VisFly/spy_datasets spy_datasets # move spy_datasets out of VisFly folder
```

## Run an Example
This is a simple example to show how to train an agent to fly in cluttered environment.
```bash
# Assume you are in the root of the project
python examples/cluttered_flight/rl.py -t 1
```
After training, the address of the model will be printed. 
And you can use the following command to test the model.
```bash
python examples/cluttered_flight/rl.py -t 0 -w ppo_1
```
Then you can find the result in the `examples/cluttered_flight/saved/test/` folder as the types of both figures and video.
The figures could be customized via `render_settings (line 121)` in the `examples/cluttered_flight/test.py`. 
The perspective and position of rendering camera in video could be set in `examples/cluttered_flight/rl.py`. 

# Simple Introduction
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
    },
    scene_kwargs={
        "path": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",  # datasets path
        "scene_type": "glb",  # assert scene_type in ["glb", "json"]

        # **use render_kwargs only when you need to output the rendered video or image.**
        "render_settings": {
            "object_path": "datasets/spy_datasets/configs/agents/DJI_Mavic_Mini_2.object_config.json",  # drone model path, or replace it with your own
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
            #         "mean": 0,
            #         "std": 0.01,
            #     }
            # }
        },
        "state_generator": {
            "class": "Uniform",  # assert in ["Uniform", "Normal"]
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
        pos_factor = -0.1 * 1/9
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
                # Noting that the structure of image extractor is different from the state extractor.
                # if use backbone
                "depth": {  # this name is same with the sensor uuid defined in the environment
                    "backbone": "resnet18",  # assert in ["resnet18", "resnet34", "resnet50", "resnet101", "efficientnet_l", "efficientnet_m", "efficientnet_s", "mobilenet_l", "mobilenet_m", "mobilenet_s"]
                    "mlp_layer": [128], # the MLP layer after the backbone
                },
                
                # if not use backbone
                "depth":{
                    "channels": [6,12,18],  # the channels of each CNN layer
                    "kernel_size": [5,3,3], # the kernel size of each CNN layer
                    "padding": [0,0,0],     # the padding of each CNN layer
                    "stride": [1,1,1],      # the stride of each CNN layer
                    "cnn_bn": False,        # Whether to use batch normalization in the CNN
                    "mlp_layer": [128],     # the MLP layer after the CNN
                    "bn": False,  # Union[List[Bool], Bool]. The length of list should be equal to the length of net_arch["pi"] if bool. 
                    "ln": False,  # All the bn and ln input format are consistent with this one.
                },
                
                "state": {
                    "mlp_layer": [128, 64],
                    "bn": False,
                    "ln": False,
                },
                "target": {
                    "mlp_layer": [128, 64],
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
        
    def get_observation(self, indices=None):
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
In the main script, three arguments are frequently used:
```
-t: 1 for training and 0 for testing.
-w: the address of the loading model. 
    if t is 1, the model will be loaded as the initail model parameters.
    if t is 0, the model will be loaded as the testing model.
-c: the commit information while saving the model, which will be added in the saving address. 
```
If you would like to create another main script, please do not forget to read the origin main script to grasp how these parameters work.

Let's make some regular schedules as examples to introduce more clearly.
In first training, the reward curve is not as expected. So we modify the net structure and want to retrain the model. 
For the convenience of comparison, you can directly use -c arguments to change the saving address.
```bash
python examples/cluttered_flight/rl.py -t 1 -c "second_structure"
```
The training process is monitored using tensorboard, and the trained model will be saved as `PPO_second_structure_1.zip`(algorithmName_commit_index). 
Noting that if `PPO_second_structure_1.zip` exists, the last number in address will improve unless this address will not conflict with any files.
Then, luckily, the reward curve shows an acceptable result. So we want to test the model.
```bash
python examples/cluttered_flight/rl.py -t 0 -w "PPO_second_structure_1"
```
Sometimes the fact implies that it is hard to well train the policy in one step.
That means we need to employ meta learning or curriculum learning which divides the training process into several stages.
First we teach the drone how to stably hover, so we define `get_reward()` as:
```python
def get_reward(self) -> th.Tensor:
    base_r = 0.1
    pos_factor = -0.1 * 1/9
    reward = (
            base_r +
             (self.position - self.target).norm(dim=1) * pos_factor +
             (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 +
             (self.velocity - 0).norm(dim=1) * -0.002 +
             (self.angular_velocity - 0).norm(dim=1) * -0.002
             + self._success * (self.max_episode_steps - self._step_count) * base_r 
    )
    return reward
```
Here, if you do not need the vision input, you can set `visual=False` while defining the environment, it accelerates the process significantly. 
Run in bash:
```bash
python examples/cluttered_flight/rl.py -t 1 -c "first_train" 
```

Now the agent has grasped basic flying skills. Then we train it to fly while avoiding the obstacles.
Change the `get_reward()` and use another cluttered scene datasets path:
```python
# in main.py
scene_path = "datasets/spy_datasets/configs/garage_simple_l_medium"

# in NavigationEnv.py
def get_reward(self) -> th.Tensor:
    # precise and stable target flight
    base_r = 0.1
    thrd_perce = th.pi/18
    reward = base_r*0 + \
            ((self.velocity *(self.target - self.position)).sum(dim=1) / (1e-6+(self.target - self.position).norm(dim=1))).clamp_max(10) * 0.01+\
             (((self.direction * self.velocity).sum(dim=1) / (1e-6+self.velocity.norm(dim=1)) / 1).clamp(-1., 1.).acos().clamp_min(thrd_perce)-thrd_perce)*-0.01+\
             (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001 + \
             (self.velocity - 0).norm(dim=1) * -0.002 + \
             (self.angular_velocity - 0).norm(dim=1) * -0.002 + \
             1 / (self.collision_dis + 0.2) * -0.01 + \
             (1-self.collision_dis ).relu() * ((self.collision_vector * (self.velocity - 0)).sum(dim=1) / (1e-6+self.collision_dis)).relu() * -0.005 + \
             self._success * (self.max_episode_steps - self._step_count) * base_r * (0.2+0.8/ (1+1*self.velocity.norm(dim=1)))
    return reward
```
Run in bash:
```bash
python examples/cluttered_flight/rl.py -t 1 -c "second_train" -w "first_train_1"
```
The well-trained model will be saved as `second_train_1.zip`. 
At present, you could transfer this model and verify in another simulator or in reality.
