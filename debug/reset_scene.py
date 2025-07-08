#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
from habitat_sim.sensor import SensorType

def test_scene_reset():
    """Test scene reset functionality"""
    print("=== Testing Scene Reset Functionality ===")
    
    # Configuration
    scene_path = "../datasets/spy_datasets/configs/garage_simple_l_medium"
    sensor_kwargs = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }]
    
    random_kwargs = {
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {"position": {"mean": [1., 0., 1.5], "half": [2.0, 2.0, 1.0]}},
            ]
        }
    }
    
    scene_kwargs = {
        "path": scene_path,
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "line_width": 6.,
            "trajectory": True,
        }
    }
    
    try:
        # Create environment
        print("Creating environment...")
        env = NavigationEnv2(
            visual=True,
            num_scene=1,
            num_agent_per_scene=4,
            random_kwargs=random_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            dynamics_kwargs={}
        )
        
        print("✓ Environment created successfully")
        
        # Test initial reset
        print("\nTesting initial reset...")
        obs = env.reset()
        initial_positions = env.position.clone()
        print(f"✓ Initial reset successful")
        print(f"  Initial positions: {initial_positions}")
        
        # Test a few steps
        print("\nTesting environment steps...")
        for i in range(5):
            action = th.rand((4, 4))
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+1}: Position = {env.position[0]}")
        
        # Test scene reset
        print("\nTesting scene reset...")
        env.reset()
        reset_positions = env.position.clone()
        print(f"✓ Scene reset successful")
        print(f"  Reset positions: {reset_positions}")
        
        # Test agent-specific reset
        print("\nTesting agent-specific reset...")
        agent_indices = [0, 2]  # Reset agents 0 and 2
        env.reset_agent_by_id(agent_indices)
        print(f"✓ Agent-specific reset successful")
        print(f"  Positions after agent reset: {env.position}")
        
        # Test scene-specific reset
        print("\nTesting scene-specific reset...")
        scene_indices = [0]  # Reset scene 0
        env.reset_env_by_id(scene_indices)
        print(f"✓ Scene-specific reset successful")
        print(f"  Positions after scene reset: {env.position}")
        
        print("\n✓ All reset tests passed!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scene_reset() 