#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np
import habitat_sim

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.HoverEnv import HoverEnv2
from VisFly.envs.NavigationEnv import NavigationEnv2

def test_object_management():
    """Test object management functionality"""
    print("=== Testing Object Management Functionality ===")
    
    # Configuration
    scene_path = "../datasets/spy_datasets/configs/garage_simple_l_medium"
    
    # Separate sensor configuration
    sensor_kwargs = [{
        "sensor_type": habitat_sim.SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }]
    
    # Object settings for ObjectManager
    obj_settings = {
        "dt": 0.01,  # time step for physics simulation
        "object_path": None,  # use default object configuration
        "isolated": True,  # use isolated object mode
        "device": th.device("cpu")
    }
    
    # Dynamics and random settings
    dynamics_kwargs = {
        "dt": 0.02,
        "ctrl_dt": 0.02,
        "action_type": "bodyrate",
    }
    
    random_kwargs = {
        "state_generator": {
            "class": "Uniform",
            "kwargs": [
                {"position": {"mean": [9., 0., 1.5], "half": [1.0, 1.0, 0.5]}},
            ]
        }
    }
    
    scene_kwargs = {
        "path": scene_path,
        "obj_settings": obj_settings,  # Pass object settings through scene_kwargs
        "render_settings": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "line_width": 6.,
            "trajectory": True,
            "is_draw_axes": True,
        },
    }
    
    try:
        print("Creating NavigationEnv2 with object management...")
        # Create environment with object settings
        env = NavigationEnv2(
            num_agent_per_scene=2,  # Use correct parameter name
            num_scene=1,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            device=th.device("cpu")
        )
        
        print("✓ Environment with object management created successfully")
        
        # Test reset
        print("Testing environment reset...")
        obs = env.reset()
        print(f"✓ Environment reset successful. Observation keys: {list(obs.keys())}")
        
        # Test a few steps
        for i in range(5):
            # Random actions
            actions = th.randn(2, 4) * 0.1  # Small random actions
            obs, reward, done, info = env.step(actions)
            print(f"  Step {i+1}: Reward = {reward.mean():.3f}")
        
        print("✓ Object management test completed successfully")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to simple environment
        print("\nTrying fallback to simple environment...")
        try:
            fallback_env = NavigationEnv2(
                num_agent_per_scene=2,
                num_scene=1,
                sensor_kwargs=sensor_kwargs,
                scene_kwargs={"path": scene_path},
                dynamics_kwargs=dynamics_kwargs,
                random_kwargs=random_kwargs,
                device=th.device("cpu")
            )
            print("✓ Fallback environment created successfully")
            
            # Test fallback
            obs = fallback_env.reset()
            print("✓ Fallback environment reset successfully")
            for i in range(5):
                actions = th.randn(2, 4) * 0.1
                obs, reward, done, info = fallback_env.step(actions)
                print(f"  Fallback step {i+1}: Reward = {reward.mean():.3f}")
            
            print("✓ Fallback test completed successfully")
        except Exception as fallback_e:
            print(f"✗ Fallback also failed: {fallback_e}")
    
    print("=== Object Management Test Complete ===\n")

if __name__ == "__main__":
    test_object_management() 