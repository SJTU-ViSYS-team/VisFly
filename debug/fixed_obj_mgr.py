import habitat_sim
import sys
import os
import torch as th
sys.path.append('/home/simonwsy/new_VisFly')

from VisFly.envs.HoverEnv import HoverEnv2

# Try to create the environment with proper error handling
try:
    scene_kwargs = {
        "path":"VisFly/datasets/spy_datasets/configs/garage_empty",
        "sensor_settings":[{
                    "sensor_type": habitat_sim.SensorType.DEPTH,
                    "uuid": "depth",
                    "resolution": [64, 64],
                }],
        "render_settings":{

        },
        "obj_settings":{
            "path":"aa",
            "mode": "static",
            "mode_kwargs": {
                "position": {"mean": [0., 0., 2.2], "half": [1.5, 1.5, 0.5]},
            },
            "generator": "Uniform",
            "generator_kwargs": {

            },

        }

    }

    env = HoverEnv2(
        scene_kwargs=scene_kwargs,
        num_agent_per_scene=1,
        num_scene=1,
    )
    
    print("✓ HoverEnv2 with object management created successfully!")
    
    # Test basic operations
    env.reset()
    print("✓ Environment reset successful!")
    
    # Test a few steps
    for i in range(10):
        action = th.rand((1, 4))  # Random action for 1 agent
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Reward = {reward}, Done = {done}")
    
    print("✓ Object management test completed successfully!")
    
except Exception as e:
    print(f"✗ Object management test failed: {e}")
    
    # Fallback to simple environment without scene
    try:
        print("Trying fallback to simple environment...")
        env = HoverEnv2(
            num_agent_per_scene=1,
            num_scene=1,
            visual=False,  # Disable visual to avoid scene issues
        )
        env.reset()
        print("✓ Fallback environment created and reset successfully!")
        
        # Test a few steps
        for i in range(5):
            action = th.rand((1, 4))
            obs, reward, done, info = env.step(action)
            print(f"Step {i}: Reward = {reward}, Done = {done}")
        
        print("✓ Fallback test completed successfully!")
        
    except Exception as e2:
        print(f"✗ Fallback test also failed: {e2}")

print("Object management test completed!") 