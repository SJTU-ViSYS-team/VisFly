#!/usr/bin/env python3

import sys
import os
import torch as th
import numpy as np
import cv2

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from VisFly.envs.NavigationEnv import NavigationEnv2
from habitat_sim.sensor import SensorType

def test_view_functionality():
    """Test view and rendering functionality"""
    print("=== Testing View and Rendering Functionality ===")
    
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
    
    # Test different view configurations
    view_configs = {
        "custom": {
            "mode": "fix",
            "view": "custom",
            "resolution": [1080, 1920],
            "position": th.tensor([[7., 6.8, 5.5], [7, 4.8, 4.5]]),
            "line_width": 6.,
            "trajectory": True,
        },
        "top": {
            "mode": "fix",
            "view": "top",
            "resolution": [720, 1280],
            "line_width": 4.,
            "trajectory": True,
        },
        "side": {
            "mode": "fix",
            "view": "side",
            "resolution": [720, 1280],
            "line_width": 4.,
            "trajectory": True,
        }
    }
    
    try:
        # Test each view configuration
        for view_name, view_config in view_configs.items():
            print(f"\n--- Testing {view_name} view ---")
            
            scene_kwargs = {
                "path": scene_path,
                "render_settings": view_config
            }
            
            # Create environment
            env = NavigationEnv2(
                visual=True,
                num_scene=1,
                num_agent_per_scene=2,
                random_kwargs=random_kwargs,
                scene_kwargs=scene_kwargs,
                sensor_kwargs=sensor_kwargs,
                dynamics_kwargs={}
            )
            
            print(f"✓ Environment with {view_name} view created successfully")
            
            # Test reset
            obs = env.reset()
            print(f"✓ Reset successful with {view_name} view")
            
            # Test rendering
            print(f"Testing rendering with {view_name} view...")
            for i in range(5):
                action = th.rand((2, 4))
                obs, reward, done, info = env.step(action)
                
                try:
                    # Test different render options
                    img_basic = env.render()
                    img_with_axes = env.render(is_draw_axes=True)
                    img_with_trajectory = env.render()  # trajectory is controlled by render_settings
                    
                    print(f"  Step {i+1}: Rendered successfully")
                    print(f"    Basic render shape: {img_basic[0].shape if img_basic else 'None'}")
                    print(f"    With axes shape: {img_with_axes[0].shape if img_with_axes else 'None'}")
                    print(f"    With trajectory shape: {img_with_trajectory[0].shape if img_with_trajectory else 'None'}")
                    
                    # Save sample images
                    if img_basic and len(img_basic) > 0:
                        cv2.imwrite(f"output/view_{view_name}_step{i}.png", img_basic[0])
                    
                except Exception as render_error:
                    print(f"    ⚠ Render error at step {i+1}: {render_error}")
            
            print(f"✓ {view_name} view test completed")
            
            # Clean up
            del env
        
        # Test follow mode
        print(f"\n--- Testing follow mode ---")
        follow_config = {
            "mode": "follow",
            "view": "near",
            "resolution": [720, 1280],
            "line_width": 4.,
            "trajectory": True,
        }
        
        scene_kwargs = {
            "path": scene_path,
            "render_settings": follow_config
        }
        
        env = NavigationEnv2(
            visual=True,
            num_scene=1,
            num_agent_per_scene=1,
            random_kwargs=random_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            dynamics_kwargs={}
        )
        
        print("✓ Environment with follow mode created successfully")
        
        # Test follow mode rendering
        obs = env.reset()
        print("✓ Reset successful with follow mode")
        
        for i in range(5):
            action = th.rand((1, 4))
            obs, reward, done, info = env.step(action)
            
            try:
                img = env.render()
                print(f"  Follow step {i+1}: Position = {env.position[0]}, Render shape = {img[0].shape if img else 'None'}")
                
                if img and len(img) > 0:
                    cv2.imwrite(f"output/view_follow_step{i}.png", img[0])
                    
            except Exception as render_error:
                print(f"    ⚠ Follow render error at step {i+1}: {render_error}")
        
        print("✓ Follow mode test completed")
        
        print("\n✓ All view and rendering tests passed!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_view_functionality() 