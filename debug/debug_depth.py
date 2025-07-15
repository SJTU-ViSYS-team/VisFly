#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import time
import psutil
import gc
import cv2

# Add the correct path to import VisFly modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.utils.launcher import rl_parser, training_params
from habitat_sim.sensor import SensorType

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def test_depth_sensor_config():
    """Test different depth sensor configurations and monitor memory usage"""
    
    print("=== Depth Sensor Memory Debug Test ===")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Use a simpler scene path that should exist
    scene_path = "../datasets/spy_datasets/configs/garage_simple_l_medium"
    
    # Test configuration 1: Current setup (64x64 resolution)
    print("\n--- Test 1: Current 64x64 depth sensor ---")
    sensor_kwargs_64 = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }]
    
    try:
        env_64 = NavigationEnv(
            num_agent_per_scene=96,  # Same as training
            num_scene=1,
            visual=True,
            sensor_kwargs=sensor_kwargs_64,
            scene_kwargs={"path": scene_path},
            max_episode_steps=256,
        )
        
        print(f"Memory after 64x64 env creation: {get_memory_usage():.2f} GB")
        
        # Test observation
        obs = env_64.reset()
        print(f"Memory after reset: {get_memory_usage():.2f} GB")
        
        # Check observation structure
        print(f"Observation keys: {list(obs.keys())}")
        if 'depth' in obs:
            depth_shape = obs['depth'].shape
            print(f"Depth shape: {depth_shape}")
            print(f"Depth dtype: {obs['depth'].dtype}")
            print(f"Depth range: [{obs['depth'].min():.3f}, {obs['depth'].max():.3f}]")
            
            # Calculate memory usage for depth observations
            depth_memory_mb = np.prod(depth_shape) * 4 / 1024 / 1024  # 4 bytes per float32
            print(f"Depth observation memory: {depth_memory_mb:.2f} MB")
        
        # Test a few steps
        for i in range(5):
            action = np.random.uniform(-1, 1, (96, 4)).astype(np.float32)  # Ensure float32
            obs, reward, done, info = env_64.step(action)
            if i == 0:
                print(f"Memory after first step: {get_memory_usage():.2f} GB")
            # Save depth images for the first step
            if i < 3 and 'depth' in obs:
                depth = obs['depth']
                # Save the first 3 agents' depth images
                for j in range(min(3, depth.shape[0])):
                    depth_img = depth[j, 0].cpu().numpy() if hasattr(depth, 'cpu') else depth[j, 0]
                    # Normalize to 0-255 for visualization
                    norm_depth = 255 * (depth_img - depth_img.min()) / (depth_img.ptp() + 1e-8)
                    norm_depth = norm_depth.astype(np.uint8)
                    cv2.imwrite(f"depth_64x64_step{i}_agent{j}.png", norm_depth)
        
        print(f"Memory after 5 steps: {get_memory_usage():.2f} GB")
        
        # Clean up
        del env_64
        gc.collect()
        print(f"Memory after cleanup: {get_memory_usage():.2f} GB")
        
    except Exception as e:
        print(f"Error with 64x64 config: {e}")
    
    # Test configuration 2: Higher resolution (128x128)
    print("\n--- Test 2: 128x128 depth sensor ---")
    sensor_kwargs_128 = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [128, 128],
    }]
    
    try:
        env_128 = NavigationEnv(
            num_agent_per_scene=96,
            num_scene=1,
            visual=True,
            sensor_kwargs=sensor_kwargs_128,
            scene_kwargs={"path": scene_path},
            max_episode_steps=256,
        )
        
        print(f"Memory after 128x128 env creation: {get_memory_usage():.2f} GB")
        
        obs = env_128.reset()
        if 'depth' in obs:
            depth_shape = obs['depth'].shape
            depth_memory_mb = np.prod(depth_shape) * 4 / 1024 / 1024
            print(f"128x128 depth shape: {depth_shape}")
            print(f"128x128 depth memory: {depth_memory_mb:.2f} MB")
            # Save PNG for first agent
            depth = obs['depth']
            depth_img = depth[0, 0].cpu().numpy() if hasattr(depth, 'cpu') else depth[0, 0]
            norm_depth = 255 * (depth_img - depth_img.min()) / (depth_img.ptp() + 1e-8)
            norm_depth = norm_depth.astype(np.uint8)
            cv2.imwrite("depth_128x128_reset_agent0.png", norm_depth)
        
        del env_128
        gc.collect()
        print(f"Memory after 128x128 cleanup: {get_memory_usage():.2f} GB")
        
    except Exception as e:
        print(f"Error with 128x128 config: {e}")
    
    # Test configuration 3: Lower resolution (32x32)
    print("\n--- Test 3: 32x32 depth sensor ---")
    sensor_kwargs_32 = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [32, 32],
    }]
    
    try:
        env_32 = NavigationEnv(
            num_agent_per_scene=96,
            num_scene=1,
            visual=True,
            sensor_kwargs=sensor_kwargs_32,
            scene_kwargs={"path": scene_path},
            max_episode_steps=256,
        )
        
        print(f"Memory after 32x32 env creation: {get_memory_usage():.2f} GB")
        
        obs = env_32.reset()
        if 'depth' in obs:
            depth_shape = obs['depth'].shape
            depth_memory_mb = np.prod(depth_shape) * 4 / 1024 / 1024
            print(f"32x32 depth shape: {depth_shape}")
            print(f"32x32 depth memory: {depth_memory_mb:.2f} MB")
            # Save PNG for first agent
            depth = obs['depth']
            depth_img = depth[0, 0].cpu().numpy() if hasattr(depth, 'cpu') else depth[0, 0]
            norm_depth = 255 * (depth_img - depth_img.min()) / (depth_img.ptp() + 1e-8)
            norm_depth = norm_depth.astype(np.uint8)
            cv2.imwrite("depth_32x32_reset_agent0.png", norm_depth)
        
        del env_32
        gc.collect()
        print(f"Memory after 32x32 cleanup: {get_memory_usage():.2f} GB")
        
    except Exception as e:
        print(f"Error with 32x32 config: {e}")

def test_sensor_loading():
    """Test if depth sensor is correctly loaded and accessible"""
    print("\n=== Testing Depth Sensor Loading ===")
    
    sensor_kwargs = [{
        "sensor_type": SensorType.DEPTH,
        "uuid": "depth",
        "resolution": [64, 64],
    }]
    
    try:
        env = NavigationEnv(
            num_agent_per_scene=1,  # Single agent for testing
            num_scene=1,
            visual=True,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs={"path": "../datasets/spy_datasets/configs/garage_simple_l_medium"},
        )
        
        print("✓ Environment created successfully")
        
        # Check sensor settings
        print(f"✓ Sensor settings: {env.envs.sceneManager.sensor_settings}")
        
        # Check observation space
        print(f"✓ Observation space keys: {list(env.observation_space.keys())}")
        if 'depth' in env.observation_space:
            print(f"✓ Depth observation space: {env.observation_space['depth']}")
        
        # Test reset and observation
        obs = env.reset()
        print(f"✓ Reset successful, observation keys: {list(obs.keys())}")
        
        if 'depth' in obs:
            depth = obs['depth']
            print(f"✓ Depth observation shape: {depth.shape}")
            print(f"✓ Depth observation dtype: {depth.dtype}")
            print(f"✓ Depth observation range: [{depth.min():.3f}, {depth.max():.3f}]")
            
            # Check for NaN or inf values
            if np.isnan(depth).any():
                print("⚠ WARNING: NaN values found in depth observation!")
            if np.isinf(depth).any():
                print("⚠ WARNING: Inf values found in depth observation!")
            
            # Test a few steps
            for i in range(3):
                action = np.random.uniform(-1, 1, (1, 4)).astype(np.float32)
                obs, reward, done, info = env.step(action)
                if 'depth' in obs:
                    depth = obs['depth']
                    depth_img = depth[0, 0].cpu().numpy() if hasattr(depth, 'cpu') else depth[0, 0]
                    norm_depth = 255 * (depth_img - depth_img.min()) / (depth_img.ptp() + 1e-8)
                    norm_depth = norm_depth.astype(np.uint8)
                    cv2.imwrite(f"depth_single_step{i}.png", norm_depth)
                    print(f"✓ Step {i+1}: depth shape {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_depth_sensor_config()
    test_sensor_loading() 