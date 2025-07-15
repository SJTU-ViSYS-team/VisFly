import os
import argparse
import habitat_sim
import numpy as np
from PIL import Image


def make_cfg(dataset_root: str, scene_json: str, width: int = 640, height: int = 480):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.join(dataset_root, "spy_datasets.scene_dataset_config.json")
    sim_cfg.scene_id = scene_json  # absolute or relative to dataset root
    sim_cfg.enable_physics = False

    sensor_specs = []
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [height, width]
    color_sensor_spec.position = [0.0, 1.6, 0.0]  # eye height
    sensor_specs.append(color_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def render_scene(dataset_root: str, scene_json: str, out_path: str):
    cfg = make_cfg(dataset_root, scene_json)
    sim = habitat_sim.Simulator(cfg)
    # default agent at origin – adjust if needed
    obs = sim.get_sensor_observations()
    color = obs["color"]  # HWC RGBA
    Image.fromarray(color[:, :, :3]).save(out_path)
    sim.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="VisFly/datasets/spy_datasets", help="Root of spy_datasets")
    parser.add_argument("--scenes", nargs="+", required=True, help="Scene instance json files to render")
    parser.add_argument("--out_dir", default="renders", help="Directory to save images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for scene in args.scenes:
        name = os.path.splitext(os.path.basename(scene))[0]
        out_file = os.path.join(args.out_dir, f"{name}.png")
        render_scene(args.dataset_root, scene, out_file)


if __name__ == "__main__":
    main() 