import argparse
import math
import os
import sys

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.SceneManager import (  # noqa: E402
    ROTOR_SPIN_DIRECTIONS,
    SceneManager,
)


def _camera_state(eye, target):
    camera_pose = mn.Matrix4.look_at(
        mn.Vector3(eye),
        mn.Vector3(target),
        mn.Vector3.y_axis(),
    )
    return habitat_sim.AgentState(
        position=camera_pose.translation,
        rotation=R.from_matrix(np.array(camera_pose)[:3, :3]).as_quat(),
    )


def _draw_body_axes(line_render, drone, axis_len):
    line_render.push_transform(drone.root_scene_node.transformation)
    line_render.draw_transformed_line(
        mn.Vector3([0, 0, 0]), mn.Vector3([axis_len, 0, 0]), mn.Color4(1, 0, 0, 1)
    )
    line_render.draw_transformed_line(
        mn.Vector3([0, 0, 0]), mn.Vector3([0, axis_len, 0]), mn.Color4(0, 1, 0, 1)
    )
    line_render.draw_transformed_line(
        mn.Vector3([0, 0, 0]), mn.Vector3([0, 0, axis_len]), mn.Color4(0, 0, 1, 1)
    )
    line_render.pop_transform()


def render_video(args):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.abspath(args.scene)
    sim_cfg.scene_dataset_config_file = os.path.abspath(args.dataset)
    sim_cfg.enable_physics = False
    sim_cfg.create_renderer = True

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [args.height, args.width]
    sensor_spec.position = mn.Vector3([0, 0, 0])

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg=sim_cfg, agents=[agent_cfg])
    )
    try:
        obj_mgr = sim.get_rigid_object_manager()
        drone = obj_mgr.add_object_by_template_handle(os.path.abspath(args.drone))
        drone.root_scene_node.translation = mn.Vector3([0.0, 0.0, 0.0])
        drone.root_scene_node.rotation = mn.Quaternion.rotation(
            mn.Rad(math.radians(args.yaw_deg)), mn.Vector3.y_axis()
        )

        rotors = SceneManager._find_mavic_rotor_nodes(drone)
        if any(node is None for node in rotors):
            raise RuntimeError(
                "Could not find all Mavic rotor nodes. Use a DJI_Mavic_* asset "
                "or add rotor metadata for this mesh."
            )
        base_rotations = [node.rotation for node in rotors]
        spin_dirs = np.asarray(ROTOR_SPIN_DIRECTIONS, dtype=np.float32)
        phase = np.zeros(4, dtype=np.float32)
        omega = np.full(4, args.rotor_rps * 2 * math.pi, dtype=np.float32)

        sim.get_agent(0).set_state(_camera_state(args.eye, args.target))

        line_render = sim.get_debug_line_render()
        line_render.set_line_width(args.axis_width)

        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {args.output}")

        for _ in range(int(args.duration * args.fps)):
            phase = (phase + omega * spin_dirs / args.fps) % (2 * np.pi)
            for rotor_id, rotor in enumerate(rotors):
                rotor.rotation = base_rotations[rotor_id] * mn.Quaternion.rotation(
                    mn.Rad(float(phase[rotor_id])), mn.Vector3.y_axis()
                )

            if args.axes:
                _draw_body_axes(line_render, drone, args.axis_len)

            rgba = sim.get_sensor_observations(0)["color"]
            rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        writer.release()
    finally:
        sim.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a debug video of a drone with spinning rotors and body axes."
    )
    parser.add_argument(
        "--output",
        default="renders/debug_rotor_axes.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--drone",
        default="datasets/visfly-beta/configs/agents/DJI_Mavic_red.object_config.json",
        help="Drone object config path.",
    )
    parser.add_argument(
        "--scene",
        default="datasets/visfly-beta/configs/scenes/empty_stage.scene_instance.json",
        help="Scene instance JSON path.",
    )
    parser.add_argument(
        "--dataset",
        default="datasets/visfly-beta/visfly-beta.scene_dataset_config.json",
        help="Habitat scene dataset config path.",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rotor-rps", type=float, default=1.5)
    parser.add_argument("--yaw-deg", type=float, default=20.0)
    parser.add_argument("--eye", nargs=3, type=float, default=[0.34, 0.24, 0.42])
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.02, 0.0])
    parser.add_argument("--axis-len", type=float, default=0.18)
    parser.add_argument("--axis-width", type=float, default=3.0)
    parser.add_argument("--axes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    render_video(args)
    print(os.path.abspath(args.output))
