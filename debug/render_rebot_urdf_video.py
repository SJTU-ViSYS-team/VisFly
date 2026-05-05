import argparse
import math
import os
from pathlib import Path

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_URDF = "datasets/visfly-beta/urdf/rebot_devarm/reBot-DevArm_fixend.urdf"
DEFAULT_SCENE = "datasets/visfly-beta/self_define_stages/box30_wall.glb"


def _camera_state(eye, target, up):
    camera_pose = mn.Matrix4.look_at(
        mn.Vector3(eye),
        mn.Vector3(target),
        mn.Vector3(up),
    )
    return habitat_sim.AgentState(
        position=camera_pose.translation,
        rotation=R.from_matrix(np.array(camera_pose)[:3, :3]).as_quat(),
    )


def _draw_link_axes(line_render, articulated_object, axis_len):
    joint_names = []
    for link_id in articulated_object.get_link_ids():
        if articulated_object.get_link_num_dofs(link_id) <= 0:
            continue

        joint_names.append(articulated_object.get_link_joint_name(link_id))
        node = articulated_object.get_link_scene_node(link_id)
        line_render.push_transform(node.absolute_transformation())
        line_render.draw_transformed_line(
            mn.Vector3([0, 0, 0]),
            mn.Vector3([axis_len, 0, 0]),
            mn.Color4(1, 0, 0, 1),
        )
        line_render.draw_transformed_line(
            mn.Vector3([0, 0, 0]),
            mn.Vector3([0, axis_len, 0]),
            mn.Color4(0, 1, 0, 1),
        )
        line_render.draw_transformed_line(
            mn.Vector3([0, 0, 0]),
            mn.Vector3([0, 0, axis_len]),
            mn.Color4(0, 0.25, 1, 1),
        )
        line_render.pop_transform()
    return joint_names


def _has_gripper_dof(joint_names):
    return any("gripper" in name.lower() or "finger" in name.lower() for name in joint_names)


def _joint_sweep(joint_limits, t, sweep_fraction):
    lower, upper = (np.asarray(values, dtype=np.float32) for values in joint_limits)
    center = 0.5 * (lower + upper)
    amplitude = 0.5 * (upper - lower) * sweep_fraction
    phases = np.linspace(0.0, math.pi, len(center), dtype=np.float32)
    rates = np.asarray([0.45, 0.35, 0.40, 0.70, 0.85, 1.05], dtype=np.float32)
    q = center + amplitude * np.sin(2.0 * math.pi * rates * t + phases)
    return np.clip(q, lower, upper)


def _overlay_text(frame, lines):
    for row, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, 28 + 26 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def render_video(args):
    if not os.path.exists(args.urdf):
        raise FileNotFoundError(
            f"reBot URDF not found: {args.urdf}. "
            "Install/copy the reBot asset package under datasets/visfly-beta/urdf/rebot_devarm first."
        )

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.abspath(args.scene)
    sim_cfg.enable_physics = True
    sim_cfg.create_renderer = True

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [args.height, args.width]
    sensor_spec.hfov = args.hfov
    sensor_spec.position = mn.Vector3([0, 0, 0])

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]

    sim = habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg=sim_cfg, agents=[agent_cfg])
    )
    try:
        rebot = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            os.path.abspath(args.urdf),
            fixed_base=True,
            global_scale=args.global_scale,
            maintain_link_order=True,
            intertia_from_urdf=True,
        )
        rebot.translation = mn.Vector3(args.position)

        sim.get_agent(0).set_state(_camera_state(args.eye, args.target, args.up))

        line_render = sim.get_debug_line_render()
        line_render.set_line_width(args.axis_width)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output}")

        joint_names = None
        num_frames = int(args.duration * args.fps)
        for frame_id in range(num_frames):
            t = frame_id / args.fps
            rebot.joint_positions = _joint_sweep(
                rebot.joint_position_limits, t, args.sweep_fraction
            )
            rebot.clamp_joint_limits()

            if args.axes:
                joint_names = _draw_link_axes(line_render, rebot, args.axis_len)

            rgba = sim.get_sensor_observations(0)["color"]
            frame = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
            cv2.rectangle(frame, (10, 10), (920, 122), (0, 0, 0), -1)
            _overlay_text(
                frame,
                [
                    "reBot URDF joint sweep inside URDF limits",
                    "q rad = "
                    + ", ".join(f"{q:.2f}" for q in rebot.joint_positions),
                    "DoF axes: X red, Y green, Z blue; "
                    + ", ".join(joint_names or []),
                    "Gripper: fixed in upstream URDF"
                    if joint_names and not _has_gripper_dof(joint_names)
                    else "",
                ],
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(output)
    finally:
        sim.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a video of the integrated reBot DevArm URDF sweeping its DoFs within URDF limits."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output", default="renders/rebot_urdf_joint_sweep.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--sweep-fraction", type=float, default=0.45)
    parser.add_argument("--axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-len", type=float, default=0.075)
    parser.add_argument("--axis-width", type=float, default=4.0)
    parser.add_argument(
        "--position", type=float, nargs=3, default=[0.0, 0.0, 0.0]
    )
    parser.add_argument("--eye", type=float, nargs=3, default=[0.55, -0.95, 0.55])
    parser.add_argument("--target", type=float, nargs=3, default=[0.02, 0.0, 0.23])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    return parser.parse_args()


if __name__ == "__main__":
    render_video(parse_args())
