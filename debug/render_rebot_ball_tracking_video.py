import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import habitat_sim
import magnum as mn
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import (  # noqa: E402
    ReBotJointDynamics,
    ReBotKinematics,
    ReBotIKParams,
    ReBotRigidBodyDynamics,
    solve_position_ik,
)


DEFAULT_URDF = "datasets/visfly-beta/urdf/rebot_devarm/reBot-DevArm_fixend.urdf"
DEFAULT_SCENE = "datasets/visfly-beta/self_define_stages/box30_wall.glb"
DEFAULT_BALL = "datasets/visfly-beta/self_define_objects/ball_R0.3.glb"


def _camera_state(eye, target, up):
    camera_pose = mn.Matrix4.look_at(mn.Vector3(eye), mn.Vector3(target), mn.Vector3(up))
    return habitat_sim.AgentState(
        position=camera_pose.translation,
        rotation=R.from_matrix(np.array(camera_pose)[:3, :3]).as_quat(),
    )


def _make_ball_config(ball_asset):
    fd, path = tempfile.mkstemp(suffix=".object_config.json", prefix="rebot_tracking_ball_")
    os.close(fd)
    cfg = {
        "render_asset": os.path.abspath(ball_asset),
        "collision_asset": os.path.abspath(ball_asset),
        "mass": 0.1,
        "COM": [0, 0, 0],
        "join_collision_meshes": True,
        "scale": [0.12, 0.12, 0.12],
        "semantic_id": 55,
    }
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _ball_position(t):
    # A reachable smooth target path in the reBot base frame.
    return th.tensor(
        [
            0.24 + 0.10 * math.sin(2.0 * math.pi * 0.17 * t),
            0.03 + 0.12 * math.sin(2.0 * math.pi * 0.23 * t + 0.6),
            0.26 + 0.07 * math.sin(2.0 * math.pi * 0.19 * t + 1.1),
        ],
        dtype=th.float64,
    )


def _draw_link_axes(line_render, articulated_object, axis_len):
    for link_id in articulated_object.get_link_ids():
        if articulated_object.get_link_num_dofs(link_id) <= 0:
            continue
        node = articulated_object.get_link_scene_node(link_id)
        line_render.push_transform(node.absolute_transformation())
        line_render.draw_transformed_line(mn.Vector3([0, 0, 0]), mn.Vector3([axis_len, 0, 0]), mn.Color4(1, 0, 0, 1))
        line_render.draw_transformed_line(mn.Vector3([0, 0, 0]), mn.Vector3([0, axis_len, 0]), mn.Color4(0, 1, 0, 1))
        line_render.draw_transformed_line(mn.Vector3([0, 0, 0]), mn.Vector3([0, 0, axis_len]), mn.Color4(0, 0.25, 1, 1))
        line_render.pop_transform()


def _overlay(frame, lines):
    cv2.rectangle(frame, (10, 10), (900, 120), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 30 + 26 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _drive_joint_target(joint_dyn, rigid_dyn, q_target, q_current, args):
    if args.control_mode == "pos_vel":
        return joint_dyn.pos_vel(q_target)
    if args.control_mode == "mit":
        return joint_dyn.mit(q_target)
    if args.control_mode == "vel":
        dq_cmd = (q_target - q_current) * args.fps
        return joint_dyn.set_vel(dq_cmd)
    if args.control_mode == "gravity_compensation":
        tau_g = rigid_dyn.gravity_vector(q_current)
        return joint_dyn.gravity_compensation(tau_g)
    raise RuntimeError(f"Unsupported control mode: {args.control_mode}")


def render_video(args):
    kin = ReBotKinematics(args.urdf)
    motor_mode = "mit" if args.control_mode == "gravity_compensation" else args.control_mode
    joint_dyn = ReBotJointDynamics(kinematics=kin, mode=motor_mode, dt=1.0 / args.fps)
    rigid_dyn = ReBotRigidBodyDynamics(kinematics=kin)
    ik_params = ReBotIKParams(
        max_iter=args.ik_max_iter,
        tolerance=args.ik_tolerance,
        step_size=args.ik_step_size,
        damping=args.ik_damping,
        backtracking_steps=args.ik_backtracking_steps,
    )
    q = th.tensor(args.initial_joints, dtype=th.float64)
    joint_dyn.reset(q=q)

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
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg=sim_cfg, agents=[agent_cfg]))

    ball_config = _make_ball_config(args.ball)
    try:
        rebot = sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            os.path.abspath(args.urdf),
            fixed_base=True,
            global_scale=1.0,
            maintain_link_order=True,
            intertia_from_urdf=True,
        )
        rebot.translation = mn.Vector3([0, 0, 0])
        rebot.joint_positions = joint_dyn.state.q.detach().cpu().numpy()

        obj_mgr = sim.get_object_template_manager()
        ids = obj_mgr.load_configs(ball_config)
        ball_handle = obj_mgr.get_template_handle_by_id(ids[0])
        ball = sim.get_rigid_object_manager().add_object_by_template_handle(ball_handle)
        ball.motion_type = habitat_sim.physics.MotionType.KINEMATIC

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

        tracking_error = th.zeros(3, dtype=th.float64)
        for frame_id in range(int(args.duration * args.fps)):
            t = frame_id / args.fps
            ball_pos = _ball_position(t)
            ball.translation = mn.Vector3(ball_pos.detach().cpu().numpy().tolist())

            q = joint_dyn.state.q
            ik_result = solve_position_ik(kin, ball_pos, q, ik_params)
            q_target = ik_result.q
            tracking_error = ik_result.error
            state = _drive_joint_target(joint_dyn, rigid_dyn, q_target, q, args)
            rebot.joint_positions = state.q.detach().cpu().numpy()

            if args.axes:
                _draw_link_axes(line_render, rebot, args.axis_len)

            rgba = sim.get_sensor_observations(0)["color"]
            frame = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
            ee = kin.end_effector_transform(state.q)[:3, 3]
            _overlay(
                frame,
                [
                    _control_label(args.control_mode),
                    "ball xyz = " + ", ".join(f"{v:.3f}" for v in ball_pos.tolist()),
                    "ee xyz   = " + ", ".join(f"{v:.3f}" for v in ee.detach().tolist()),
                    f"tracking error norm = {tracking_error.item():.4f} m",
                ],
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(output)
    finally:
        sim.close()
        if os.path.exists(ball_config):
            os.remove(ball_config)


def _control_label(control_mode):
    if control_mode == "gravity_compensation":
        return "gravity_compensation: official MIT hold-current-q + tau=g(q)"
    return f"{control_mode}: end-effector tracks moving ball"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a reBot position-control video tracking a moving ball."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--ball", default=DEFAULT_BALL)
    parser.add_argument("--output", default="renders/rebot_ball_tracking.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--control-mode",
        choices=["pos_vel", "mit", "vel", "gravity_compensation"],
        default="pos_vel",
    )
    parser.add_argument("--ik-max-iter", type=int, default=20)
    parser.add_argument("--ik-tolerance", type=float, default=1e-4)
    parser.add_argument("--ik-step-size", type=float, default=0.5)
    parser.add_argument("--ik-damping", type=float, default=1e-6)
    parser.add_argument("--ik-backtracking-steps", type=int, default=4)
    parser.add_argument("--axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-len", type=float, default=0.06)
    parser.add_argument("--axis-width", type=float, default=3.0)
    parser.add_argument("--eye", type=float, nargs=3, default=[0.65, -1.05, 0.58])
    parser.add_argument("--target", type=float, nargs=3, default=[0.05, 0.0, 0.25])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument(
        "--initial-joints",
        type=float,
        nargs=6,
        default=[0.0, -0.9, -0.8, 0.4, -0.2, 0.0],
    )
    return parser.parse_args()


if __name__ == "__main__":
    render_video(parse_args())
