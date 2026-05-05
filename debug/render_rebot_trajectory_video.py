import argparse
import os
import sys
from pathlib import Path

import cv2
import habitat_sim
import magnum as mn
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.getcwd()))

from VisFly.utils.rebot import (  # noqa: E402
    ReBotCLIKParams,
    ReBotIKParams,
    ReBotJointDynamics,
    ReBotKinematics,
    TrajPlanParams,
    TrajProfile,
    compute_traj_stats,
    make_transform,
    plan_cartesian_geodesic_trajectory,
    plan_joint_space_trajectory,
    solve_pose_ik,
)


DEFAULT_URDF = "datasets/visfly-beta/urdf/rebot_devarm/reBot-DevArm_fixend.urdf"
DEFAULT_SCENE = "datasets/visfly-beta/self_define_stages/box30_wall.glb"


def _camera_state(eye, target, up):
    camera_pose = mn.Matrix4.look_at(mn.Vector3(eye), mn.Vector3(target), mn.Vector3(up))
    return habitat_sim.AgentState(
        position=camera_pose.translation,
        rotation=R.from_matrix(np.array(camera_pose)[:3, :3]).as_quat(),
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


def _draw_pose_axes(line_render, T, axis_len):
    T_np = T.detach().cpu().numpy()
    origin = T_np[:3, 3]
    axes = T_np[:3, :3]
    colors = [mn.Color4(1, 0, 0, 1), mn.Color4(0, 1, 0, 1), mn.Color4(0, 0.25, 1, 1)]
    for i, color in enumerate(colors):
        end = origin + axes[:, i] * axis_len
        line_render.draw_transformed_line(mn.Vector3(origin.tolist()), mn.Vector3(end.tolist()), color)


def _draw_polyline(line_render, points, color):
    if len(points) < 2:
        return
    for p0, p1 in zip(points[:-1], points[1:]):
        line_render.draw_transformed_line(mn.Vector3(p0), mn.Vector3(p1), color)


def _overlay(frame, lines):
    cv2.rectangle(frame, (10, 10), (1010, 150), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 32 + 26 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _profile(value):
    return TrajProfile(value)


def render_video(args):
    kin = ReBotKinematics(args.urdf)
    q_start = th.tensor(args.initial_joints, dtype=th.float64)
    target_pose = make_transform(
        th.tensor(args.target_xyz, dtype=th.float64),
        rpy=th.tensor(args.target_rpy, dtype=th.float64),
    )

    ik_result = solve_pose_ik(
        kin,
        target_pose,
        q_start,
        ReBotIKParams(
            max_iter=args.ik_max_iter,
            tolerance=args.ik_tolerance,
            step_size=args.ik_step_size,
            damping=args.ik_damping,
            backtracking_steps=args.ik_backtracking_steps,
        ),
    )
    if not ik_result.success and not args.allow_ik_failure:
        raise RuntimeError(f"Target pose IK failed: error={ik_result.error.item():.3e}")

    traj_params = TrajPlanParams(
        dt=args.traj_dt,
        profile=_profile(args.profile),
        accel_ratio=args.accel_ratio,
    )
    clik_params = ReBotCLIKParams(
        max_iter=args.clik_max_iter,
        tolerance=args.clik_tolerance,
        damping=args.clik_damping,
        step_size=args.clik_step_size,
    )
    start_pose = kin.end_effector_transform(q_start)
    cart_result = plan_cartesian_geodesic_trajectory(start_pose, target_pose, args.duration, traj_params)
    joint_traj = plan_joint_space_trajectory(
        kin,
        q_start,
        ik_result.q,
        args.duration,
        params=traj_params,
        ik_params=clik_params,
        null_gain=args.null_gain,
        start_pose=start_pose,
        end_pose=target_pose,
    )
    stats = compute_traj_stats(kin, joint_traj, start_pose, target_pose, args.duration, traj_params)
    if not joint_traj:
        raise RuntimeError("Generated empty reBot trajectory")

    joint_dyn = ReBotJointDynamics(kinematics=kin, mode="pos_vel", dt=1.0 / args.fps)
    joint_dyn.reset(q=q_start)

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

        sim.get_agent(0).set_state(_camera_state(args.eye, args.camera_target, args.up))
        line_render = sim.get_debug_line_render()
        line_render.set_line_width(args.line_width)

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

        ref_points = [pt.pose[:3, 3].detach().cpu().numpy().tolist() for pt in cart_result.trajectory.points()]
        visited_points = []
        num_frames = int(args.duration * args.fps)
        for frame_id in range(num_frames):
            progress = min(1.0, frame_id / max(1, num_frames - 1))
            traj_index = min(len(joint_traj) - 1, int(round(progress * (len(joint_traj) - 1))))
            q_ref = joint_traj[traj_index].q
            state = joint_dyn.pos_vel(q_ref)
            rebot.joint_positions = state.q.detach().cpu().numpy()

            ee_pose = kin.end_effector_transform(state.q)
            visited_points.append(ee_pose[:3, 3].detach().cpu().numpy().tolist())

            _draw_polyline(line_render, ref_points, mn.Color4(1.0, 0.75, 0.05, 1))
            _draw_polyline(line_render, visited_points, mn.Color4(0.0, 0.75, 1.0, 1))
            _draw_pose_axes(line_render, target_pose, args.target_axis_len)
            if args.axes:
                _draw_link_axes(line_render, rebot, args.axis_len)

            rgba = sim.get_sensor_observations(0)["color"]
            frame = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
            pose_err = (ee_pose - target_pose).abs().max().item()
            _overlay(
                frame,
                [
                    "official trajectory: SE(3) geodesic -> CLIK -> POS_VEL playback",
                    f"profile={args.profile}  points={len(joint_traj)}  success={stats.success_rate:.2f}",
                    f"IK error={ik_result.error.item():.3e}  traj avg={stats.avg_ik_error:.3e}  max={stats.max_ik_error:.3e}",
                    "target xyz = " + ", ".join(f"{v:.3f}" for v in args.target_xyz),
                    f"current-to-target transform max abs = {pose_err:.3e}",
                ],
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        print(output)
    finally:
        sim.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render official-style reBot SE(3) geodesic trajectory + CLIK playback."
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--output", default="renders/rebot_trajectory_official.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--traj-dt", type=float, default=0.02)
    parser.add_argument("--profile", choices=[p.value for p in TrajProfile], default=TrajProfile.MIN_JERK.value)
    parser.add_argument("--accel-ratio", type=float, default=0.25)
    parser.add_argument("--null-gain", type=float, default=0.1)
    parser.add_argument("--ik-max-iter", type=int, default=300)
    parser.add_argument("--ik-tolerance", type=float, default=1e-4)
    parser.add_argument("--ik-step-size", type=float, default=0.5)
    parser.add_argument("--ik-damping", type=float, default=1e-6)
    parser.add_argument("--ik-backtracking-steps", type=int, default=4)
    parser.add_argument("--clik-max-iter", type=int, default=200)
    parser.add_argument("--clik-tolerance", type=float, default=1e-4)
    parser.add_argument("--clik-step-size", type=float, default=0.8)
    parser.add_argument("--clik-damping", type=float, default=1e-6)
    parser.add_argument("--allow-ik-failure", action="store_true")
    parser.add_argument("--axes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--axis-len", type=float, default=0.055)
    parser.add_argument("--target-axis-len", type=float, default=0.09)
    parser.add_argument("--line-width", type=float, default=3.0)
    parser.add_argument("--eye", type=float, nargs=3, default=[0.65, -1.05, 0.58])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.05, 0.0, 0.25])
    parser.add_argument("--up", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--target-xyz", type=float, nargs=3, default=[0.28, 0.08, 0.30])
    parser.add_argument("--target-rpy", type=float, nargs=3, default=[0.0, 0.35, 0.2])
    parser.add_argument(
        "--initial-joints",
        type=float,
        nargs=6,
        default=[0.0, -0.9, -0.8, 0.4, -0.2, 0.0],
    )
    return parser.parse_args()


if __name__ == "__main__":
    render_video(parse_args())
