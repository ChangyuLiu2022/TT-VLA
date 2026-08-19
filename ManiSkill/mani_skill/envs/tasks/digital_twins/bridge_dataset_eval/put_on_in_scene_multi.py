import os
from pathlib import Path
import cv2
import numpy as np
import sapien
import torch
import torch.nn.functional as F
from sapien.physx import PhysxMaterial
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import BRIDGE_DATASET_ASSET_PATH, \
    WidowX250SBridgeDatasetFlatTable
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.registration import register_env

CARROT_DATASET_DIR = Path(__file__).parent / ".." / ".." / ".." / ".." / "assets" / "carrot"


def masks_to_boxes_pytorch(masks):
    b, H, W = masks.shape
    boxes = []
    for i in range(b):
        pos = masks[i].nonzero(as_tuple=False)  # [N, 2]
        if pos.shape[0] == 0:
            boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.long, device=masks.device))
        else:
            ymin, xmin = pos.min(dim=0)[0]
            ymax, xmax = pos.max(dim=0)[0]
            boxes.append(torch.stack([xmin, ymin, xmax, ymax]))
    return torch.stack(boxes, dim=0)  # [b, 4]


class PutOnPlateInScene25(BaseEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]

    obj_static_friction = 1.0
    obj_dynamic_friction = 1.0

    rgb_camera_name: str = "3rd_view_camera"
    rgb_overlay_mode: str = "background"  # 'background' or 'object' or 'debug' or combinations of them

    overlay_images_numpy: list[np.ndarray]
    overlay_textures_numpy: list[np.ndarray]
    overlay_mix_numpy: list[float]
    overlay_images: torch.Tensor
    overlay_textures: torch.Tensor
    overlay_mix: torch.Tensor
    model_db_carrot: dict[str, dict]
    model_db_plate: dict[str, dict]
    carrot_names: list[str]
    plate_names: list[str]
    select_carrot_ids: torch.Tensor
    select_plate_ids: torch.Tensor
    select_overlay_ids: torch.Tensor
    select_pos_ids: torch.Tensor
    select_quat_ids: torch.Tensor

    initial_qpos: np.ndarray
    initial_robot_pos: sapien.Pose
    safe_robot_pos: sapien.Pose

    def __init__(self, **kwargs):
        # random pose
        self._generate_init_pose()

        # widowx
        self.initial_qpos = np.array([
            -0.01840777, 0.0398835, 0.22242722,
            -0.00460194, 1.36524296, 0.00153398,
            0.037, 0.037,
        ])
        self.initial_robot_pos = sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
        self.safe_robot_pos = sapien.Pose([0.147, 0.028, 1.870], q=[0, 0, 0, 1])

        # stats
        self.extra_stats = dict()

        super().__init__(
            robot_uids=WidowX250SBridgeDatasetFlatTable,
            **kwargs
        )

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.869532),
                            ]
                        )
                    )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    def _build_actor_helper(self, name: str, path: Path, density: float, scale: float, pose: Pose):
        """helper function to build actors by ID directly and auto configure physical materials"""
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()

        collision_file = str(path / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(path / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(path / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(path / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

        builder.initial_pose = pose
        actor = builder.build(name=name)
        return actor

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
        )

    def _load_scene(self, options: dict):
        # original SIMPLER envs always do this? except for open drawer task
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3
            )

        # load background
        builder = self.scene.create_actor_builder()  # Warning should be dissmissed, for we set the initial pose below -> actor.set_pose
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])

        scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")

        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)
        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        # models
        self.model_bbox_sizes = {}

        # carrot
        self.objs_carrot: dict[str, Actor] = {}

        for idx, name in enumerate(self.model_db_carrot):
            model_path = CARROT_DATASET_DIR / "more_carrot" / name
            density = self.model_db_carrot[name].get("density", 1000)
            scale_list = self.model_db_carrot[name].get("scale", [1.0])
            bbox = self.model_db_carrot[name]["bbox"]

            scale = self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([1.0, 0.3 * idx, 1.0]))
            self.objs_carrot[name] = self._build_actor_helper(name, model_path, density, scale, pose)

            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])  # [3]
            self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)  # [3]

        # plate
        self.objs_plate: dict[str, Actor] = {}

        for idx, name in enumerate(self.model_db_plate):
            model_path = CARROT_DATASET_DIR / "more_plate" / name
            density = self.model_db_plate[name].get("density", 1000)
            scale_list = self.model_db_plate[name].get("scale", [1.0])
            bbox = self.model_db_plate[name]["bbox"]

            scale = self.np_random.choice(scale_list)
            pose = Pose.create_from_pq(torch.tensor([2.0, 0.3 * idx, 1.0]))
            self.objs_plate[name] = self._build_actor_helper(name, model_path, density, scale, pose)

            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])  # [3]
            self.model_bbox_sizes[name] = common.to_tensor(bbox_size * scale, device=self.device)  # [3]

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )
        self.extra_stats = dict()

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _settle(self, t=0.5):
        """run the simulation for some steps to help settle the objects"""
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

    def evaluate(self, success_require_src_completely_on_target=True):
        xy_flag_required_offset = 0.01
        z_flag_required_offset = 0.05
        netforce_flag_required_offset = 0.03

        b = self.num_envs

        # actor
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        carrot_p = torch.stack([a.pose.p[idx] for idx, a in enumerate(carrot_actor)])  # [b, 3]
        carrot_q = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        plate_p = torch.stack([a.pose.p[idx] for idx, a in enumerate(plate_actor)])  # [b, 3]
        plate_q = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]

        # whether moved the correct object
        # source_obj_xy_move_dist = torch.linalg.norm(
        #     self.episode_source_obj_xyz_after_settle[:, :2] - source_obj_pose.p[:, :2],
        #     dim=1,
        # )
        # other_obj_xy_move_dist = []
        # for obj_name in self.objs.keys():
        #     obj = self.objs[obj_name]
        #     obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[obj_name]
        #     if obj.name == self.source_obj_name:
        #         continue
        #     other_obj_xy_move_dist.append(
        #         torch.linalg.norm(
        #             obj_xyz_after_settle[:, :2] - obj.pose.p[:, :2], dim=1
        #         )
        #     )

        # moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
        #     all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        # )
        # moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
        #     [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        # )
        # moved_correct_obj = False
        # moved_wrong_obj = False

        # whether the source object is grasped

        is_src_obj_grasped = torch.zeros((b,), dtype=torch.bool, device=self.device)  # [b]

        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            grasped = self.agent.is_grasping(self.objs_carrot[name])  # [b]
            is_src_obj_grasped = torch.where(is_select, grasped, is_src_obj_grasped)  # [b]

        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
                self.plate_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.carrot_bbox_world / 2

        pos_src = carrot_p
        pos_tgt = plate_p
        offset = pos_src - pos_tgt
        xy_flag = (
                torch.linalg.norm(offset[:, :2], dim=1)
                <= tgt_obj_half_length_bbox.max(dim=1).values + xy_flag_required_offset
        )
        z_flag = (offset[:, 2] > 0) & (
                offset[:, 2] - tgt_obj_half_length_bbox[:, 2] - src_obj_half_length_bbox[:, 2]
                <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag
        # src_on_target = False

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            net_forces = torch.zeros((b,), dtype=torch.float32, device=self.device)  # [b]
            for idx in range(self.num_envs):
                force = self.scene.get_pairwise_contact_forces(
                    self.objs_carrot[select_carrot[idx]],
                    self.objs_plate[select_plate[idx]],
                )[idx]
                force = torch.linalg.norm(force)
                net_forces[idx] = force

            src_on_target = src_on_target & (net_forces > netforce_flag_required_offset)

        success = src_on_target

        # prepare dist
        gripper_p = (self.agent.finger1_link.pose.p + self.agent.finger2_link.pose.p) / 2  # [b, 3]
        gripper_q = (self.agent.finger1_link.pose.q + self.agent.finger2_link.pose.q) / 2  # [b, 4]
        gripper_carrot_dist = torch.linalg.norm(gripper_p - carrot_p, dim=1)  # [b, 3]
        gripper_plate_dist = torch.linalg.norm(gripper_p - plate_p, dim=1)  # [b, 3]
        carrot_plate_dist = torch.linalg.norm(carrot_p - plate_p, dim=1)  # [b, 3]

        # self.episode_stats["moved_correct_obj"] = moved_correct_obj
        # self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        self.episode_stats["consecutive_grasp"] = self.episode_stats["consecutive_grasp"] | consecutive_grasp
        self.episode_stats["gripper_carrot_dist"] = gripper_carrot_dist
        self.episode_stats["gripper_plate_dist"] = gripper_plate_dist
        self.episode_stats["carrot_plate_dist"] = carrot_plate_dist

        self.extra_stats["extra_pos_carrot"] = carrot_p
        self.extra_stats["extra_q_carrot"] = carrot_q
        self.extra_stats["extra_pos_plate"] = plate_p
        self.extra_stats["extra_q_plate"] = plate_q
        self.extra_stats["extra_pos_gripper"] = gripper_p
        self.extra_stats["extra_q_gripper"] = gripper_q

        return dict(**self.episode_stats, success=success)

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True

    def get_language_instruction(self):
        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        instruct = []
        for idx in range(self.num_envs):
            carrot_name = self.model_db_carrot[select_carrot[idx]]["name"]
            plate_name = self.model_db_plate[select_plate[idx]]["name"]
            instruct.append(f"put {carrot_name} on {plate_name}")

        return instruct

    def _after_reconfigure(self, options: dict):
        target_object_actor_ids = [
            x._objs[0].per_scene_id
            for x in self.scene.actors.values()
            if x.name not in ["ground", "goal_site", "", "arena"]
        ]
        self.target_object_actor_ids = torch.tensor(
            target_object_actor_ids, dtype=torch.int16, device=self.device
        )
        # get the robot link ids
        robot_links = self.agent.robot.get_links()
        self.robot_link_ids = torch.tensor(
            [x._objs[0].entity.per_scene_id for x in robot_links],
            dtype=torch.int16,
            device=self.device,
        )

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img, overlay_texture, overlay_mix):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        # mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(actor_seg.device)

        mask = torch.isin(actor_seg, torch.concat([self.robot_link_ids, self.target_object_actor_ids]))
        mask = (~mask).to(torch.float32)  # [b, H, W]
        # m = torch.isin(actor_seg, self.robot_link_ids) # "object" mode

        mask = mask.unsqueeze(-1)  # [b, H, W, 1]
        # mix = overlay_mix.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 1]

        # perform overlay on the RGB observation image
        assert rgb.shape == overlay_img.shape
        assert rgb.shape == overlay_texture.shape

        rgb = rgb.to(torch.float32)  # [b, H, W, 3]

        rgb_ret = overlay_img * mask  # [b, H, W, 3]
        rgb_ret += rgb * (1 - mask)  # [b, H, W, 3]

        rgb_ret = torch.clamp(rgb_ret, 0, 255)  # [b, H, W, 3]
        rgb_ret = rgb_ret.to(torch.uint8)  # [b, H, W, 3]

        # rgb = rgb * (1 - mask) + overlay_img * mask
        # rgb = rgb * 0.5 + overlay_img * 0.5 # "debug" mode

        return rgb_ret

    def get_obs(self, info: dict = None):
        obs = super().get_obs(info)

        # "greenscreen" process
        if self.obs_mode_struct.visual.rgb and self.obs_mode_struct.visual.segmentation and self.overlay_images_numpy:
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            camera_name = self.rgb_camera_name
            assert "segmentation" in obs["sensor_data"][camera_name].keys()

            overlay_img = self.overlay_images.to(obs["sensor_data"][camera_name]["rgb"].device)
            overlay_texture = self.overlay_textures.to(obs["sensor_data"][camera_name]["rgb"].device)
            overlay_mix = self.overlay_mix.to(obs["sensor_data"][camera_name]["rgb"].device)

            green_screened_rgb = self._green_sceen_rgb(
                obs["sensor_data"][camera_name]["rgb"],
                obs["sensor_data"][camera_name]["segmentation"],
                overlay_img,
                overlay_texture,
                overlay_mix
            )
            obs["sensor_data"][camera_name]["rgb"] = green_screened_rgb
        return obs

    # widowx
    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
            ),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )


@register_env("PutOnPlateInScene25Main-v3", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25MainV3(PutOnPlateInScene25):
    def __init__(self, **kwargs):
        self._prep_init()

        super().__init__(**kwargs)

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lc = 16
            lc_offset = 0
            lo = 16
            lo_offset = 0
        elif obj_set == "test":
            lc = 9
            lc_offset = 16
            lo = 5
            lo_offset = 16
        elif obj_set == "all":
            lc = 25
            lc_offset = 0
            lo = 21
            lo_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs


@register_env("PutOnPlateInScene25Single-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25Single(PutOnPlateInScene25MainV3):
    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        only_carrot_name = list(self.model_db_carrot.keys())[0]
        self.model_db_carrot = {k: v for k, v in self.model_db_carrot.items() if k == only_carrot_name}
        assert len(self.model_db_carrot) == 1

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )
        only_table_name = list(model_db_table.keys())[0]
        model_db_table = {k: v for k, v in model_db_table.items() if k == only_table_name}
        assert len(model_db_table) == 1

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 1
        assert len(self.overlay_textures_numpy) == 1
        assert len(self.overlay_mix_numpy) == 1

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lc = 1
            lc_offset = 0
        elif obj_set == "test":
            lc = 1
            lc_offset = 0
        elif obj_set == "all":
            lc = 1
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lo = 1
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2


@register_env("PutOnPlateInScene25MainCarrot-v3", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25MainCarrotV3(PutOnPlateInScene25MainV3):
    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lc = 16
            lc_offset = 0
        elif obj_set == "test":
            lc = 9
            lc_offset = 16
        elif obj_set == "all":
            lc = 25
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lo = 1
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2


@register_env("PutOnPlateInScene25MainImage-v3", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25MainImageV3(PutOnPlateInScene25MainV3):
    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        only_carrot_name = list(self.model_db_carrot.keys())[0]
        self.model_db_carrot = {k: v for k, v in self.model_db_carrot.items() if k == only_carrot_name}
        assert len(self.model_db_carrot) == 1

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lo = 16
            lo_offset = 0
        elif obj_set == "test":
            lo = 5
            lo_offset = 16
        elif obj_set == "all":
            lo = 21
            lo_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 1
        lc_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2


# Vision

@register_env("PutOnPlateInScene25VisionImage-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25VisionImage(PutOnPlateInScene25MainV3):
    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lo = 16
            lo_offset = 0
        elif obj_set == "test":
            lo = 5
            lo_offset = 16
        elif obj_set == "all":
            lo = 21
            lo_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2


@register_env("PutOnPlateInScene25VisionTexture03-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25VisionTexture03(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    overlay_texture_mix_ratio = 0.3

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            le = 1
            le_offset = 0
        elif obj_set == "test":
            le = 16
            le_offset = 1
        elif obj_set == "all":
            le = 17
            le_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * le * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (le * lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_extra_ids = (episode_id // (lp * lo * l1 * l2)) % le + le_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_extra_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_extra_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img, overlay_texture, overlay_mix):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        # mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(actor_seg.device)

        robot_item_ids = torch.concat([self.robot_link_ids, self.target_object_actor_ids])
        arm_obj_mask = torch.isin(actor_seg, robot_item_ids)  # [b, H, W]

        mask = (~arm_obj_mask).to(torch.float32).unsqueeze(-1)  # [b, H, W, 1]
        mix = overlay_mix.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 1]
        mix = mix * self.overlay_texture_mix_ratio
        assert rgb.shape == overlay_img.shape
        assert rgb.shape == overlay_texture.shape

        # Step 1
        b, H, W, _ = mask.shape
        boxes = masks_to_boxes_pytorch(arm_obj_mask)  # [b, 4], [xmin, ymin, xmax, ymax]

        # Step 2
        xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(4)]  # [b]
        h_box = (ymax - ymin + 1).clamp(min=1)  # [b]
        w_box = (xmax - xmin + 1).clamp(min=1)  # [b]

        # Step 3
        max_h, max_w = h_box.max().item(), w_box.max().item()
        texture = overlay_texture.permute(0, 3, 1, 2).float()  # [b, 3, H_tex, W_tex]
        texture_resized = F.interpolate(texture, size=(max_h, max_w), mode='bilinear', align_corners=False)
        # [b, 3, max_h, max_w]

        # Step 4
        rgb = rgb.to(torch.float32)
        rgb_ret = overlay_img * mask

        for i in range(b):
            tex_crop = texture_resized[i, :, :h_box[i], :w_box[i]].permute(1, 2, 0)  # [h_box, w_box, 3]
            y0, y1 = ymin[i].item(), (ymin[i] + h_box[i]).item()
            x0, x1 = xmin[i].item(), (xmin[i] + w_box[i]).item()
            rgb_box = rgb[i, y0:y1, x0:x1, :]  # [h_box, w_box, 3]
            overlay_img_box = overlay_img[i, y0:y1, x0:x1, :]  # [h_box, w_box, 3]
            mask_box = arm_obj_mask[i, y0:y1, x0:x1]  # [h_box, w_box]
            mix_val = mix[i].item()

            mask_box_3 = mask_box.unsqueeze(-1).to(rgb_box.dtype)  # [h_box, w_box, 1]
            blended = tex_crop * mix_val + rgb_box * (1.0 - mix_val)
            out_box = blended * mask_box_3 + overlay_img_box * (1 - mask_box_3)

            rgb_ret[i, y0:y1, x0:x1, :] = out_box

        rgb_ret = torch.clamp(rgb_ret, 0, 255)
        rgb_ret = rgb_ret.to(torch.uint8)

        return rgb_ret


@register_env("PutOnPlateInScene25VisionTexture05-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25VisionTexture05(PutOnPlateInScene25VisionTexture03):
    overlay_texture_mix_ratio = 0.5


@register_env("PutOnPlateInScene25VisionWhole03-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25VisionWhole03(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    overlay_texture_mix_ratio = 0.3

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        only_plate_name = list(self.model_db_plate.keys())[0]
        self.model_db_plate = {k: v for k, v in self.model_db_plate.items() if k == only_plate_name}
        assert len(self.model_db_plate) == 1

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (1280, 960))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            le = 1
            le_offset = 0
        elif obj_set == "test":
            le = 16
            le_offset = 1
        elif obj_set == "all":
            le = 17
            le_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * le * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (le * lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_extra_ids = (episode_id // (lp * lo * l1 * l2)) % le + le_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_extra_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_extra_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img, overlay_texture, overlay_mix):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        # mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(actor_seg.device)

        robot_item_ids = torch.concat([self.robot_link_ids, self.target_object_actor_ids])
        arm_obj_mask = torch.isin(actor_seg, robot_item_ids)  # [b, H, W]

        mask = (~arm_obj_mask).to(torch.float32).unsqueeze(-1)  # [b, H, W, 1]
        mix = overlay_mix.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [b, 1, 1, 1]
        mix = mix * self.overlay_texture_mix_ratio
        assert rgb.shape == overlay_img.shape
        assert rgb.shape[1] * 2 == overlay_texture.shape[1]
        assert rgb.shape[2] * 2 == overlay_texture.shape[2]

        b, H, W, _ = mask.shape
        boxes = masks_to_boxes_pytorch(arm_obj_mask)  # [b, 4], [xmin, ymin, xmax, ymax]

        xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(4)]  # [b]
        x_mean = torch.clamp((xmin + xmax) // 2, min=1, max=639)
        y_mean = torch.clamp((ymin + ymax) // 2, min=1, max=479)

        rgb = rgb.to(torch.float32)
        rgb_ret = overlay_img * mask + rgb * (1 - mask)

        for i in range(b):
            ym = y_mean[i]
            xm = x_mean[i]
            tex_crop = overlay_texture[i, ym:ym + 480, xm:xm + 640, :]  # [h_box, w_box, 3]

            mix_val = mix[i].item()
            rgb_ret[i] = rgb_ret[i] * (1 - mix_val) + tex_crop * mix_val

        rgb_ret = torch.clamp(rgb_ret, 0, 255)
        rgb_ret = rgb_ret.to(torch.uint8)

        return rgb_ret


@register_env("PutOnPlateInScene25VisionWhole05-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25VisionWhole05(PutOnPlateInScene25VisionWhole03):
    overlay_texture_mix_ratio = 0.5


# Language

@register_env("PutOnPlateInScene25Carrot-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25Carrot(PutOnPlateInScene25MainV3):
    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lc = 16
            lc_offset = 0
        elif obj_set == "test":
            lc = 9
            lc_offset = 16
        elif obj_set == "all":
            lc = 25
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2


@register_env("PutOnPlateInScene25Instruct-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25Instruct(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            le = 1
            le_offset = 0
        elif obj_set == "test":
            le = 16
            le_offset = 1
        elif obj_set == "all":
            le = 17
            le_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * le * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (le * lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_extra_ids = (episode_id // (lp * lo * l1 * l2)) % le + le_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def get_language_instruction(self):
        templates = [
            "put $C$ on $P$",
            "Place the $C$ on the $P$",
            "set $C$ on $P$",
            "move the $C$ to the $P$",
            "Take the $C$ and put it on the $P$",

            "pick up $C$ and set it down on $P$",
            "please put the $C$ on the $P$",
            "Put $C$ onto $P$.",
            "place the $C$ onto the $P$ surface",
            "Make sure $C$ is on $P$.",

            "on the $P$, put the $C$",
            "put the $C$ where the $P$ is",
            "Move the $C$ from the table to the $P$",
            "Move $C$ so it’s on $P$.",
            "Can you put $C$ on $P$?",

            "$C$ on the $P$, please.",

            "the $C$ should be placed on the $P$.",  # test
            "Lay the $C$ down on the $P$.",
            "could you place $C$ over $P$",
            "position the $C$ atop the $P$",
            "Arrange for the $C$ to be resting on $P$.",
        ]
        assert len(templates) == 21
        temp_idx = self.select_extra_ids

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]

        instruct = []
        for idx in range(self.num_envs):
            carrot_name = self.model_db_carrot[select_carrot[idx]]["name"]
            plate_name = self.model_db_plate[select_plate[idx]]["name"]

            temp = templates[temp_idx[idx]]
            temp = temp.replace("$C$", carrot_name)
            temp = temp.replace("$P$", plate_name)
            instruct.append(temp)

            # instruct.append(f"put {carrot_name} on {plate_name}")

        return instruct


@register_env("PutOnPlateInScene25Plate-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25Plate(PutOnPlateInScene25MainV3):
    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        assert len(self.model_db_plate) == 17

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lp = 1
            lp_offset = 0
        elif obj_set == "test":
            lp = 16
            lp_offset = 1
        elif obj_set == "all":
            lp = 17
            lp_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp + lp_offset
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2


@register_env("PutOnPlateInScene25MultiCarrot-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25MultiCarrot(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                for k, grid_pos_3 in enumerate(grid_pos):
                    if (
                            np.linalg.norm(grid_pos_1 - grid_pos_2) > 0.070
                            and np.linalg.norm(grid_pos_3 - grid_pos_2) > 0.070
                            and np.linalg.norm(grid_pos_1 - grid_pos_3) > 0.15
                    ):
                        xyz_configs.append(
                            np.array(
                                [
                                    np.append(grid_pos_1, 0.95),  # carrot
                                    np.append(grid_pos_2, 0.92),  # plate
                                    np.append(grid_pos_3, 1.0),  # extra carrot
                                ]
                            )
                        )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

        print(f"xyz_configs: {xyz_configs.shape}")
        print(f"quat_configs: {quat_configs.shape}")

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lc = 16
            lc_offset = 0
        elif obj_set == "test":
            lc = 9
            lc_offset = 16
        elif obj_set == "all":
            lc = 25
            lc_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        le = lc - 1
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * le * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (le * lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_extra_ids = (episode_id // (lp * lo * l1 * l2)) % le  # [b]
        self.select_extra_ids = (self.select_carrot_ids + self.select_extra_ids + 1) % lc + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        select_extra = [self.carrot_names[idx] for idx in self.select_extra_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]
        extra_actor = [self.objs_carrot[n] for n in select_extra]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            is_select = self.select_carrot_ids == idx  # [b]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            is_select_extra = self.select_extra_ids == idx  # [b]
            p_select_extra = xyz_configs[self.select_pos_ids, 2].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]
            p = torch.where(is_select_extra.unsqueeze(1).repeat(1, 3), p_select_extra, p)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]
            q = torch.where(is_select_extra.unsqueeze(1).repeat(1, 4), q_select, q)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])
        e_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(extra_actor)])
        e_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(extra_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin) + torch.linalg.norm(e_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang) + torch.linalg.norm(e_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )


@register_env("PutOnPlateInScene25MultiPlate-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25MultiPlate(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    def _prep_init(self):
        # models
        self.model_db_carrot: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_carrot" / "model_db.json"
        )
        assert len(self.model_db_carrot) == 25

        self.model_db_plate: dict[str, dict] = io_utils.load_json(
            CARROT_DATASET_DIR / "more_plate" / "model_db.json"
        )
        assert len(self.model_db_plate) == 17

        # random configs
        self.carrot_names = list(self.model_db_carrot.keys())
        self.plate_names = list(self.model_db_plate.keys())

        # rgb overlay
        model_db_table = io_utils.load_json(
            CARROT_DATASET_DIR / "more_table" / "model_db.json"
        )

        img_fd = CARROT_DATASET_DIR / "more_table" / "imgs"
        texture_fd = CARROT_DATASET_DIR / "more_table" / "textures"
        self.overlay_images_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(img_fd / k)), cv2.COLOR_BGR2RGB), (640, 480))
            for k in model_db_table  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_textures_numpy = [
            cv2.resize(cv2.cvtColor(cv2.imread(str(texture_fd / v["texture"])), cv2.COLOR_BGR2RGB), (640, 480))
            for v in model_db_table.values()  # [H, W, 3]
        ]  # (B) [H, W, 3]
        self.overlay_mix_numpy = [
            v["mix"] for v in model_db_table.values()  # []
        ]
        assert len(self.overlay_images_numpy) == 21
        assert len(self.overlay_textures_numpy) == 21
        assert len(self.overlay_mix_numpy) == 21

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                for k, grid_pos_3 in enumerate(grid_pos):
                    if (
                            np.linalg.norm(grid_pos_1 - grid_pos_2) > 0.070
                            and np.linalg.norm(grid_pos_3 - grid_pos_2) > 0.150
                            and np.linalg.norm(grid_pos_1 - grid_pos_3) > 0.070
                    ):
                        xyz_configs.append(
                            np.array(
                                [
                                    np.append(grid_pos_1, 1.0),  # carrot
                                    np.append(grid_pos_2, 0.92),  # plate
                                    np.append(grid_pos_3, 0.96),  # extra plate
                                ]
                            )
                        )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

        print(f"xyz_configs: {xyz_configs.shape}")
        print(f"quat_configs: {quat_configs.shape}")

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            lp = 1
            lp_offset = 0
            le = 16
            le_mod = 17
        elif obj_set == "test":
            lp = 16
            lp_offset = 1
            le = 15
            le_mod = 16
        elif obj_set == "all":
            lp = 17
            lp_offset = 0
            le = 16
            le_mod = 17
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lo = len(self.overlay_images_numpy)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * le * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * le * lo * l1 * l2)  # [b]
        self.select_plate_ids = (episode_id // (le * lo * l1 * l2)) % lp
        self.select_extra_ids = (episode_id // (lo * l1 * l2)) % le
        self.select_extra_ids = (self.select_plate_ids + self.select_extra_ids + 1) % le_mod
        self.select_plate_ids += lp_offset
        self.select_extra_ids += lp_offset

        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        select_extra = [self.plate_names[idx] for idx in self.select_extra_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]
        extra_actor = [self.objs_plate[n] for n in select_extra]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            is_select = self.select_carrot_ids == idx  # [b]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            is_select = self.select_plate_ids == idx  # [b]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            is_select_extra = self.select_extra_ids == idx  # [b]
            p_select_extra = xyz_configs[self.select_pos_ids, 2].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]
            p = torch.where(is_select_extra.unsqueeze(1).repeat(1, 3), p_select_extra, p)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]
            q = torch.where(is_select_extra.unsqueeze(1).repeat(1, 4), q_select, q)

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])
        e_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(extra_actor)])
        e_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(extra_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin) + torch.linalg.norm(e_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang) + torch.linalg.norm(e_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)
        self.agent.reset(init_qpos=self.initial_qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )


# Action

@register_env("PutOnPlateInScene25Position-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25Position(PutOnPlateInScene25MainV3):
    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            l1 = self.xyz_configs_len1
            l1_offset = 0
            l2 = self.quat_configs_len1
            l2_offset = 0
        elif obj_set == "test":
            l1 = self.xyz_configs_len2
            l1_offset = self.xyz_configs_len1
            l2 = self.quat_configs_len2
            l2_offset = self.quat_configs_len1
        elif obj_set == "all":
            l1 = self.xyz_configs_len1 + self.xyz_configs_len2
            l1_offset = 0
            l2 = self.quat_configs_len1 + self.quat_configs_len2
            l2_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1 + l1_offset
        self.select_quat_ids = episode_id % l2 + l2_offset

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        # 1
        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs1 = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs1.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs1 = np.stack(xyz_configs1)

        quat_configs1 = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        # 2
        grid_pos = np.array([
            [-0.2, -0.2], [-0.2, 0.0], [-0.2, 0.2], [-0.2, 0.4], [-0.2, 0.6], [-0.2, 0.8], [-0.2, 1.0], [-0.2, 1.2],
            [0.0, -0.2], [0.0, 1.2],
            [0.2, -0.2], [0.2, 1.2],
            [0.4, -0.2], [0.4, 1.2],
            [0.6, -0.2], [0.6, 1.2],
            [0.8, -0.2], [0.8, 1.2],
            [1.0, -0.2], [1.0, 1.2],
            [1.2, -0.2], [1.2, 0.0], [1.2, 0.2], [1.2, 0.4], [1.2, 0.6], [1.2, 0.8], [1.2, 1.0], [1.2, 1.2]
        ]) * 2 - 1  # [28, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs2 = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs2.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs2 = np.stack(xyz_configs2)

        quat_configs2 = np.stack(
            [
                np.array([euler2quat(0, 0, - np.pi / 8), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, - np.pi * 3 / 8), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, - np.pi * 5 / 8), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, - np.pi * 7 / 8), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs_len1 = xyz_configs1.shape[0]
        self.xyz_configs_len2 = xyz_configs2.shape[0]
        self.quat_configs_len1 = quat_configs1.shape[0]
        self.quat_configs_len2 = quat_configs2.shape[0]

        self.xyz_configs = np.concatenate([xyz_configs1, xyz_configs2], axis=0)
        self.quat_configs = np.concatenate([quat_configs1, quat_configs2], axis=0)

        assert self.xyz_configs.shape[0] == self.xyz_configs_len1 + self.xyz_configs_len2
        assert self.quat_configs.shape[0] == self.quat_configs_len1 + self.quat_configs_len2


@register_env("PutOnPlateInScene25EEPose-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25EEPose(PutOnPlateInScene25MainV3):
    select_extra_ids: torch.Tensor

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j and np.linalg.norm(grid_pos_2 - grid_pos_1) > 0.070:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.95),
                                np.append(grid_pos_2, 0.95),
                            ]
                        )
                    )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

        robot_qpos = []

        robot_qpos.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        for p0 in [-0.15, -0.08, 0.0, 0.08, 0.15]:
            for p1 in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                for p2 in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                    for p3 in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                        for p4 in [-0.15, -0.08, 0.0, 0.08, 0.15]:
                            for p5 in [-0.6, -0.3, 0.0, 0.3, 0.6]:
                                p012sum = p1 + p2
                                if abs(p012sum) > 0.25:
                                    continue
                                robot_qpos.append(np.array([p0, p1, p2, p3, p4, p5, 0.0, 0.0]))

        robot_qpos = np.stack(robot_qpos)
        self.robot_qpos = robot_qpos

        print(f"xyz_configs: {xyz_configs.shape}")
        print(f"quat_configs: {quat_configs.shape}")
        print(f"robot_qpos: {robot_qpos.shape}")

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            le = 1
            le_offset = 0
        elif obj_set == "test":
            le = len(self.robot_qpos) - 1
            le_offset = 1
        elif obj_set == "all":
            le = len(self.robot_qpos)
            le_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * le * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (le * lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_extra_ids = (episode_id // (lp * lo * l1 * l2)) % le + le_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_episode_pre(env_idx, options)

        b = self.num_envs

        # rgb overlay
        sensor = self._sensor_configs[self.rgb_camera_name]
        assert sensor.width == 640
        assert sensor.height == 480
        overlay_images = np.stack([self.overlay_images_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_images = torch.tensor(overlay_images, device=self.device)  # [b, H, W, 3]
        overlay_textures = np.stack([self.overlay_textures_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_textures = torch.tensor(overlay_textures, device=self.device)  # [b, H, W, 3]
        overlay_mix = np.array([self.overlay_mix_numpy[idx] for idx in self.select_overlay_ids])
        self.overlay_mix = torch.tensor(overlay_mix, device=self.device)  # [b]

        # xyz and quat
        xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
        quat_configs = torch.tensor(self.quat_configs, device=self.device)

        select_carrot = [self.carrot_names[idx] for idx in self.select_carrot_ids]
        select_plate = [self.plate_names[idx] for idx in self.select_plate_ids]
        carrot_actor = [self.objs_carrot[n] for n in select_carrot]
        plate_actor = [self.objs_plate[n] for n in select_plate]

        # for motion planning capability
        self.source_obj_name = select_carrot[0]
        self.target_obj_name = select_plate[0]
        self.objs = {
            self.source_obj_name: carrot_actor[0],
            self.target_obj_name: plate_actor[0]
        }

        # set pose for robot
        self.agent.robot.set_pose(self.safe_robot_pos)
        # self._settle(0.5)

        # set pose for objs
        for idx, name in enumerate(self.model_db_carrot):
            is_select = self.select_carrot_ids == idx  # [b]
            p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 0].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

        for idx, name in enumerate(self.model_db_plate):
            is_select = self.select_plate_ids == idx  # [b]
            p_reset = torch.tensor([2.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
            p_select = xyz_configs[self.select_pos_ids, 1].reshape(b, 3)  # [b, 3]
            p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

            q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
            q_select = quat_configs[self.select_quat_ids, 1].reshape(b, 4)  # [b, 4]
            q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

            self.objs_plate[name].set_pose(Pose.create_from_pq(p=p, q=q))

        self._settle(0.5)

        # Some objects need longer time to settle
        c_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(carrot_actor)])
        c_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(carrot_actor)])
        p_lin = torch.stack([a.linear_velocity[i] for i, a in enumerate(plate_actor)])
        p_ang = torch.stack([a.angular_velocity[i] for i, a in enumerate(plate_actor)])

        lin_vel = torch.linalg.norm(c_lin) + torch.linalg.norm(p_lin)
        ang_vel = torch.linalg.norm(c_ang) + torch.linalg.norm(p_ang)

        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(6)

        # measured values for bridge dataset
        self.agent.robot.set_pose(self.initial_robot_pos)

        robot_qpos = torch.tensor(self.robot_qpos, device=self.device)
        initial_qpos = torch.tensor(self.initial_qpos, device=self.device).reshape(1, -1)  # [1, 8]
        qpos = robot_qpos[self.select_extra_ids] + initial_qpos  # [b, 8]
        self.agent.reset(init_qpos=qpos)

        # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
        self.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
        self.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]
        corner_signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=self.device)

        # carrot
        carrot_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_carrot])  # [b, 3]
        c_bbox_half = carrot_bbox_world / 2  # [b, 3]
        c_bbox_corners = c_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        c_q_matrix = rotation_conversions.quaternion_to_matrix(self.carrot_q_after_settle)  # [b, 3, 3]
        c_bbox_corners_rot = torch.matmul(c_bbox_corners, c_q_matrix.transpose(1, 2))  # [b, 8, 3]
        c_rotated_bbox_size = c_bbox_corners_rot.max(dim=1).values - c_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.carrot_bbox_world = c_rotated_bbox_size  # [b, 3]

        # plate
        plate_bbox_world = torch.stack([self.model_bbox_sizes[n] for n in select_plate])  # [b, 3]
        p_bbox_half = plate_bbox_world / 2  # [b, 3]
        p_bbox_corners = p_bbox_half[:, None, :] * corner_signs[None, :, :]  # [b, 8, 3]

        p_q_matrix = rotation_conversions.quaternion_to_matrix(self.plate_q_after_settle)  # [b, 3, 3]
        p_bbox_corners_rot = torch.matmul(p_bbox_corners, p_q_matrix.transpose(1, 2))  # [b, 8, 3]
        p_rotated_bbox_size = p_bbox_corners_rot.max(dim=1).values - p_bbox_corners_rot.min(dim=1).values  # [b, 3]
        self.plate_bbox_world = p_rotated_bbox_size  # [b, 3]

        # stats to track
        self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32, device=self.device)
        self.episode_stats = dict(
            # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
            # moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
            # moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
            # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
            is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool, device=self.device),
            # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
            consecutive_grasp=torch.zeros((b,), dtype=torch.bool, device=self.device),
            src_on_target=torch.zeros((b,), dtype=torch.bool, device=self.device),

            gripper_carrot_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            gripper_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
            carrot_plate_dist=torch.zeros((b,), dtype=torch.float32, device=self.device),
        )


@register_env("PutOnPlateInScene25PositionChangeTo-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25PositionChange(PutOnPlateInScene25MainV3):
    can_change_position: bool
    change_position_timestep = 5

    def _generate_init_pose(self):
        xy_center = np.array([-0.16, 0.00]).reshape(1, 2)
        half_edge_length = np.array([0.075, 0.075]).reshape(1, 2)

        grid_pos = np.array([
            [0.0, 0.0], [0.0, 0.2], [0.0, 0.4], [0.0, 0.6], [0.0, 0.8], [0.0, 1.0],
            [0.2, 0.0], [0.2, 0.2], [0.2, 0.4], [0.2, 0.6], [0.2, 0.8], [0.2, 1.0],
            [0.4, 0.0], [0.4, 0.2], [0.4, 0.4], [0.4, 0.6], [0.4, 0.8], [0.4, 1.0],
            [0.6, 0.0], [0.6, 0.2], [0.6, 0.4], [0.6, 0.6], [0.6, 0.8], [0.6, 1.0],
            [0.8, 0.0], [0.8, 0.2], [0.8, 0.4], [0.8, 0.6], [0.8, 0.8], [0.8, 1.0],
            [1.0, 0.0], [1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8], [1.0, 1.0],
        ]) * 2 - 1  # [36, 2]
        grid_pos = grid_pos * half_edge_length + xy_center

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                for k, grid_pos_3 in enumerate(grid_pos):
                    if (
                            np.linalg.norm(grid_pos_1 - grid_pos_2) > 0.070
                            and np.linalg.norm(grid_pos_3 - grid_pos_2) > 0.070
                            and np.linalg.norm(grid_pos_1 - grid_pos_3) > 0.15
                    ):
                        xyz_configs.append(
                            np.array(
                                [
                                    np.append(grid_pos_1, 0.95),  # carrot
                                    np.append(grid_pos_2, 0.92),  # plate
                                    np.append(grid_pos_3, 1.0),  # extra carrot
                                ]
                            )
                        )
        xyz_configs = np.stack(xyz_configs)

        quat_configs = np.stack(
            [
                np.array([euler2quat(0, 0, 0.0), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 4), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
                np.array([euler2quat(0, 0, np.pi * 3 / 4), [1, 0, 0, 0]]),
            ]
        )

        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs

        print(f"xyz_configs: {xyz_configs.shape}")
        print(f"quat_configs: {quat_configs.shape}")

        self.can_change_position = False

    def _initialize_episode_pre(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        b = len(env_idx)
        assert b == self.num_envs

        obj_set = options.get("obj_set", "train")
        if obj_set == "train":
            self.can_change_position = False
        elif obj_set == "test":
            self.can_change_position = True
        elif obj_set == "all":
            self.can_change_position = True
        else:
            raise ValueError(f"Unknown obj_set: {obj_set}")

        lc = 16
        lc_offset = 0
        lo = 16
        lo_offset = 0
        lp = len(self.plate_names)
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        ltt = lc * lp * lo * l1 * l2

        # rand and select
        episode_id = options.get("episode_id",
                                 torch.randint(low=0, high=ltt, size=(b,), device=self.device))
        episode_id = episode_id.reshape(b)
        episode_id = episode_id % ltt

        self.select_carrot_ids = episode_id // (lp * lo * l1 * l2) + lc_offset  # [b]
        self.select_plate_ids = (episode_id // (lo * l1 * l2)) % lp
        self.select_overlay_ids = (episode_id // (l1 * l2)) % lo + lo_offset
        self.select_pos_ids = (episode_id // l2) % l1
        self.select_quat_ids = episode_id % l2

    def evaluate(self, success_require_src_completely_on_target=True):
        if self.can_change_position and self.elapsed_steps[0].item() == self.change_position_timestep:
            b = self.num_envs

            xyz_configs = torch.tensor(self.xyz_configs, device=self.device)
            quat_configs = torch.tensor(self.quat_configs, device=self.device)

            for idx, name in enumerate(self.model_db_carrot):
                p_reset = torch.tensor([1.0, 0.3 * idx, 1.0], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 3]
                is_select = self.select_carrot_ids == idx  # [b]
                p_select = xyz_configs[self.select_pos_ids, 2].reshape(b, 3)  # [b, 3]
                p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)  # [b, 3]

                q_reset = torch.tensor([0, 0, 0, 1], device=self.device).reshape(1, -1).repeat(b, 1)  # [b, 4]
                q_select = quat_configs[self.select_quat_ids, 0].reshape(b, 4)  # [b, 4]
                q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)  # [b, 4]

                self.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

            self._settle(0.5)

        return super().evaluate(success_require_src_completely_on_target)
