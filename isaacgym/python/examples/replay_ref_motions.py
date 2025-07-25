"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).

replay reference motion after retargeting

"""

import os
import math
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
from phc.utils.motion_lib_h1 import MotionLibH1
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
import torch
import sys
import time
import joblib
import json
def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)
class ReplayMotion():
    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee':300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     "wrist" : 20,
                    #  "index" : 2,
                    #  "middle" : 2,
                    #  "pinky" : 2,
                    #  "ring" : 2,
                    #  "thumb" : 2
                     }  # [N*m/rad]
        # d_gains
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     "wrist" : 5,
                    #  "index" : 10,
                    #  "middle" : 10,
                    #  "pinky" : 10,
                    #  "ring" : 10,
                    #  "thumb" : 10
                     }  # [N*m/rad]  # [N*m*s/rad]
        action_scale = 1
        decimation = 1
    class sim:
        # dt =  1.0 / 60.0  # 0.005
        # dt =  0.02  # 0.005
        dt =  1.0 / 30.0  
        # dt =  1.0 / 5.0  # fast action
        substeps = 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class init_state:
        h1_default_joint_angles = {
            "left_hip_yaw_joint" : 0.,
            "left_hip_roll_joint" : 0.,          
            "left_hip_pitch_joint" : -0.4,
            "left_knee_joint" : 0.8,  
            "left_ankle_joint" : -0.4,  
            "right_hip_yaw_joint" : 0., 
            "right_hip_roll_joint" : 0,
            "right_hip_pitch_joint" : -0.4,                                    
            "right_knee_joint" : 0.8,                                   
            "right_ankle_joint" : -0.4,                                  
            "torso_joint" : 0., 
            "left_shoulder_pitch_joint" : 0.,
            "left_shoulder_roll_joint" : 0,
            "left_shoulder_yaw_joint" : 0.,
            "left_elbow_joint"  : 0.,
            "right_shoulder_pitch_joint" : 0.,
            "right_shoulder_roll_joint" : 0.0,
            "right_shoulder_yaw_joint" : 0.,
            "right_elbow_joint" : 0.
        }
        h1_2_default_joint_angles = { # : target angles [rad] when action : 0.0
            "left_hip_yaw_joint" : 0., 
            "left_hip_roll_joint" : 0,            
            "left_hip_pitch_joint" : -0.4,
            "left_knee_joint" : 0.8,
            "left_ankle_pitch_joint" : -0.4  ,
            "left_ankle_roll_joint" : 0.,
            "right_hip_yaw_joint" : 0. ,
            "right_hip_roll_joint" : 0 ,
            "right_hip_pitch_joint" : -0.4,
            "right_knee_joint" : 0.8,                                    
            "right_ankle_pitch_joint" : -0.4,
            "right_ankle_roll_joint" : 0.,                                  
            "torso_joint" : 0.,
            "left_shoulder_pitch_joint" : 0. ,
            "left_shoulder_roll_joint" : 0 ,
            "left_shoulder_yaw_joint" : 0.,
            "left_elbow_joint"  : 0.,
            "right_shoulder_pitch_joint" : 0.,
            "right_shoulder_roll_joint" : 0.0,
            "right_shoulder_yaw_joint" : 0.,
            "right_elbow_joint" : 0.,
            }
        g1_default_joint_angles = {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            }


    def __init__(self, args):
        self.args = args
        # initialize gym
        # self.device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))
        self.device = torch.device("cpu")
        self.gym = gymapi.acquire_gym()

        # configure sim
        self.sim_params = gymapi.SimParams()

        # self.sim_params.up_axis = gymapi.UP_AXIS_Z
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        cfg = {"sim": class_to_dict(self.sim)}
        gymutil.parse_sim_config(cfg["sim"], self.sim_params)

        # if args.physics_engine == gymapi.SIM_FLEX:
        #     pass
        # elif args.physics_engine == gymapi.SIM_PHYSX:
        #     self.sim_params.physx.solver_type = 1
        #     self.sim_params.physx.num_position_iterations = 4
        #     self.sim_params.physx.num_velocity_iterations = 0
        #     self.sim_params.physx.use_gpu = True
        #     self.sim_params.physx.num_threads = 4
        #     # self.sim_params.contact_offset = 0.01
        #     # self.sim_params.rest_offset = 0.0
        #     self.sim_params.bounce_threshold_velocity = 0.2
        #     self.sim_params.max_depenetration_velocity = 10  # 1.0
        #     self.sim_params.max_gpu_contact_pairs = 16777216 #  -> needed for 8000 envs and more
        #     self.sim_params.default_buffer_size_multiplier = 10
        #     self.sim_params.contact_collection = 2 
        self.sim_params.use_gpu_pipeline = False

        # if args.use_gpu_pipeline:
        #     print("WARNING: Forcing CPU pipeline.")


        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params)

        # # initialize gym
        # self.device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))
        # self.gym = gymapi.acquire_gym()

        # # configure sim
        # self.sim_params = gymapi.SimParams()
        # self.sim_params.dt = dt = 1.0 / 60.0
        # self.sim_params.up_axis = gymapi.UP_AXIS_Z
        # self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # # self.sim_params.substeps = 2
        # if args.physics_engine == gymapi.SIM_FLEX:
        #     pass
        # elif args.physics_engine == gymapi.SIM_PHYSX:
        #     self.sim_params.physx.solver_type = 1
        #     self.sim_params.physx.num_position_iterations = 6
        #     self.sim_params.physx.num_velocity_iterations = 0
        #     # self.sim_params.physx.rest_offset = 0.0
        #     # self.sim_params.physx.contact_offset = 0.02
        #     # self.sim_params.physx.friction_offset_threshold = 0.001
        #     # self.sim_params.physx.friction_correlation_distance = 0.0005
        #     self.sim_params.physx.num_threads = args.num_threads
        #     self.sim_params.physx.use_gpu = args.use_gpu
        #     self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline

        # self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params)
                
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        from datetime import datetime as dt
        self.save_video = False
        self.save_video_dir = os.path.join(self.args.output_path, "imgs/{}/".format(dt.now().strftime("%Y-%m-%d_%H-%M-%S")))
        self.save_img_count = 0
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L,"toggle_video_record")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "RESET")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "NEXT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "Save_motion_to_yes")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "Save_motion_to_no")

        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        # set up the env grid
        self.num_envs = 1
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.num_per_row = 1
        self.spacing = 1.5
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.start_idx = 0
        self.dt = self.control.decimation * self.sim_params.dt
        # self.dt = self.sim_params.dt
        self.enable_viewer_sync = True
        self._load_and_set_asset()
        self._create_envs_with_actors()

        self._init_state_buffers()
        self._load_motion()
        
        self.motion_files = joblib.load(args.motion_path)
        self.keys = [*self.motion_files]

        if os.path.isfile("../../../legged_gym/resources/yes.json") and os.path.isfile("../../../legged_gym/resources/no.json"):
            with open('../../../legged_gym/resources/yes.json', 'r') as f:
                self.yes_list = json.load(f)
            with open('../../../legged_gym/resources/no.json', 'r') as f:
                self.no_list = json.load(f)
            self.start_idx = max(self.yes_list[-1][1], self.no_list[-1][1]) + 1
        else:
            self.yes_list = []
            self.no_list = []
            self.start_idx = 0
        
        print('')

    def _load_and_set_asset(self):
        
        print(self.args.asset_filename)
        if os.path.exists(self.args.asset_filename):
            asset_descriptor = AssetDesc(self.args.asset_filename, False)
        else:
            print("*** Invalid asset_filename specified.")
            quit()
        asset_root = os.path.dirname(asset_descriptor.file_name)
        asset_file = os.path.basename(asset_descriptor.file_name)

        asset_options = gymapi.AssetOptions()
        if self.args.control_type == "torque":
            asset_options.default_dof_drive_mode = 3
        else:
            asset_options.default_dof_drive_mode = 1
          # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000
        asset_options.max_linear_velocity = 1000
        asset_options.armature = 0.
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True

        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def lookat(self, i):
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
        look_at_pos = self.root_states[i, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def _create_envs_with_actors(self):
        # get array of DOF names
        self.dof_names = self.gym.get_asset_dof_names(self.asset)

        # get array of DOF properties
        dof_props_asset = self.gym.get_asset_dof_properties(self.asset)
        if self.args.control_type == "torque":
            dof_props_asset["driveMode"] = gymapi.DOF_MODE_EFFORT
        else:
            dof_props_asset["driveMode"] = gymapi.DOF_MODE_POS
        # create an array of DOF states that will be used to update the actors
        self.num_dof = self.gym.get_asset_dof_count(self.asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        dof_states = np.zeros(self.num_dof, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        dof_types = [self.gym.get_asset_dof_type(self.asset, i) for i in range(self.num_dof)]

        # get the position slice of the DOF state array
        dof_positions = dof_states['pos']


        env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)

        self.free_cam = True
        # position the camera
        cam_pos = gymapi.Vec3(7, 1.5, 3.0)
        cam_target = gymapi.Vec3(0, 1.5, 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # cache useful handles
        self.envs = []
        self.actor_handles = []
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        init_z = 0.0
        self.env_origins[:,2] = init_z
        print("Creating %d environments" % self.num_envs)
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, self.num_per_row)
            self.envs.append(env)

            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, init_z)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            actor_handle = self.gym.create_actor(env, self.asset, pose, "actor", i, 0)  # 0 enable self_collision, 1 disable
            self.actor_handles.append(actor_handle)
            dof_props = self._process_dof_props(dof_props_asset, i)
            old_state = self.gym.get_actor_dof_properties(env, actor_handle)
            set_state = self.gym.set_actor_dof_properties(env, actor_handle, dof_props)  # env_handle is a gym obj
            new_state = self.gym.get_actor_dof_properties(env, actor_handle)
            if set_state:
                print("ok")
                print(new_state)
            # set default DOF positions
            self.gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
        self.gym.prepare_sim(self.sim) # !!!!!!!!!!!!!!!!!!!!!!!!!
        # show dof props
        # ---------------------------------------------------------------------------------
        # get the limit-related slices of the DOF properties array
        stiffnesses = dof_props['stiffness']
        dampings = dof_props['damping']
        armatures = dof_props['armature']
        has_limits = dof_props['hasLimits']
        lower_limits = dof_props['lower']
        upper_limits = dof_props['upper']

        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.zeros(self.num_dof)
        speeds = np.zeros(self.num_dof)
        for i in range(self.num_dof):
            if has_limits[i]:
                if dof_types[i] == gymapi.DOF_ROTATION:
                    lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                    upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
                # make sure our default position is in range
                if lower_limits[i] > 0.0:
                    defaults[i] = lower_limits[i]
                elif upper_limits[i] < 0.0:
                    defaults[i] = upper_limits[i]
            else:
                # set reasonable animation limits for unlimited joints
                if dof_types[i] == gymapi.DOF_ROTATION:
                    # unlimited revolute joint
                    lower_limits[i] = -math.pi
                    upper_limits[i] = math.pi
                elif dof_types[i] == gymapi.DOF_TRANSLATION:
                    # unlimited prismatic joint
                    lower_limits[i] = -1.0
                    upper_limits[i] = 1.0
            # set DOF position to default
            dof_positions[i] = defaults[i]
            # set speed depending on DOF type and range of motion
            if dof_types[i] == gymapi.DOF_ROTATION:
                speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
            else:
                speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

        # Print DOF properties
        for i in range(self.num_dof):
            print("DOF %d" % i, self.dof_names[i])
            # print("  Name:     '%s'" % self.dof_names[i])
            # print("  Type:     %s" % self.gym.get_dof_type_string(dof_types[i]))
            # print("  Stiffness:  %r" % stiffnesses[i])
            # print("  Damping:  %r" % dampings[i])
            # print("  Armature:  %r" % armatures[i])
            # print("  Limited?  %r" % has_limits[i])
            # if has_limits[i]:
            #     print("    Lower   %f" % lower_limits[i])
            #     print("    Upper   %f" % upper_limits[i])
        # rigid body props
        # --------------------------------------------------------------------------------
        # get array of rigid body names
        rigid_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], 0)

        # get array of rigid body properties
        rigid_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], 0)

        # get array of rigid shape properties
        rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[0], 0)

        # Print rigid body properties 
        mass = 0
        for i in range(len(rigid_body_names)):
            print("Rigid Body %d :" % i, rigid_body_names[i])
            # print("  Name:     '%s'" % rigid_body_names[i])
            # print("  Mass:     %f" % rigid_body_props[i].mass)
            mass += rigid_body_props[i].mass
            # print("  CoM: {:.3}, {:.3}, {:.3}".format(rigid_body_props[i].com.x, rigid_body_props[i].com.y, rigid_body_props[i].com.z))
            # print("  Inertia:")
            # print("    x:  {:.3}, {:.3}, {:.3}".format(rigid_body_props[i].inertia.x.x, rigid_body_props[i].inertia.x.y, rigid_body_props[i].inertia.x.z))
            # print("    y:  {:.3}, {:.3}, {:.3}".format(rigid_body_props[i].inertia.y.x, rigid_body_props[i].inertia.y.y, rigid_body_props[i].inertia.y.z))
            # print("    z:  {:.3}, {:.3}, {:.3}".format(rigid_body_props[i].inertia.z.x, rigid_body_props[i].inertia.z.y, rigid_body_props[i].inertia.z.z))
            # print("  Contact_offset:     '%s'" % rigid_shape_props[i].contact_offset)
            # print("  Filter:     '%d'" % rigid_shape_props[i].filter)
            # print("  Friction:     '%s'" % rigid_shape_props[i].friction)
            # print("  Rest_offset:     '%s'" % rigid_shape_props[i].rest_offset)
        print("Total mass : ", mass)

    def _init_state_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            if args.robot == "h1-2":
                angle = self.init_state.h1_2_default_joint_angles[name]
            elif args.robot == "g1":
                angle = self.init_state.g1_default_joint_angles[name]
            else:
                angle = self.init_state.h1_default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.control.stiffness[dof_name]
                    self.d_gains[i] = self.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _load_motion(self):
        self.ref_motion_cache = {}
        motion_path = self.args.motion_path
        skeleton_path = self.args.skeleton_path
        self._motion_lib = MotionLibH1(motion_file=motion_path, device=self.device, masterfoot_conifg=None, fix_height=False,multi_thread=False,mjcf_file=skeleton_path, extend_head=True, robot_name=self.args.robot) #multi_thread=True doesn't work
        sk_tree = SkeletonTree.from_mjcf(skeleton_path)

        self.skeleton_trees = [sk_tree] * self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False)
        # find specific motion name !!!!!!!!!!!!!!!!!!!!!!
        # for idx, name in enumerate(self._motion_lib._motion_data_keys):
        #     if name == "g1_dance1_subject2":
        #         self.start_idx = idx
        #         break
        # self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)


        self.motion_dt = self._motion_lib._motion_dt
        
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item() * 0.8
                
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * 0.85
                self.dof_pos_limits[i, 1] = m + 0.5 * r * 0.85
            print(self.torque_limits)
        # TODO pyx add: set props if use pd controller in gym
        if self.args.control_type == "position":
            for i in range(len(props)):
                props["driveMode"][i] = gymapi.DOF_MODE_POS # int(gymapi.DOF_MODE_POS) = 1 
                name = self.dof_names[i]
                found = False
                for dof_name in self.control.stiffness.keys():
                    if dof_name in name:
                        props["stiffness"][i] = self.control.stiffness[dof_name] 
                        props["damping"][i] = self.control.damping[dof_name]
                        found = True
                    if not found:
                        props["stiffness"][i] = 0.0
                        props["damping"][i] = 0.0 
                        if self.control.control_type in ["P", "V"]:
                            print(f"PD gain of joint {name} were not defined, setting them to zero")
            # props["stiffness"][i].fill(1000.0)
            # props["damping"][i].fill(200.0)
        return props
    
    def reset_idx(self, env_ids):  
        # reset buffers
        self.episode_length_buf[env_ids] = 0.
        self._reset_dofs(env_ids)
        self._reset_humanoid_root_states(env_ids)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)




    def _reset_dofs(self, env_ids):
        motion_times = (self.episode_length_buf) * self.dt # next frames so +1
        offset = self.env_origins
        motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        self.dof_pos[env_ids] = motion_res['dof_pos'][env_ids]
        self.dof_vel[env_ids] = motion_res['dof_vel'][env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    

    def _reset_humanoid_root_states(self, env_ids):
        # base position
        motion_times = (self.episode_length_buf) * self.dt # next frames so +1
        offset = self.env_origins
        motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        
        self.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
        self.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
        self.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids] # ZL: use random velicty initation should be more robust? 
        self.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
        
        self._rigid_body_pos[env_ids] = motion_res['rg_pos'][env_ids]
        self._rigid_body_rot[env_ids] = motion_res['rb_rot'][env_ids]
        self._rigid_body_vel[env_ids] =   motion_res['body_vel'][env_ids]
        self._rigid_body_ang_vel[env_ids] = motion_res['body_ang_vel'][env_ids]
            
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def render(self, sync_frame_time=False, filter_motion=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                if filter_motion:
                    with open("../../../legged_gym/resources/yes.json", "w") as f1:
                        json.dump(self.yes_list, f1)
                    
                    with open("../../../legged_gym/resources/no.json", "w") as f2:
                        json.dump(self.no_list, f2)
                sys.exit()
            if not self.free_cam:
                self.lookat(0)
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    if filter_motion:
                        with open("../../../legged_gym/resources/yes.json", "w") as f1:
                            json.dump(self.yes_list, f1)
                        
                        with open("../../../legged_gym/resources/no.json", "w") as f2:
                            json.dump(self.no_list, f2)
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_video_record" and evt.value > 0:
                    print("toggle_video_record")
                    self.save_img_count = 0
                    os.makedirs(self.save_video_dir, exist_ok=True)
                    self.save_video = not self.save_video
                elif evt.action == "RESET" and evt.value > 0:
                    self.reset_idx(torch.arange(self.num_envs, device=self.device))
                elif evt.action == "NEXT" and evt.value > 0:
                    self.start_idx += self.num_envs
                    self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
                    self.reset_idx(torch.arange(self.num_envs, device=self.device))
                elif evt.action == "Save_motion_to_yes" and evt.value > 0:
                    self.yes_list.append([self.keys[self.start_idx], self.start_idx])
                    print("Save %s to yes" % self.keys[self.start_idx])
                    if self.start_idx == len(self.keys) - 1 and filter_motion:
                        with open("../../../legged_gym/resources/yes.json", "w") as f1:
                            json.dump(self.yes_list, f1)
                            sys.exit()
                    else:
                        self.start_idx += self.num_envs
                        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
                        self.reset_idx(torch.arange(self.num_envs, device=self.device))

                elif evt.action == "Save_motion_to_no" and evt.value > 0:
                    self.no_list.append([self.keys[self.start_idx], self.start_idx])
                    print("Save %s to no" % self.keys[self.start_idx])
                    if self.start_idx == len(self.keys) - 1 and filter_motion:
                        with open("../../../legged_gym/resources/no.json", "w") as f1:
                            json.dump(self.no_list, f1)
                            sys.exit()
                    else:
                        self.start_idx += self.num_envs
                        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=[torch.zeros(17)] * self.num_envs, limb_weights=[np.zeros(10)] * self.num_envs, random_sample=False, start_idx=self.start_idx)
                        self.reset_idx(torch.arange(self.num_envs, device=self.device))

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.control.action_scale
        
        control_type = self.control.control_type
        if control_type=="P":
            # torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos)
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
# simple asset descriptor
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments
    

if __name__=='__main__':
    # parse arguments
    args = gymutil.parse_arguments(
        description="Joint monkey: Animate degree-of-freedom ranges",
        custom_parameters=[   # ../../assets/mjcf/h1_2.xml
            # "default": "/home/yixuanpan/sda/yixuanpan/human2humanoid/legged_gym/resources/robots/h1/h1.xml"
            {"name": "--output_path", "type": str, "help": "save video path", "default": "../../../video_output/imgs"},
            {"name": "--auto_save_all_videos", "type": bool, "help": "auto save video", "default": False},
            # {"name": "--asset_filename", "type": str, "help": "Asset filename", "default": "../../assets/mjcf/open_ai_assets/hand/shadow_hand.xml"},
            # {"name": "--asset_filename", "type": str, "help": "Asset filename", "default": "../../../legged_gym/resources/robots/h1_2/h1_2.urdf"},
            # {"name": "--asset_filename", "type": str, "help": "Asset filename", "default": "../../../legged_gym/resources/robots/h1/urdf/h1.urdf"},
            {"name": "--robot", "type": str, "help": "robot name: h1-2 or h1 or g1", "default": "g1"},   # 换h1记得把motion_lib_h1中的from phc.utils.torch_h1_2_humanoid_batch import Humanoid_Batch替换
            {"name": "--control_type", "type": str, "default": "torque", "help": "Animation speed scale"},
            # {"name": "--control_type", "type": str, "default": "position", "help": "Animation speed scale"},
            {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
            {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])
    if args.robot == "h1-2":
        # args.asset_filename = "../../../legged_gym/resources/robots/h1_2/h1_2.urdf"
        args.asset_filename = "../../../legged_gym/resources/robots/h1-2/urdf/h1_5_21dof.urdf"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/h1_2_21dof_amass_phc_filtered_all_less_bias_consistency.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/h1_2_21dof_amass_phc_filtered_all_less_bias_consistency_ordered.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/amass_20motion_h1_2.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/amass_motion_all_bias_2_h1_2.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/amass_motion_all_bias_h1_2.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/0-BioMotionLab_NTroje_rub007_0018_kicking1_poses.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/test_orientation.pkl"
        args.motion_path = "../../../legged_gym/resources/motions/h1/0-KIT_3_wave_left15_poses.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/wave_hand.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/stable_h1_2.pkl"
        args.skeleton_path = "../../../legged_gym/resources/robots/h1-2/xml/h1_5_21dof.xml"
    elif args.robot == "g1":
        args.asset_filename = "../../../legged_gym/resources/robots/g1_from_asap/g1_29dof_anneal_23dof_with_hand_pyx_0408.urdf"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/test_3_cmus.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/g1_stable_punch.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/g1_23dof_all_8k_fixed_waist_wrist_bias.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/lafan_g1_all_wo_jump_fall.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/custom_all.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Charleston_dance.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Bruce_Lee_pose.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Hooks_punch.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Horse-stance_pose.pkl"
        args.motion_path = "../../../legged_gym/resources/motions/g1/lafan_g1_all_wo_jump_fall.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Horse-stance_punch.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Roundhouse_kick.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/Side_kick.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/merged_demos.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/g1/lafan_g1_all_seg.pkl"
        args.skeleton_path = "../../../legged_gym/resources/robots/g1_from_asap/g1_29dof_anneal_23dof_pyx_0408.xml"
    else:
        args.asset_filename = "../../../legged_gym/resources/robots/h1/urdf/h1.urdf"
        args.motion_path = "../../../legged_gym/resources/motions/h1/amass_phc_filtered.pkl"
        # args.motion_path = "../../../legged_gym/resources/motions/h1/stable_punch.pkl"
        args.skeleton_path = "../../../legged_gym/resources/robots/h1/xml/h1.xml"
    replay = ReplayMotion(args)
    replay.reset_idx(torch.arange(replay.num_envs).to(replay.device))
    env_ids = torch.arange(replay.num_envs).to(replay.device)
    set_init_state = True
    cnt = 0
    # save imgs
    while not replay.gym.query_viewer_has_closed(replay.viewer):
        if replay.num_envs == 1 and args.auto_save_all_videos and replay.episode_length_buf == 0:
            replay.save_img_count = 0
            replay.save_video_dir = os.path.join(args.output_path, replay._motion_lib.curr_motion_keys)
            os.makedirs(replay.save_video_dir, exist_ok=True)
            replay.save_video = not replay.save_video
        motion_len = replay._motion_lib.get_motion_length(replay.motion_ids)
        motion_times = (replay.episode_length_buf) * replay.dt # next frames so +1
        if replay.num_envs == 1 and motion_times > motion_len and args.auto_save_all_videos:
            replay.start_idx += replay.num_envs
            replay._motion_lib.load_motions(skeleton_trees=replay.skeleton_trees, gender_betas=[torch.zeros(17)] * replay.num_envs, limb_weights=[np.zeros(10)] * replay.num_envs, random_sample=False, start_idx=replay.start_idx)
            replay.reset_idx(torch.arange(replay.num_envs, device=replay.device))
            replay.save_video = not replay.save_video
            if replay.start_idx >= replay._motion_lib.all_motion_len:
                break
            else:
                continue
        offset = replay.env_origins
        motion_res = replay._get_state_from_motionlib_cache(replay.motion_ids, motion_times, offset= offset)
            
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if set_init_state:
            replay.dof_pos[env_ids] = motion_res['dof_pos'][env_ids]
            replay.dof_vel[env_ids] = motion_res['dof_vel'][env_ids]
            replay.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            # if cnt == 0:
            #     cnt += 1
            #     init_root_pos = replay.root_states[env_ids, :3]
            # replay.root_states[env_ids, :3] = init_root_pos
            replay.root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            replay.root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            replay.root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids] # ZL: use random velicty initation should be more robust? 
            replay.root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            
            replay._rigid_body_pos[env_ids] = motion_res['rg_pos'][env_ids]
            replay._rigid_body_rot[env_ids] = motion_res['rb_rot'][env_ids]
            replay._rigid_body_vel[env_ids] =   motion_res['body_vel'][env_ids]
            replay._rigid_body_ang_vel[env_ids] = motion_res['body_ang_vel'][env_ids]
            replay.gym.set_dof_state_tensor_indexed(replay.sim,
                                                gymtorch.unwrap_tensor(replay.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

            replay.gym.set_actor_root_state_tensor_indexed(replay.sim,
                                                        gymtorch.unwrap_tensor(replay.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            set_init_state = True
        replay.gym.clear_lines(replay.viewer)
        for pos_joint in motion_res['rg_pos_t'][0, 1:]: # idx 0 torso (duplicate with 11)
            sphere_geom2 = gymutil.WireframeSphereGeometry(0.04, 20, 20, None, color=(1, 0.0, 0.0))
            sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
            gymutil.draw_lines(sphere_geom2, replay.gym, replay.viewer, replay.envs[0], sphere_pose) 
        replay.render()
                # render frame if we're saving video
        if replay.save_video and replay.viewer is not None:
            num = str(replay.save_img_count)
            num = '0' * (6 - len(num)) + num
            replay.gym.write_viewer_image_to_file(replay.viewer, f"{replay.save_video_dir}/frame_{num}.png")
            replay.save_img_count += 1

        actions_all = torch.ones(replay.num_envs, replay.num_dof).to(replay.device).to(dtype=torch.float32) * 0.
        # print("111")
        for idx in range(replay.control.decimation):
            # if idx == replay.control.decimation-1:
            #     print(1)
        #     if args.control_type == "torque":
        #         demo_dofs = motion_res['dof_pos'][env_ids].clone().contiguous()
        #         replay.torques = replay._compute_torques(demo_dofs).view(replay.torques.shape)
        #         # replay.torques = torch.zeros(replay.num_envs, replay.num_dof).to(replay.device)
        #         replay.gym.set_dof_actuation_force_tensor(replay.sim, gymtorch.unwrap_tensor(replay.torques))
        #     else:
        #         demo_dofs = motion_res['dof_pos'][env_ids].clone().contiguous()
        #         # print("demo_dofs_0:", demo_dofs_0)
        #         # demo_dofs = actions_all   # 给动作就飞，直接给demo_dofs就一切正常, 给的动作和现在动作差异较大就直接飞
        # #         demo_dofs = torch.tensor([[ 0.0976, -0.4417, -0.0568,  1.1723, -0.4699,  0.0000, -0.2651, -0.5395,
        # #  -0.0167,  1.2363, -0.4473,  0.0000,  0.0326, -0.0509, -0.0114,  0.1759,
        # #   1.3284,  0.0069,  0.1092,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        # #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        # #  -0.0603, -0.0505, -0.3738,  1.2789, -0.0093,  0.1059,  0.0000,  0.0000,
        # #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
        # #   0.0000,  0.0000,  0.0000]])   # 给动作就飞，直接给demo_dofs就一切正常
        #         # demo_dofs = torch.ones_like(demo_dofs, device=replay.device) * demo_dofs
        #         # demo_dofs = demo_dofs.to(dtype=torch.float32)
        #         # print("demo_dofs_1:", demo_dofs)
        #         replay.gym.set_dof_position_target_tensor(replay.sim, gymtorch.unwrap_tensor(demo_dofs))
        #         # targets = np.ones(replay.num_dof).astype('f') * np.array(demo_dofs[0])
        #         # replay.gym.set_actor_dof_position_targets(replay.envs[0], replay.actor_handles[0], gymtorch.unwrap_tensor(targets))
        #         # replay.gym.set_actor_dof_position_targets(replay.envs[0], replay.actor_handles[0], targets)

            replay.gym.simulate(replay.sim)
            replay.gym.fetch_results(replay.sim, True)
            replay.gym.refresh_dof_state_tensor(replay.sim)
            # print("cur:",replay.dof_pos)
            # print("ref:",demo_dofs)
            # replay.render()

        # time.sleep(0.1)
        replay.gym.refresh_actor_root_state_tensor(replay.sim)
        replay.gym.refresh_net_contact_force_tensor(replay.sim)
        replay.gym.refresh_rigid_body_state_tensor(replay.sim)
        replay.episode_length_buf += 1

    print("Done")

    replay.gym.destroy_viewer(replay.viewer)
    replay.gym.destroy_sim(replay.sim)