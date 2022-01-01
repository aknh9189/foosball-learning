import numpy as np
import os
import time

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

import torch
#from torch.tensor import Tensor
from typing import Tuple, Dict


class Foos(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        #load parameters
        self.rew_scales = {}

        #load randomization
        # (domain and initial condition)

        # plane params (probs not needed)
        
        # base init state
        
        #load default joint positions
        
        # other
        self.dt = sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self.cfg["env"]["numObservations"] = 48
        self.cfg["env"]["numActions"] = 12

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors ( CHECK THIS WITH FOOS ASSET)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        
        '''
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_actions = torch.zeros((self.num_envs, 12), device= self.device, dtype=torch.float, requires_grad=False)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device= self.device, dtype=torch.float, requires_grad=False)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), device= self.device, dtype=torch.float, requires_grad=False)
        self.foot_in_contact = torch.zeros((self.num_envs, 4), device= self.device, dtype=torch.float, requires_grad=False) == 1
        self.foot_first_contact = torch.zeros((self.num_envs, 4), device= self.device, dtype=torch.float, requires_grad=False) == 0
        self.foot_air_time = torch.zeros((self.num_envs, 4), device= self.device, dtype=torch.float, requires_grad=False)


        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
        '''
        # initialize some data used later on
        self.extras = {}
        torch_zeros = lambda : torch.zeros((self.num_envs, ), device=self.device, dtype= torch.float, requires_grad=False)
        self.reward_terms = {'rew1': torch_zeros(),
                             'rew2': torch_zeros(), 
                             }

        #self.initial_root_states = self.root_states.clone()
        #self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        #self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        #self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        #self.time_out_buf = torch.zeros_like(self.reset_buf)

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.getcwd()+"/assets"
        asset_file = "urdf/yobo_model/yobotics_description/urdf/yobotics.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        cheetah_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cheetah_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(cheetah_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(cheetah_asset)
        self.dof_names = self.gym.get_asset_dof_names(cheetah_asset)
        hip_names = [s for s in body_names if 'hip' in s]
        extremity_name = "calf" if asset_options.collapse_fixed_joints else "foot"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.hip_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "THIGH" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(cheetah_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.Kp
            dof_props['damping'][i] = self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.cheetah_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            cheetah_handle = self.gym.create_actor(env_ptr, cheetah_asset, start_pose, "cheetah", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, cheetah_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, cheetah_handle)
            self.envs.append(env_ptr)
            self.cheetah_handles.append(cheetah_handle)
    
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.cheetah_handles[0], "name_base")
        

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.push_cheetah:
            self.push_robots()
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]
        self.last_base_lin_vel[:] = self.base_lin_vel
        self.last_base_ang_vel[:] = self.base_ang_vel

    def compute_reward(self, actions):
        rew_1 = 0
        rew_2 = 0
        self.rew_buf = torch.clip(self.rew_buf, 0., None)
        
        # reset agents
        reset = False# RESET CONDITION
        time_out = self.progress_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = reset | time_out

        self.reward_terms['rew1'] += rew_1
        self.reward_terms['rew2'] += rew_2
        

        

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_obs(self.root_states)
    
    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        '''
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        norm_small = torch.norm(self.commands[env_ids, :2], dim =1 ) < 0.2

        idx = torch.where(norm_small)[0] 
        self.commands[idx, :2] = 0.
        norm_small = torch.abs(self.commands[env_ids, 2]) < 0.2
        idx = torch.where(norm_small)[0] 
        self.commands[idx, 2] = 0.
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_dof_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_base_ang_vel[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.foot_air_time[env_ids, :] = 0.
        self.foot_first_contact[env_ids, :] = True

        self.extras = {}
        for key in self.reward_terms.keys():
            self.extras['reward_'+key] = (torch.mean(self.reward_terms[key][env_ids])/self.max_episode_length_s).view(-1,1)
            self.reward_terms[key][env_ids] = 0
        '''


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_obs(root_states: Tensor) -> Tensor:
    pass