
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, euler_from_quat
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        # noise_vec = torch.zeros(47, device = self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    
        
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        self.upper_body_rpy = torch.zeros(self.num_envs,4)
        self._init_upper_body()
        self.base_orn_rp = self.get_body_orientation() # [r, p]
        self.com = self.calculate_upper_body_com_local()
        # Initialize last whole body angular momentum
        self.last_whole_body_angular_momentum = torch.zeros((self.num_envs, 3), device=self.device)
        # self.centroidal_momentum = self._compute_centroidal_momentum()
        


    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]


    def _extract_upper_body_rpy(self):
        """
        Extract roll, pitch, and yaw for the pelvis, waist_roll_link, and torso_link individually,
        then sum the absolute values of their RPYs together.

        Returns:
            torch.Tensor: Summed absolute roll, pitch, and yaw values for the upper body.
        """
        upper_body_names = ['pelvis', 'waist_roll_link', 'torso_link']

        # Extract indices for each body part separately
        pelvis_idx = self.body_names.index('pelvis')
        waist_idx = self.body_names.index('waist_roll_link')
        torso_idx = self.body_names.index('torso_link')

        # Extract states for each body part separately
        pelvis_state = self.rigid_body_states_view[:, pelvis_idx, :]
        waist_state = self.rigid_body_states_view[:, waist_idx, :]
        torso_state = self.rigid_body_states_view[:, torso_idx, :]

        # Extract quaternions
        pelvis_quat = pelvis_state[:, 3:7]  # Quaternion for pelvis
        waist_quat = waist_state[:, 3:7]    # Quaternion for waist_roll_link
        torso_quat = torso_state[:, 3:7]    # Quaternion for torso_link

        # Convert quaternions to roll, pitch, yaw
        pelvis_rpy = get_euler_xyz_in_tensor(pelvis_quat)  # Shape: [num_envs, 3]
        waist_rpy = get_euler_xyz_in_tensor(waist_quat)    # Shape: [num_envs, 3]
        torso_rpy = get_euler_xyz_in_tensor(torso_quat)    # Shape: [num_envs, 3]

        # Compute the sum of absolute values
        upper_body_rpy_sum = torch.abs(pelvis_rpy) + torch.abs(waist_rpy) + torch.abs(torso_rpy)




        return upper_body_rpy_sum, pelvis_rpy, waist_rpy, torso_rpy

    def _extract_upper_body_angular_velocity(self):
        """
        Extract angular velocities (roll rate, pitch rate, yaw rate) for the pelvis, 
        waist_roll_link, and torso_link individually.

        Returns:
            tuple: Summed absolute angular velocities and individual angular velocities 
                for pelvis, waist, and torso.
        """
        upper_body_names = ['pelvis', 'waist_roll_link', 'torso_link']

        # Extract indices for each body part separately
        pelvis_idx = self.body_names.index('pelvis')
        waist_idx = self.body_names.index('waist_roll_link')
        torso_idx = self.body_names.index('torso_link')

        # Extract angular velocity states for each body part
        pelvis_state = self.rigid_body_states_view[:, pelvis_idx, :]
        waist_state = self.rigid_body_states_view[:, waist_idx, :]
        torso_state = self.rigid_body_states_view[:, torso_idx, :]

        # Extract angular velocity components (indices 10:13)
        pelvis_ang_vel = pelvis_state[:, 10:13]  # Shape: [num_envs, 3]
        waist_ang_vel = waist_state[:, 10:13]    # Shape: [num_envs, 3]
        torso_ang_vel = torso_state[:, 10:13]    # Shape: [num_envs, 3]

        # Compute the sum of absolute angular velocities
        upper_body_ang_vel_sum = torch.abs(pelvis_ang_vel) + torch.abs(waist_ang_vel) + torch.abs(torso_ang_vel)

        return upper_body_ang_vel_sum, pelvis_ang_vel, waist_ang_vel, torso_ang_vel

    


    def _compute_centroidal_momentum(self):
        """
        Computes the centroidal momentum vector [P, L] using rigid body states and inertia tensors.

        Returns:
            centroidal_momentum (torch.Tensor): Shape [num_envs, 6], containing [Px, Py, Pz, Lx, Ly, Lz]
        """
        num_envs = self.num_envs

        # ✅ Retrieve rigid body states (position, rotation, linear & angular velocity)
        rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(rb_states).view(num_envs, self.num_bodies, 13)

        # ✅ Extract positions, velocities, and angular velocities
        body_positions = rb_states[:, :, 0:3]    # [num_envs, num_bodies, 3]
        body_velocities = rb_states[:, :, 7:10]  # [num_envs, num_bodies, 3]
        body_ang_velocities = rb_states[:, :, 10:13]  # [num_envs, num_bodies, 3]

        # ✅ Retrieve mass of each rigid body
        body_masses = torch.tensor(
            [self.body_masses[name] for name in self.body_names], 
            device=self.device, dtype=torch.float32
        )  # Shape: [num_bodies]

        # ✅ Compute Center of Mass (CoM) position and velocity
        total_mass = body_masses.sum()
        com_position = (body_positions * body_masses[None, :, None]).sum(dim=1) / total_mass
        com_velocity = (body_velocities * body_masses[None, :, None]).sum(dim=1) / total_mass

        # ✅ Compute Linear Momentum (P)
        linear_momentum = total_mass * com_velocity  # [num_envs, 3]

        # ✅ Retrieve Rigid Body Inertia Tensors (Assuming Diagonal Inertia)
        rb_inertias = self.gym.acquire_rigid_body_inertia_tensor(self.sim)
        rb_inertias = gymtorch.wrap_tensor(rb_inertias).view(num_envs, self.num_bodies, 3)  
        # Shape: [num_envs, num_bodies, 3] (Ixx, Iyy, Izz)

        # ✅ Retrieve Rotation Matrices to Transform Inertia to World Frame
        rb_rotations = rb_states[:, :, 3:7]  # Quaternion [x, y, z, w]
        rb_rotation_matrices = self._quaternion_to_rotation_matrix(rb_rotations)  # Shape [num_envs, num_bodies, 3, 3]

        # ✅ Convert Diagonal Inertia Tensors to Full World-Frame Inertia Matrices
        I_world = torch.zeros((num_envs, self.num_bodies, 3, 3), device=self.device)
        for i in range(self.num_bodies):
            I_body = torch.diag_embed(rb_inertias[:, i, :])  # Convert diagonal tensor to matrix
            I_world[:, i, :, :] = torch.matmul(
                torch.matmul(rb_rotation_matrices[:, i, :, :], I_body),  # R * I_body
                rb_rotation_matrices[:, i, :, :].transpose(-1, -2)  # * R^T
            )

        # ✅ Compute Angular Momentum (L)
        angular_momentum = torch.zeros_like(linear_momentum)  # Initialize [num_envs, 3]

        for i in range(self.num_bodies):
            r = body_positions[:, i, :] - com_position  # Position relative to CoM
            mass = body_masses[i]

            # ✅ First term: Linear contribution to angular momentum
            angular_momentum += mass * torch.cross(r, body_velocities[:, i, :])

            # ✅ Second term: Rotational inertia contribution (I * ω)
            ang_momentum_i = torch.bmm(I_world[:, i, :, :], body_ang_velocities[:, i, :].unsqueeze(-1)).squeeze(-1)
            angular_momentum += ang_momentum_i  # Add inertia contribution

        # ✅ Combine linear and angular momentum
        centroidal_momentum = torch.cat([linear_momentum, angular_momentum], dim=1)  # Shape: [num_envs, 6]

        return centroidal_momentum

    def _compute_centroidal_momentum_rate(self):
        """
        Computes the rate of change of angular momentum (dot{L}) using:
            dot{L} = H(q) q̈ + dot{H} q̇
        
        Returns:
            angular_momentum_rate (torch.Tensor): Shape [num_envs, 3] -> (L̇x, L̇y, L̇z)
        """
        centroidal_momentum = self._compute_centroidal_momentum()
        angular_momentum = centroidal_momentum[:, 3:6]  # Extract Lx, Ly, Lz

        if hasattr(self, "last_angular_momentum"):
            angular_momentum_rate = (angular_momentum - self.last_angular_momentum) / self.dt
        else:
            angular_momentum_rate = torch.zeros_like(angular_momentum)

        self.last_angular_momentum = angular_momentum.clone().detach()

        return angular_momentum_rate


    def _init_upper_body(self):
        """
        Initialize upper body roll, pitch, and yaw.
        """
        upper_body_rpy, pelvis_rpy, waist_rpy, torso_rpy = self._extract_upper_body_rpy()
        self.upper_roll = upper_body_rpy[:,0]
        self.upper_pitch = upper_body_rpy[:,1]
        self.upper_yaw = upper_body_rpy[:,2]
        self.pelvis_roll = pelvis_rpy[:,0]
        self.waist_roll = waist_rpy[:,0]
        self.torso_roll = torso_rpy[:,0]
        self.pelvis_pitch = pelvis_rpy[:,1]
        self.waist_pitch = waist_rpy[:,1]
        self.torso_pitch = torso_rpy[:,1]

    def update_body_state(self):
        """
        Refresh the rigid body states and update the upper body roll, pitch, and yaw.
        """
        # Refresh the tensor to get the latest simulation state
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Update upper body RPY
        upper_body_rpy, pelvis_rpy, waist_rpy, torso_rpy = self._extract_upper_body_rpy()
        
        self.upper_roll = upper_body_rpy[:,0]
        self.upper_pitch = upper_body_rpy[:,1]
        self.upper_yaw = upper_body_rpy[:,2]
        self.pelvis_roll = pelvis_rpy[:,0]
        self.waist_roll = waist_rpy[:,0]
        self.torso_roll = torso_rpy[:,0]
        self.pelvis_pitch = pelvis_rpy[:,1]
        self.waist_pitch = waist_rpy[:,1]
        self.torso_pitch = torso_rpy[:,1]


    def calculate_center_of_mass(self):
        """
        Calculate the Center of Mass (CoM) for each robot in all environments.

        Returns:
            torch.Tensor: The CoM position for each environment (shape: [num_envs, 3]).
        """
        num_envs = self.num_envs
        num_bodies = self.num_bodies

        # Get rigid body states (shape: [num_envs, num_bodies, 13])
        rigid_body_states = self.rigid_body_states  # [num_envs, num_bodies, 13]
        rigid_body_states = rigid_body_states.view(self.num_envs,num_bodies, 13)

        # Extract body positions (only first 3 elements are [x, y, z] positions)
        body_positions = rigid_body_states[:, :, :3]  # Shape: [num_envs, num_bodies, 3]

        # Retrieve mass of each body
        body_masses = []
        for env_id in range(num_envs):
            actor_handle = self.actor_handles[env_id]
            body_props = self.gym.get_actor_rigid_body_properties(self.envs[env_id], actor_handle)
            mass_list = [prop.mass for prop in body_props]  # Get mass of each rigid body
            body_masses.append(mass_list)

        # Convert mass list to tensor (Shape: [num_envs, num_bodies])
        body_masses = torch.tensor(body_masses, device=self.device)

        # Compute total mass for each environment (Shape: [num_envs, 1])
        total_mass = torch.sum(body_masses, dim=1, keepdim=True)  # Shape: [num_envs, 1]

        # Compute weighted sum of positions (Shape: [num_envs, 3])
        com = torch.sum(body_positions * body_masses.unsqueeze(-1), dim=1) / total_mass

        return com  # Shape: [num_envs, 3]


    def calculate_upper_body_com(self):
        """
        Calculate the Center of Mass (CoM) for the upper body only.

        Returns:
            torch.Tensor: The CoM position for each environment (shape: [num_envs, 3]).
        """
        upper_body_names = ['pelvis', 'waist_roll_link', 'torso_link']

        # Get indices of upper body parts
        upper_body_indices = [self.body_names.index(name) for name in upper_body_names]

        # Extract states for upper body parts
        upper_body_states = self.rigid_body_states_view[:, upper_body_indices, :]
        upper_body_positions = upper_body_states[:, :, :3]  # Extract [x, y, z] positions

        # Retrieve mass properties for upper body parts
        body_masses = torch.tensor([
        [self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id])[idx].mass 
         for idx in upper_body_indices]  # Get masses using indices
        for env_id in range(self.num_envs)
    ], dtype=torch.float, device=self.device)  # Shape: [num_envs, num_upper_bodies]

        # Compute total mass for the upper body
        total_mass = torch.sum(body_masses, dim=1, keepdim=True)  # Shape: [num_envs, 1]

        # Prevent division by zero
        total_mass = torch.clamp(total_mass, min=1e-6)

        # Compute weighted sum of positions for upper body CoM
        upper_body_com = torch.sum(upper_body_positions * body_masses.unsqueeze(-1), dim=1) / total_mass

        return upper_body_com  # Shape: [num_envs, 3]

    def calculate_upper_body_com_local(self):
        """
        Calculate the Center of Mass (CoM) for the upper body relative to the pelvis (local frame).

        Returns:
            torch.Tensor: The upper body CoM relative to the pelvis for each environment (shape: [num_envs, 3]).
        """
        upper_body_names = ['pelvis', 'waist_roll_link', 'torso_link']

        # Get indices of upper body parts
        upper_body_indices = [self.body_names.index(name) for name in upper_body_names]
        pelvis_idx = self.body_names.index('pelvis')

        # Extract positions for upper body parts
        upper_body_states = self.rigid_body_states_view[:, upper_body_indices, :]
        upper_body_positions = upper_body_states[:, :, :3]  # Extract [x, y, z] positions

        # Retrieve mass properties for upper body parts by index
        body_masses = torch.tensor([
            [self.gym.get_actor_rigid_body_properties(self.envs[env_id], self.actor_handles[env_id])[idx].mass 
            for idx in upper_body_indices]
            for env_id in range(self.num_envs)
        ], dtype=torch.float, device=self.device)  # Shape: [num_envs, num_upper_bodies]

        # Compute total mass for the upper body
        total_mass = torch.sum(body_masses, dim=1, keepdim=True)  # Shape: [num_envs, 1]

        # Prevent division by zero
        total_mass = torch.clamp(total_mass, min=1e-6)

        # Compute global CoM of upper body
        upper_body_com_global = torch.sum(upper_body_positions * body_masses.unsqueeze(-1), dim=1) / total_mass  # Shape: [num_envs, 3]

        # Extract pelvis position (reference frame)
        pelvis_position = self.rigid_body_states_view[:, pelvis_idx, :3]  # Shape: [num_envs, 3]

        # Compute local CoM relative to pelvis
        upper_body_com_local = upper_body_com_global - pelvis_position  # Shape: [num_envs, 3]

        return upper_body_com_local


    def calculate_upper_body_com_local_velocity(self):
        """
        Calculate the velocity of the Center of Mass (CoM) for the upper body relative to the whole body,
        normalized by the timestep (dt).
        
        Args:
            dt (float): Time interval between steps (default: 1.0).
        
        Returns:
            torch.Tensor: The velocity of the upper body CoM relative to the whole body CoM 
                        for each environment (shape: [num_envs, 3]).
        """
        upper_body_com_local = self.calculate_upper_body_com_local()  # Current CoM
        if not hasattr(self, "prev_upper_body_com_local"):
            self.prev_upper_body_com_local = torch.zeros_like(upper_body_com_local)
        upper_body_com_velocity = (upper_body_com_local - self.prev_upper_body_com_local) / self.dt
        self.prev_upper_body_com_local = upper_body_com_local.clone()
        return upper_body_com_velocity
        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        self.update_body_state()
        self.com = self.calculate_upper_body_com_local()
        # self.centroidal_momentum = self._compute_centroidal_momentum()



        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        contact_states = torch.norm(self.sensor_forces[:, :, :2], dim=2) > 1.
        contact_forces = self.sensor_forces.flatten(1)
        contact_normals = self.contact_normal
        base_lin_vel = self.base_lin_vel
        if self.friction_coeffs is not None:
            friction_coefficients = self.friction_coeffs.squeeze(1).repeat(
                1, 2).to(self.device)
        else:
            friction_coefficients = torch.tensor(
                self.cfg.terrain.static_friction).repeat(self.num_envs,
                                                         2).to(self.device)
            
            
        if self.restitution_coeffs is not None:
            restitution_coefficients = self.restitution_coeffs.squeeze(1).repeat(
                1, 2).to(self.device)
        else:
            restitution_coefficients = torch.tensor(
                self.cfg.terrain.restitution).repeat(self.num_envs,
                                                         2).to(self.device)
        hip_and_knee_contact = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :],
            dim=-1) > 0.1
        external_forces_and_torques = torch.cat(
            (self.push_forces[:, 0, :], self.push_torques[:, 0, :]), dim=-1)
        external_forces = self.push_forces[:, 0, :]
        external_position = self.push_positions[:, 0, :]
        airtime = self.feet_air_time
        
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)

        # self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands[:, :3] * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions,
        #                             sin_phase,
        #                             cos_phase
        #                             ),dim=-1)
        # self.privileged_obs_buf = torch.cat((  
        #                             contact_states * self.priv_obs_scales.contact_state,#2
        #                             contact_forces * self.priv_obs_scales.contact_force,#6
        #                             friction_coefficients * self.priv_obs_scales.friction,#4
        #                             restitution_coefficients * self.priv_obs_scales.restitution,#4
        #                             hip_and_knee_contact *#8
        #                             self.priv_obs_scales.thigh_and_shank_contact_state,
        #                             external_forces *#3
        #                             self.priv_obs_scales.external_wrench,
        #                             external_position#3
        #                             ),dim=-1)
        
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel, #dim = 3
                                    self.projected_gravity,#dim = 3
                                    self.commands[:, :3] * self.commands_scale,#dim = 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #dim = num_action/dof
                                    self.dof_vel * self.obs_scales.dof_vel,#dim = num_action/dof
                                    self.actions,#dim = num_action/dof
                                    sin_phase,#dim = 1
                                    cos_phase#dim = 1
                                    ),dim=-1)

        # print('self.base_ang_vel.shape:',self.base_ang_vel.shape)
        # print('self.projected_gravity.shape:',self.projected_gravity.shape)
        # print('self.commands[:, :3].shape:',self.commands[:, :3].shape)
        # print('self.dof_pos.shape:',self.dof_pos.shape)
        # print('self.dof_vel.shape:',self.dof_vel.shape)
        # print('self.actions.shape:',self.actions.shape)
        # print('sin_phase.shape:',sin_phase.shape)
        # print('cos_phase.shape:',cos_phase.shape)

        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        privileged_obs = torch.cat((contact_states * self.priv_obs_scales.contact_state,#2
                                    contact_forces * self.priv_obs_scales.contact_force,#6
                                    #  contact_normals * self.priv_obs_scales.contact_normal,
                                    friction_coefficients * self.priv_obs_scales.friction,#2
                                    restitution_coefficients * self.priv_obs_scales.restitution,#4
                                    hip_and_knee_contact *#12
                                    self.priv_obs_scales.thigh_and_shank_contact_state,
                                    external_forces *#3
                                    self.priv_obs_scales.external_wrench,
                                    external_position),dim = -1)
        # print('contact_states * self.priv_obs_scales.contact_state:',contact_states.shape)
        # print('contact_forces',contact_forces.shape)
        # print('friction_coefficients',friction_coefficients.shape)
        # print('restitution_coefficients',restitution_coefficients.shape)
        # print('hip_and_knee_contact',hip_and_knee_contact.shape)
        # print('external_forces',external_forces.shape)
        # print('external_position',external_position.shape)
        # print('self.privileged_obs_buf',self.privileged_obs_buf.shape)





        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # self.obs_buf = torch.cat((self.obs_buf, privileged_obs), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, privileged_obs), dim=-1)
        

        # self.obs_buf = torch.cat((self.obs_buf, self.privileged_obs_buf), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights_in_sim:
            base_height = self.root_states[:, 2].unsqueeze(1) - self.measured_heights
            # base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
            terrain_obs_buf = torch.clip(
                        base_height -
                        self.cfg.rewards.base_height_target, -1,
                        1.) * self.obs_scales.height_measurements   
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, terrain_obs_buf), dim=-1)
            # self.obs_buf = torch.cat((self.obs_buf, terrain_obs_buf), dim=-1)
            # print('terrain_obs_buf:',terrain_obs_buf.shape)


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        r, p = self.base_orn_rp[:, 0], self.base_orn_rp[:, 1]
        
        z = self.root_states[:, 2]

        r_threshold_buff = r.abs() > self.cfg.termination.r_threshold
        p_threshold_buff = p.abs() > self.cfg.termination.p_threshold
        z_threshold_buff = z < self.cfg.termination.z_threshold
        self.reset_buf |= r_threshold_buff
        self.reset_buf |= p_threshold_buff
        self.reset_buf |= z_threshold_buff

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf




    #------------ reward functions----------------
    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        if return_yaw:
            return torch.stack([r, p, y], dim=-1)
        else:
            return torch.stack([r, p], dim=-1)

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0


    # ----------------------------------------------
    # ✅ 1. Contact & Feet-related Rewards
    # ----------------------------------------------         

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))

    def _reward_feet_drag(self):
        # Determine the size of rigid body states
        num_envs = self.num_envs
        num_bodies = self.num_bodies
        state_size = self.rigid_body_states.shape[1]

        # Reshape rigid_body_states to [num_envs, num_bodies, state_size]
        rigid_body_states = self.rigid_body_states.view(num_envs, num_bodies, state_size)

        # Compute the feet velocity
        feet_xyz_vel = torch.abs(rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)

        # Filter velocities based on contact state
        dragging_vel = self.contact_filt * feet_xyz_vel

        # Compute the total reward/penalty for dragging
        rew = dragging_vel.sum(dim=-1)

        return rew

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)

    def _reward_penalty_feet_slippage(self):
        """
        Penalizes foot slippage when the feet are in contact with the ground.
        """
        # ✅ Extract foot velocities (linear velocity in XYZ)
        foot_vel = self.rigid_body_states_view[:, self.feet_indices, 7:10]  # Shape: [num_envs, num_feet, 3]

        # ✅ Get contact forces (Z-direction force to check ground contact)
        foot_contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0  # Shape: [num_envs, num_feet]

        # ✅ Compute penalty (penalize velocity when contact force is present)
        penalty = torch.sum(torch.norm(foot_vel, dim=-1) * foot_contact, dim=1)  # Sum over feet

        return penalty

    # ----------------------------------------------
    # ✅ 2. Posture & Stability Rewards
    # ----------------------------------------------



    # def _reward_center_of_mass_stability(self):
    #     """
    #     Calculate a reward for maintaining CoM stability in the X and Y directions.

    #     Args:
    #         weight (float): Weighting factor for the penalty.

    #     Returns:
    #         torch.Tensor: The reward value.
    #     """
    #     # Compute current Center of Mass
    #     self.com = self.calculate_upper_body_com_local()  # Shape: [num_envs, 3]

    #     # Desired CoM in X and Y (keep Z free)
    #     desired_com = torch.zeros_like(self.com)  # Default target at [0, 0, free]
    #     desired_com[:, 2] = self.com[:, 2]  # Keep Z unchanged

        
    #     rew = torch.exp(-torch.norm(self.com - desired_com, dim=1))

    #     # Reward is the negative penalty, scaled by weight
    #     return rew   # Higher reward for lower deviation

    def _reward_minimize_com_velocity(self):
        """
        Reward to penalize excessive upper body CoM velocity to reduce oscillations.

        Returns:
            torch.Tensor: Reward values for minimizing CoM velocity.
        """
        # Compute the velocity of the upper body CoM
        com_velocity = self.calculate_upper_body_com_local_velocity()  # Shape: [num_envs, 3]

        # Compute the magnitude of the velocity
        velocity_magnitude = torch.norm(com_velocity, dim=1)  # Shape: [num_envs]

        # Reward is inversely proportional to the velocity magnitude
        reward = torch.exp(-velocity_magnitude)  # Higher reward for lower velocity
        return reward

    def _reward_upper_body_roll(self,roll_weight=1.0):
        """
        Compute a reward for stability based on roll, pitch, and yaw deviations.

        Args:
            roll (float): Roll angle (radians).
            pitch (float): Pitch angle (radians).
            yaw (float): Yaw angle (radians).
            roll_weight (float): Weighting factor for roll stability.
            pitch_weight (float): Weighting factor for pitch stability.
            yaw_weight (float): Weighting factor for yaw stability.

        Returns:
            float: Stability reward (higher is better).
        """
        num_envs = self.num_envs
        num_body_parts = len(self.upper_body_index)  # Assume upper_body_index has the correct indices for upper body parts.

        # Compute stability penalty (squared deviation)


        upper_roll_error = torch.square(self.upper_roll)
        
        # Reshape to [num_envs, num_body_parts]
        upper_roll_error = upper_roll_error.view(num_envs, num_body_parts)

        # Debug: Check the reshaped tensor

        # Sum across body parts (dim=1) to get the penalty for each environment
        upper_roll_error = torch.sum(upper_roll_error, dim=1)

        # Return negative stability penalty as the reward
        return -upper_roll_error

    def _reward_upper_body_pitch(self,roll_weight=1.0):
        """
        Compute a reward for stability based on roll, pitch, and yaw deviations.

        Args:
            roll (float): Roll angle (radians).
            pitch (float): Pitch angle (radians).
            yaw (float): Yaw angle (radians).
            roll_weight (float): Weighting factor for roll stability.
            pitch_weight (float): Weighting factor for pitch stability.
            yaw_weight (float): Weighting factor for yaw stability.

        Returns:
            float: Stability reward (higher is better).
        """
        num_envs = self.num_envs
        num_body_parts = len(self.upper_body_index)  # Assume upper_body_index has the correct indices for upper body parts.

        # Compute stability penalty (squared deviation)


        upper_pitch_error = torch.square(self.upper_roll)
        
        # Reshape to [num_envs, num_body_parts]
        upper_pitch_error = upper_pitch_error.view(num_envs, num_body_parts)

        # Debug: Check the reshaped tensor

        # Sum across body parts (dim=1) to get the penalty for each environment
        upper_pitch_error = torch.sum(upper_pitch_error, dim=1)

        # Return negative stability penalty as the reward
        return -upper_pitch_error
    
    def _reward_rpy(self):
        
        return -torch.sum(torch.abs(self.rpy[:,0:2]),dim=-1)



    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)

    

    def _reward_upper_body(self):
        # Indices for knee joints (left and right knees)
        upper_indices = [12, 13]  # Indices for left_knee_link and right_knee_link
        # Penalize deviation from straight knee position, only when the foot is in contact
        # self.contact_filt should correspond to the contact state of the feet (same order as knee indices)
        
        upper_error = torch.square(self.dof_pos[:, upper_indices]) 
        # Return negative reward for deviation (penalty)
        return -torch.sum(upper_error, dim=1)


    def _reward_stabilize_waist_yaw(self):
        """
        Computes the reward for stabilizing the waist yaw joint by penalizing
        excessive yaw velocity and acceleration with an exponential penalty.

        Returns:
            reward (torch.Tensor): Shape [num_envs], reward values for each environment.
        """
        # Define reward weights
        w1, w2 = 0.1, 0.0003  # Tune these values as needed
        self.waist_yaw_idx = self.dof_names.index('waist_yaw_joint')
        # ✅ Extract waist yaw joint velocity
        waist_yaw_velocity = self.dof_vel[:, self.waist_yaw_idx]

        # ✅ Exponential penalty for excessive yaw movement
        r_yaw_stability = - (torch.square(waist_yaw_velocity))


        # ✅ Exponential penalty for sudden yaw acceleration
        r_yaw_smoothness = - torch.square((self.last_dof_vel[:, self.waist_yaw_idx] - self.dof_vel[:, self.waist_yaw_idx]) / self.dt)
        # ✅ Compute final reward
        reward = w1 * r_yaw_stability + w2 * r_yaw_smoothness


        return reward




    def _reward_minimize_waist_pitch_deviation(self, target_pitch=0.0, weight=1.0, log=False):
        """
        Penalize deviation of waist_pitch_joint from a target angle.

        Args:
            target_pitch (float): Desired waist pitch angle in radians (default: 0.0).
            weight (float): Scaling factor for the reward (default: 1.0).
            log (bool): Whether to log the deviation (default: False).

        Returns:
            torch.Tensor: Reward value for minimizing waist pitch deviation.
        """
        self.waist_pitch_index = self.dof_names.index('waist_pitch_joint')  # Rotates around Y-axis

        # Extract waist pitch joint angle
        waist_pitch_angle = self.dof_pos[:, self.waist_pitch_index]  # [num_envs]

        # Compute penalty for deviation from target pitch
        deviation_penalty = torch.abs(waist_pitch_angle - target_pitch)  # Absolute deviation

        # Log deviation for debugging
        if log:
            print(f"Waist Pitch Deviation: {torch.mean(deviation_penalty).item()}")

        # Compute final reward (exponential penalty with weight)
        reward = weight * torch.exp(-2.0 * deviation_penalty)

        return reward

    def _reward_tracking_pelvis_roll(self):
        
        demo_roll = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.pelvis_roll - demo_roll, dim=1))
        return rew
    def _reward_tracking_torso_roll(self):
        
        demo_roll = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.torso_roll - demo_roll, dim=1))
        return rew
    def _reward_tracking_waist_roll(self):
        
        demo_roll = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.waist_roll - demo_roll, dim=1))
        return rew
    

    def _reward_tracking_pelvis_pitch(self):
        
        demo_pitch = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.pelvis_pitch - demo_pitch, dim=1))
        return rew

    def _reward_tracking_torso_pitch(self):
        
        demo_pitch = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.torso_pitch - demo_pitch, dim=1))
        return rew
    def _reward_tracking_waist_pitch(self):

        demo_pitch = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.waist_pitch - demo_pitch, dim=1))
        return rew

    def _reward_tracking_pitch(self):
        demo_pitch = torch.zeros(self.num_envs, 1, device = self.device)
        rew = torch.exp(-torch.norm(self.upper_pitch - demo_pitch, dim=1))
        return rew

    def _reward_tracking_roll_pitch(self):
        cur_roll_pitch = torch.stack((self.upper_roll, self.upper_pitch), dim=1)
        demo_roll_pitch = torch.zeros(self.num_envs, 2, device = self.device)
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
        return rew

    def _reward_minimize_torso_angular_velocity(self):
        """
        Penalize large angular velocities in the upper body to stabilize motion.

        Returns:
            torch.Tensor: The reward value for minimizing angular velocity.
        """
        _, pelvis_ang_vel, waist_ang_vel, torso_ang_vel = self._extract_upper_body_angular_velocity()


        # Compute penalty for angular velocity magnitude
        angular_velocity_penalty = torch.norm(torso_ang_vel, dim=1)  # Shape: [num_envs]

        # Reward is inversely proportional to the penalty
        reward = torch.exp(-angular_velocity_penalty)  # Penalize high angular velocity
        return reward


    def _reward_penalty_ang_vel_xy_torso(self):
        """
        Penalize high angular velocity in the XY plane for the torso.
        This helps stabilize the upper body by reducing excessive rotation.
        """

        # ✅ Extract torso quaternion & angular velocity
        _, _, _, torso_rpy = self._extract_upper_body_rpy()

        # ✅ Get torso angular velocity (already in world frame)
        torso_ang_vel = self.rigid_body_states_view[:, self.body_names.index('torso_link'), 10:13]  # Shape: [num_envs, 3]

        # ✅ Transform angular velocity to torso frame
        torso_ang_vel_local = quat_rotate_inverse(torso_rpy, torso_ang_vel)

        # ✅ Penalize XY angular velocity
        penalty = torch.sum(torch.square(torso_ang_vel_local[:, :2]), dim=1)  # Penalize X & Y, ignore Z

        return penalty


    # ----------------------------------------------
    # ✅ 5. Gait-related Rewards
    # ----------------------------------------------
    def _reward_straight_knee(self):
        # Indices for knee joints (left and right knees)
        self.left_knee_index = self.dof_names.index('left_knee_joint')  
        self.right_knee_index = self.dof_names.index('right_knee_joint')  

        knee_indices = [self.left_knee_index, self.right_knee_index]  # Indices for left_knee_link and right_knee_link
        # Penalize deviation from straight knee position, only when the foot is in contact
        # self.contact_filt should correspond to the contact state of the feet (same order as knee indices)
        straight_knee_error = torch.square(self.dof_pos[:, knee_indices]) * self.contact_filt[:, :len(knee_indices)]
        # Return negative reward for deviation (penalty)
        return -torch.sum(straight_knee_error, dim=1)


    def _reward_stable_standing(self):
        """
        Encourage the robot to remain stable when given a zero velocity command.
        Penalizes unnecessary foot movement while standing still.
        """

        # ✅ Retrieve foot velocities (XY-plane)
        left_foot_velocity = torch.norm(self.rigid_body_states_view[:, self.feet_indices[0], 7:9], dim=-1)
        right_foot_velocity = torch.norm(self.rigid_body_states_view[:, self.feet_indices[1], 7:9], dim=-1)

        # ✅ Retrieve commanded velocity (should be zero if standing still)
        commanded_lin_vel = torch.norm(self.commands[:, :2], dim=-1)  # XY velocity

        # ✅ Compute reward for minimizing foot movement when standing still
        standing_reward = torch.exp(-10.0 * (left_foot_velocity + right_foot_velocity))  # Penalize foot movement

        # ✅ Reduce reward if commanded velocity is nonzero (don't penalize walking)
        standing_reward *= (commanded_lin_vel < 0.05).float()

        return standing_reward



    def _reward_heel_toe_walking(self):
        """
        Encourage heel-strike and toe-off walking behavior.
        Penalizes flat-footed landing or incorrect foot placement.
        """

        # ✅ Get foot quaternions (orientation)
        left_foot_quat = self.rigid_body_states_view[:, self.feet_indices[0], 3:7]  # Left foot orientation (quaternion)
        right_foot_quat = self.rigid_body_states_view[:, self.feet_indices[1], 3:7]  # Right foot orientation (quaternion)

        # ✅ Compute foot orientation relative to gravity
        left_foot_up = quat_rotate_inverse(left_foot_quat, self.gravity_vec)  # Maps quaternion to gravity frame
        right_foot_up = quat_rotate_inverse(right_foot_quat, self.gravity_vec)

        # ✅ Extract roll and pitch angles (important for detecting heel-strike and toe-off)
        left_foot_pitch = torch.atan2(left_foot_up[:, 1], left_foot_up[:, 2])
        right_foot_pitch = torch.atan2(right_foot_up[:, 1], right_foot_up[:, 2])

        # ✅ Compute foot contact forces (check if feet are on the ground)
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0

        # ✅ Heel-strike reward (encourage negative pitch at initial contact)
        left_heel_strike = left_contact * torch.clamp(left_foot_pitch, max=0)  # Encourage negative pitch (heel down)
        right_heel_strike = right_contact * torch.clamp(right_foot_pitch, max=0)

        # ✅ Toe-off reward (encourage positive pitch before lifting)
        left_toe_off = ~left_contact * torch.clamp(left_foot_pitch, min=0)  # Encourage positive pitch (toe up)
        right_toe_off = ~right_contact * torch.clamp(right_foot_pitch, min=0)

        # ✅ Combine both rewards
        reward = left_heel_strike + right_heel_strike + left_toe_off + right_toe_off

        return torch.exp(reward)  # Reward is higher when correct heel-toe behavior occurs

    

    # ----------------------------------------------
    # ✅ 4. Arm-related Rewards
    # ----------------------------------------------
    
    def _reward_minimize_arm_torque(self):
        """
        Penalize excessive torque applied to shoulder and elbow joints.
        """

        self.left_shoulder_pitch_idx = self.dof_names.index('left_shoulder_pitch_joint')
        self.right_shoulder_pitch_idx = self.dof_names.index('right_shoulder_pitch_joint')
        # ✅ Extract torque values
        left_shoulder_torque = torch.abs(self.torques[:, self.left_shoulder_pitch_idx])
        right_shoulder_torque = torch.abs(self.torques[:, self.right_shoulder_pitch_idx])
        left_elbow_torque = torch.zeros_like(left_shoulder_torque)
        right_elbow_torque = torch.zeros_like(right_shoulder_torque)
        # left_elbow_torque = torch.abs(self.torques[:, self.left_elbow_idx])
        # right_elbow_torque = torch.abs(self.torques[:, self.right_elbow_idx])

        # ✅ Compute total arm torque (sum of absolute values)
        total_arm_torque = (left_shoulder_torque + right_shoulder_torque + left_elbow_torque + right_elbow_torque)

        # ✅ Normalize for stability
        total_arm_torque = total_arm_torque / self.num_envs

        # ✅ Convert to reward (higher reward for lower torque)
        return torch.exp(-0.5 * total_arm_torque)  # Scale for better gradient response
    

    def _reward_minimize_whole_body_angular_momentum(self):
        """
        Reward function to minimize whole-body angular momentum, encouraging natural arm swing.
        """

        # ✅ Compute centroidal angular momentum L
        centroidal_momentum = self._compute_centroidal_momentum()
        whole_body_angular_momentum = centroidal_momentum[:, 3:6]  # Extract Lx, Ly, Lz

        # ✅ Penalize large whole-body angular momentum
        reward = -torch.norm(whole_body_angular_momentum, dim=1)

        return torch.exp(reward)  # Higher reward when L is small



    def _reward_minimize_angular_momentum_rate(self):
        """
        Reward function to encourage smoother arm movements by minimizing the rate of change of angular momentum (dot{L}).
        """

        # ✅ Compute the rate of change of angular momentum (L̇)
        angular_momentum_rate = self._compute_centroidal_momentum_rate()  # Shape: [num_envs, 3]

        # ✅ Penalize large changes in angular momentum
        reward = -torch.norm(angular_momentum_rate, dim=1)  # Minimize sudden momentum changes

        return torch.exp(reward)  # Higher reward when L̇ is small

    

    

    def _reward_arm_leg_coordination(self):
        """
        Reward function to encourage inverse-phase movement of arms and legs.
        """
        # Extract velocities of hip and shoulder
        self.left_hip_pitch_idx = self.dof_names.index('left_hip_pitch_joint')
        self.right_hip_pitch_idx = self.dof_names.index('right_hip_pitch_joint')
        self.left_shoulder_pitch_idx = self.dof_names.index('left_shoulder_pitch_joint')
        self.right_shoulder_pitch_idx = self.dof_names.index('right_shoulder_pitch_joint')



        left_leg_vel = self.dof_vel[:, self.left_hip_pitch_idx]  # Left hip pitch velocity
        right_leg_vel = self.dof_vel[:, self.right_hip_pitch_idx]  # Right hip pitch velocity

        left_arm_vel = self.dof_vel[:, self.left_shoulder_pitch_idx]  # Left shoulder pitch velocity
        right_arm_vel = self.dof_vel[:, self.right_shoulder_pitch_idx]  # Right shoulder pitch velocity

        # Compute correlation (negative means opposite movement)
        coordination = -(left_leg_vel * left_arm_vel + right_leg_vel * right_arm_vel)

        return torch.exp(coordination)  # Higher reward when arms and legs are in opposite phase
    
    
    
    
    


    