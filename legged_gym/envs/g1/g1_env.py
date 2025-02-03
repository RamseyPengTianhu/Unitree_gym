
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, euler_from_quat
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
import os


from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR


import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import torch
# from motion_lib import MotionLib
from ASE.ase.utils.motion_lib import MotionLib



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
        self.com = self.calculate_center_of_mass()
        


    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]


    def _extract_upper_body_rpy(self):
        """
        Extract roll, pitch, and yaw for the upper body from the rigid body states.

        Returns:
            tuple: Roll, pitch, and yaw as NumPy floats.
        """
        # self.upper_body_index = [13,14]
        upper_body_names = ['pelvis', 'waist_roll_link','torso_link']
        # upper_body_names = [ 'torso_link']
        self.upper_body_index = [self.body_names.index(name) for name in upper_body_names]

        # Extract upper body state
        upper_body_state = self.rigid_body_states_view[:, self.upper_body_index, :]
        upper_quaternions = upper_body_state[:, :, 3:7]  # Extract quaternions
        upper_body_rpy = get_euler_xyz_in_tensor(upper_quaternions.view(-1, 4))  # Shape: [num_envs * 3, 3]


        # Convert to NumPy
        return upper_body_rpy

    def _init_upper_body(self):
        """
        Initialize upper body roll, pitch, and yaw.
        """
        upper_body_rpy = self._extract_upper_body_rpy()
        self.upper_roll = upper_body_rpy[:,0]
        self.upper_pitch = upper_body_rpy[:,1]
        self.upper_yaw = upper_body_rpy[:,2]

    def update_body_state(self):
        """
        Refresh the rigid body states and update the upper body roll, pitch, and yaw.
        """
        # Refresh the tensor to get the latest simulation state
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Update upper body RPY
        upper_body_rpy = self._extract_upper_body_rpy()
        
        self.upper_roll = upper_body_rpy[:,0]
        self.upper_pitch = upper_body_rpy[:,1]
        self.upper_yaw = upper_body_rpy[:,2]


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

    def post_physics_step(self):
        # self._motion_sync()
        super().post_physics_step()

        # step motion lib
        self._motion_times += self._motion_dt
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.
        self.update_demo_obs()
        # self.update_mimic_obs()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.draw_rigid_bodies_demo()
            self.draw_rigid_bodies_actual()

        return
        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        self.update_body_state()
        self.com = self.calculate_center_of_mass()


        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        

        # ---Add Motion Tracking---
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % self.cfg.motion.resample_step_inplace_interval == 0:
            self.resample_step_inplace_ids()

        return super()._post_physics_step_callback()
    
    def resample_step_inplace_ids(self, ):
        self.step_inplace_ids = torch.rand(self.num_envs, device=self.device) < self.cfg.motion.step_inplace_prob


    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.motion.resample_step_inplace_interval = np.ceil(self.cfg.motion.resample_step_inplace_interval_s / self.dt)


    def _update_goals(self):
        """
        Updates the humanoid robot's target position and orientation to follow a moving goal.
        The target is reset periodically and updated based on demonstration velocity  Only for nly sets a target position (x, y) and target yaw.
        """

        # Determine if the target position should be reset based on episode progress
        # This resets every `global_keybody_reset_time` seconds
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0

        # If the condition is met, reset target position to current robot position (absolute)
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]

        # Update the target position by integrating the demonstration velocity
        # This ensures the target moves smoothly over time
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]

        # Compute the target position in the robot's local coordinate frame
        # Converts global coordinates (world frame) into a relative position (robot frame)
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])

        # Convert the demonstration's quaternion orientation to Euler angles (roll, pitch, yaw)
        r, p, y = euler_from_quaternion(self._curr_demo_quat)

        # Set the target yaw (rotation around the Z-axis) based on the demonstration
        self.target_yaw = y.clone()

        # Compute desired velocity magnitude from the demonstration (if enabled)
        # This helps determine how fast the humanoid should move




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
        
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
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


    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        if return_yaw:
            return torch.stack([r, p, y], dim=-1)
        else:
            return torch.stack([r, p], dim=-1)

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
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)

    def _reward_straight_knee(self):
        # Indices for knee joints (left and right knees)
        knee_indices = [4, 10]  # Indices for left_knee_link and right_knee_link
        # Penalize deviation from straight knee position, only when the foot is in contact
        # self.contact_filt should correspond to the contact state of the feet (same order as knee indices)
        straight_knee_error = torch.square(self.dof_pos[:, knee_indices]) * self.contact_filt[:, :len(knee_indices)]
        # Return negative reward for deviation (penalty)
        return -torch.sum(straight_knee_error, dim=1)

    def _reward_upper_body(self):
        # Indices for knee joints (left and right knees)
        upper_indices = [12, 13]  # Indices for left_knee_link and right_knee_link
        # Penalize deviation from straight knee position, only when the foot is in contact
        # self.contact_filt should correspond to the contact state of the feet (same order as knee indices)
        
        upper_error = torch.square(self.dof_pos[:, upper_indices]) 
        # Return negative reward for deviation (penalty)
        return -torch.sum(upper_error, dim=1)

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



    def _reward_center_of_mass_stability(self, weight=1.0):
        """
        Calculate a reward for maintaining CoM stability in the X and Y directions.

        Args:
            weight (float): Weighting factor for the penalty.

        Returns:
            torch.Tensor: The reward value.
        """
        # Compute current Center of Mass
        self.com = self.calculate_center_of_mass()  # Shape: [num_envs, 3]

        # Desired CoM in X and Y (keep Z free)
        desired_com = torch.zeros_like(self.com)  # Default target at [0, 0, free]
        desired_com[:, 2] = self.com[:, 2]  # Keep Z unchanged

        # Compute squared distance in X and Y only
        stability_penalty = torch.sum((self.com[:, :2] - desired_com[:, :2]) ** 2, dim=1)

        # Reward is the negative penalty, scaled by weight
        return -weight * stability_penalty  # Higher reward for lower deviation

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew
    



    def init_motions(self, cfg):
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)
        # ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 
        # 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 
        # 'torso_link', 
        # 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_hand_keypoint_link', 
        # 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_hand_keypoint_link']
        self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
                                               6, 9, 10,
                                               12, 15, 16, # Left Shoulder pitch, Elbow, hand
                                               17, 20, 21], device=self.device)
        self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13
        self._dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
                             11, 
                             14, 15, 16, 19, 20, 21]  # 14
        self._valid_dof_body_ids = torch.ones(len(self._dof_body_ids)+2*4, device=self.device, dtype=torch.bool)
        self._valid_dof_body_ids[-1] = 0
        self._valid_dof_body_ids[-6] = 0
        self.dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], device=self.device, dtype=torch.long)
        
        # self._dof_ids_subset = torch.tensor([0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
        self._dof_ids_subset = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device)  # no knee and ankle
        self._n_demo_dof = len(self._dof_ids_subset)

        #['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 
        #'left_knee_joint', 'left_ankle_joint', 
        #'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 
        #'right_knee_joint', 'right_ankle_joint', 
        #'torso_joint', 
        #'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        #'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
        # self.dof_ids_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], device=self.device, dtype=torch.long)
        # motion_name = "17_04_stealth"



                # Define base directory for motion reference data
        MOTION_BASE_DIR = "/home/tianhu/unitree_rl_gym/reference_data"

        # Specify motion file name from configuration
        cfg.motion.motion_name = "02_01.npy"

        # Construct the full motion file path
        motion_file = os.path.join(MOTION_BASE_DIR, cfg.motion.motion_name)

        # Ensure the motion file exists before proceeding
        if not os.path.exists(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        print(f"Loading motion file: {motion_file}")




        # motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}.npy")
        # if cfg.motion.motion_type == "single":
        #     motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}.npy")
        # else:
        #     assert cfg.motion.motion_type == "yaml"
        #     motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")

        
        self._load_motion(motion_file, cfg.motion.no_keybody)


    def init_motion_buffers(self, cfg):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        if cfg.motion.motion_curriculum:
            self._max_motion_difficulty = 9
            # self._motion_ids = self._motion_lib.sample_motions(self.num_envs, self._max_motion_difficulty)
        else:
            self._max_motion_difficulty = 9
        self._motion_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)
        # self._motion_features = self._motion_lib.get_motion_features(self._motion_ids)

        self._motion_dt = self.dt
        self._motion_num_future_steps = self.cfg.env.n_demo_steps
        self._motion_demo_offsets = torch.arange(0, self.cfg.env.n_demo_steps * self.cfg.env.interval_demo_steps, self.cfg.env.interval_demo_steps, device=self.device)
        self._demo_obs_buf = torch.zeros((self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo), device=self.device)
        self._curr_demo_obs_buf = self._demo_obs_buf[:, 0, :]
        self._next_demo_obs_buf = self._demo_obs_buf[:, 1, :]
        # self._curr_mimic_obs_buf = torch.zeros_like(self._curr_demo_obs_buf, device=self.device)

        self._curr_demo_root_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._curr_demo_root_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._curr_demo_keybody = torch.zeros((self.num_envs, self._num_key_bodies, 3), device=self.device)
        self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.dof_term_threshold = 3 * torch.ones(self.num_envs, device=self.device)
        self.keybody_term_threshold = 0.3 * torch.ones(self.num_envs, device=self.device)
        self.yaw_term_threshold = 0.5 * torch.ones(self.num_envs, device=self.device)
        self.height_term_threshold = 0.2 * torch.ones(self.num_envs, device=self.device)

        # self.step_inplace_ids = self.resample_step_inplace_ids()
        

    def _load_motion(self, motion_file, no_keybody=False):
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return

    def update_motion_ids(self, env_ids):
        self._motion_times[env_ids] = self.resample_motion_times(env_ids)
        self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
        self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])

    def resample_motion_times(self, env_ids):
        return self._motion_lib.sample_time(self._motion_ids[env_ids])

    def reset_idx(self, env_ids, init=False):
        """ Reset some environments.
        Includes motion curriculum learning, terrain adaptation, velocity integral reset, 
        physics simulation updates, and buffer clearing.

        Args:
        env_ids (list[int]): List of environment ids to reset.
        init (bool, optional): Whether this is the first reset. Defaults to False.
        """
        if len(env_ids) == 0:
            return

        # --- MOTION CURRICULUM LEARNING ---
        if self.cfg.motion.motion_curriculum:
            completion_rate = self.episode_length_buf[env_ids] * self.dt / self._motion_lengths[env_ids]
            completion_rate_mean = completion_rate.mean()

            # Adjust curriculum difficulty based on completion rates
            relax_ids = completion_rate < 0.3
            strict_ids = completion_rate > 0.9

            self.dof_term_threshold[env_ids[relax_ids]] += 0.05
            self.dof_term_threshold[env_ids[strict_ids]] -= 0.05
            self.dof_term_threshold.clamp_(1.5, 3)

            self.height_term_threshold[env_ids[relax_ids]] += 0.01
            self.height_term_threshold[env_ids[strict_ids]] -= 0.01
            self.height_term_threshold.clamp_(0.03, 0.1)

            self.keybody_term_threshold[env_ids[completion_rate < 0.6]] -= 0.05
            self.keybody_term_threshold[env_ids[completion_rate > 0.9]] += 0.05
            self.keybody_term_threshold.clamp_(0.1, 0.4)

            self.yaw_term_threshold[env_ids[completion_rate < 0.4]] -= 0.05
            self.yaw_term_threshold[env_ids[completion_rate > 0.8]] += 0.05
            self.yaw_term_threshold.clamp_(0.1, 0.6)

        # --- UPDATE MOTION STATES (if using motion imitation) ---
        self.update_motion_ids(env_ids)
        motion_ids = self._motion_ids[env_ids]
        motion_times = self._motion_times[env_ids]
        root_pos, root_rot, dof_pos_motion, root_vel, root_ang_vel, dof_vel, key_pos = \
            self._motion_lib.get_motion_state(motion_ids, motion_times)
        print('dof_pos_motion.shape:',dof_pos_motion.shape)
        # Adjust DOF states based on motion reference
        dof_pos_motion, dof_vel = self.reindex_dof_pos_vel(dof_pos_motion, dof_vel)

        # --- RESET ROBOT STATES ---
        self._reset_dofs(env_ids, dof_pos_motion, dof_vel)
        self._reset_root_states(env_ids, root_vel, root_rot, root_pos[:, 2])

        # --- UPDATE CURRICULUM (if enabled) ---
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # --- INITIALIZE OR UPDATE ROOT POSITION ---
        if init:
            self.init_root_pos_global = self.root_states[:, :3].clone()
            self.init_root_pos_global_demo = root_pos[:].clone()
            self.target_pos_abs = self.init_root_pos_global.clone()[:, :2]
        else:
            self.init_root_pos_global[env_ids] = self.root_states[env_ids, :3].clone()
            self.init_root_pos_global_demo[env_ids] = root_pos[:].clone()
            self.target_pos_abs[env_ids] = self.init_root_pos_global[env_ids].clone()[:, :2]

        # --- RESAMPLE COMMANDS ---
        self._resample_commands(env_ids)

        # --- SIMULATE ONE STEP TO UPDATE PHYSICS STATE ---
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # --- RESET BUFFERS ---
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0

        # --- RESET VELOCITY INTEGRALS (if tracking velocity) ---
        self.episode_v_integral[env_ids].zero_()
        self.episode_w_integral[env_ids].zero_()

        # --- LOG EPISODE METRICS ---
        self.extras["episode"] = {}
        self.extras["episode"]["curriculum_completion"] = completion_rate_mean

        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        self.episode_length_buf[env_ids] = 0

        self.extras["episode"]["curriculum_motion_difficulty_level"] = self._max_motion_difficulty
        self.extras["episode"]["curriculum_dof_term_thresh"] = self.dof_term_threshold.mean()
        self.extras["episode"]["curriculum_keybody_term_thresh"] = self.keybody_term_threshold.mean()
        self.extras["episode"]["curriculum_yaw_term_thresh"] = self.yaw_term_threshold.mean()
        self.extras["episode"]["curriculum_height_term_thresh"] = self.height_term_threshold.mean()

        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids, dof_pos, dof_vel):
            
        # dof_pos_default = self.default_dof_pos + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device) * self.default_dof_pos
        print('dof_pos.shape:',dof_pos.shape)
        print(' self.dof_pos.shape:', self.dof_pos.shape)
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

            # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(0., 0.5, (len(env_ids), self.num_dof), device=self.device)
            # self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



        


    def _update_goals(self):
        """
        Updates the humanoid robot's target position and orientation to follow a moving goal.
        The target is reset periodically and updated based on demonstration velocity  Only for nly sets a target position (x, y) and target yaw.
        """

        # Determine if the target position should be reset based on episode progress
        # This resets every `global_keybody_reset_time` seconds
        reset_target_pos = self.episode_length_buf % (self.cfg.motion.global_keybody_reset_time // self.dt) == 0

        # If the condition is met, reset target position to current robot position (absolute)
        self.target_pos_abs[reset_target_pos] = self.root_states[reset_target_pos, :2]

        # Update the target position by integrating the demonstration velocity
        # This ensures the target moves smoothly over time
        self.target_pos_abs += (self._curr_demo_root_vel * self.dt)[:, :2]

        # Compute the target position in the robot's local coordinate frame
        # Converts global coordinates (world frame) into a relative position (robot frame)
        self.target_pos_rel = global_to_local_xy(self.yaw[:, None], self.target_pos_abs - self.root_states[:, :2])

        # Convert the demonstration's quaternion orientation to Euler angles (roll, pitch, yaw)
        r, p, y = euler_from_quaternion(self._curr_demo_quat)

        # Set the target yaw (rotation around the Z-axis) based on the demonstration
        self.target_yaw = y.clone()

        # Compute desired velocity magnitude from the demonstration (if enabled)
        # This helps determine how fast the humanoid should move


    def update_demo_obs(self):
        # Compute Motion Time for Demo Retrieval
        demo_motion_times = self._motion_demo_offsets + self._motion_times[:, None]  # [num_envs, demo_dim]
        # Retrieve Motion Data from Motion Library
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, local_key_body_pos \
            = self._motion_lib.get_motion_state(self._motion_ids.repeat_interleave(self._motion_num_future_steps), demo_motion_times.flatten(), get_lbp=True)
        # Adjust Joint Position and Velocity Order
        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
        
        # Store the Retrieved Motion Data
        self._curr_demo_root_pos[:] = root_pos.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        self._curr_demo_quat[:] = root_rot.view(self.num_envs, self._motion_num_future_steps, 4)[:, 0, :]
        self._curr_demo_root_vel[:] = root_vel.view(self.num_envs, self._motion_num_future_steps, 3)[:, 0, :]
        self._curr_demo_keybody[:] = local_key_body_pos[:, self._key_body_ids_sim_subset].view(self.num_envs, self._motion_num_future_steps, self._num_key_bodies, 3)[:, 0, :, :]
        self._in_place_flag = 0*(torch.norm(self._curr_demo_root_vel, dim=-1) < 0.2)
        # for i in range(13):
        #     feet_pos_global = key_pos[:, i]# - root_pos + self.root_states[:, :3]
        #     pose = gymapi.Transform(gymapi.Vec3(feet_pos_global[self.lookat_id, 0], feet_pos_global[self.lookat_id, 1], feet_pos_global[self.lookat_id, 2]), r=None)
        #     gymutil.draw_lines(edge_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        demo_obs = build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos[:, self._dof_ids_subset], dof_vel, key_pos, local_key_body_pos[:, self._key_body_ids_sim_subset, :], self._dof_offsets)
        self._demo_obs_buf[:] = demo_obs.view(self.num_envs, self.cfg.env.n_demo_steps, self.cfg.env.n_demo)[:]

    def compute_obs_buf(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            torch.sin(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            torch.cos(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            # self.target_pos_rel,  
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()*0-0.5),
                            ),dim=-1)

    def compute_obs_demo(self):
        obs_demo = self._next_demo_obs_buf.clone()#self._demo_obs_buf.clone().flatten(start_dim=1)
        obs_demo[self._in_place_flag, self._n_demo_dof:self._n_demo_dof+3] = 0
        return obs_demo

    def _motion_sync(self):
        num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        # print(self._motion_times[self.lookat_id])
        # motion_times = self.episode_length_buf * self._motion_dt

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
           = self._motion_lib.get_motion_state(motion_ids, self._motion_times)
        
        root_pos[:, :2] = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[:, :2]
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:10] = root_vel
        self.root_states[env_ids, 10:13] = root_ang_vel

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        return

    ######### utils #########
    
    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self._valid_dof_body_ids)
        return dof_pos, dof_vel

    def draw_rigid_bodies_demo(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(0, 1, 0))
        local_body_pos = self._curr_demo_keybody.clone().view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        global_body_pos = local_to_global(self._curr_demo_quat, local_body_pos, curr_demo_xyz)
        for i in range(global_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(global_body_pos[self.lookat_id, i, 0], global_body_pos[self.lookat_id, i, 1], global_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def draw_rigid_bodies_actual(self, ):
        geom = gymutil.WireframeSphereGeometry(0.06, 32, 32, None, color=(1, 0, 0))
        rigid_body_pos = self.rigid_body_states[:, self._key_body_ids_sim, :3].clone()
        for i in range(rigid_body_pos.shape[1]):
            pose = gymapi.Transform(gymapi.Vec3(rigid_body_pos[self.lookat_id, i, 0], rigid_body_pos[self.lookat_id, i, 1], rigid_body_pos[self.lookat_id, i, 2]), r=None)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_goals(self, ):
        demo_geom = gymutil.WireframeSphereGeometry(0.2, 32, 32, None, color=(1, 0, 0))
        
        pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
        # print(self._curr_demo_obs_buf[self.lookat_id, 2*self.num_dof:2*self.num_dof+3])
        # demo_pos = (self._curr_demo_root_pos - self.init_root_pos_global_demo + self.init_root_pos_global)[self.lookat_id]
        # pose = gymapi.Transform(gymapi.Vec3(demo_pos[0], demo_pos[1], demo_pos[2]), r=None)
        # gymutil.draw_lines(demo_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        if not self.cfg.depth.use_camera:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            norm = torch.norm(self._curr_demo_root_vel[:, :2], dim=-1, keepdim=True)
            target_vec_norm = self._curr_demo_root_vel[:, :2] / (norm + 1e-5)
            for i in range(5):
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
    
    def _reward_tracking_demo_goal_vel(self):
        norm = torch.norm(self._curr_demo_root_vel[:, :3], dim=-1, keepdim=True)
        target_vec_norm = self._curr_demo_root_vel[:, :3] / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:10]
        norm_squeeze = norm.squeeze(-1)
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), norm_squeeze) / (norm_squeeze + 1e-5)

        rew_zeros = torch.exp(-4*torch.norm(cur_vel, dim=-1))
        small_cmd_ids = (norm<0.1).squeeze(-1)
        rew[small_cmd_ids] = rew_zeros[small_cmd_ids]
        # return torch.exp(-2 * torch.norm(cur_vel - self._curr_demo_root_vel[:, :2], dim=-1))
        return rew.squeeze(-1)

    def _reward_tracking_vx(self):
        rew = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        # print("vx rew", rew, self.base_lin_vel[:, 0], self.commands[:, 0])
        return rew

    def _reward_tracking_demo_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        # print("yaw rew", rew, self.target_yaw, self.yaw)
        return rew

    def _reward_tracking_demo_dof_pos(self):
        demo_dofs = self._curr_demo_obs_buf[:, :self._n_demo_dof]
        dof_pos = self.dof_pos[:, self._dof_ids_subset]
        rew = torch.exp(-0.7 * torch.norm((dof_pos - demo_dofs), dim=1))
        # print(rew[self.lookat_id].cpu().numpy())
        # print("dof_pos", dof_pos)
        # print("demo_dofs", demo_dofs)
        return rew

    def _reward_tracking_demo_ang_vel(self):
        demo_ang_vel = self._curr_demo_obs_buf[:, self._n_demo_dof+3:self._n_demo_dof+6]
        rew = torch.exp(-torch.norm(self.base_ang_vel - demo_ang_vel, dim=1))
        return rew

    def _reward_tracking_demo_roll_pitch(self):
        demo_roll_pitch = self._curr_demo_obs_buf[:, self._n_demo_dof+6:self._n_demo_dof+8]
        cur_roll_pitch = torch.stack((self.roll, self.pitch), dim=1)
        rew = torch.exp(-torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
        return rew

    def _reward_tracking_demo_height(self):
        demo_height = self._curr_demo_obs_buf[:, self._n_demo_dof+8]
        cur_height = self.root_states[:, 2]
        rew = torch.exp(- 4 * torch.abs(cur_height - demo_height))
        return rew


    def _reward_tracking_demo_key_body(self):
        # demo_key_body_pos_local = self._curr_demo_obs_buf[:, self.num_dof*2+8:].view(self.num_envs, self._num_key_bodies, 3)[:,self._key_body_ids_sim_subset,:].view(self.num_envs, -1)
        # cur_key_body_pos_local = global_to_local(self.base_quat, self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3], self.root_states[:, :3]).view(self.num_envs, -1)
        
        demo_key_body_pos_local = self._curr_demo_keybody.view(self.num_envs, self._num_key_bodies, 3)
        if self.cfg.motion.global_keybody:
            curr_demo_xyz = torch.cat((self.target_pos_abs, self._curr_demo_root_pos[:, 2:3]), dim=-1)
        else:
            curr_demo_xyz = torch.cat((self.root_states[:, :2], self._curr_demo_root_pos[:, 2:3]), dim=-1)
        demo_global_body_pos = local_to_global(self._curr_demo_quat, demo_key_body_pos_local, curr_demo_xyz).view(self.num_envs, -1)
        cur_global_body_pos = self.rigid_body_states[:, self._key_body_ids_sim[self._key_body_ids_sim_subset], :3].view(self.num_envs, -1)

        # cur_local_body_pos = global_to_local(self.base_quat, cur_global_body_pos.view(self.num_envs, -1, 3), self.root_states[:, :3]).view(self.num_envs, -1)
        # print(cur_local_body_pos)
        rew = torch.exp(-torch.norm(cur_global_body_pos - demo_global_body_pos, dim=1))
        # print("key body rew", rew[self.lookat_id].cpu().numpy())
        return rew

    def _reward_energy(self):
        return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    def _reward_feet_height(self):
        feet_height = self.rigid_body_states[:, self.feet_indices, 2]
        rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
        rew[self._in_place_flag] = 0
        # print("height: ", rew[self.lookat_id])
        return rew
    
    def _reward_feet_force(self):
        rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
        rew[rew < 500] = 0
        rew[rew > 500] -= 500
        rew[self._in_place_flag] = 0
        # print(rew[self.lookat_id])
        # print(self.dof_names)
        return rew





def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets):
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    local_root_vel = quat_rotate_inverse(root_rot, root_vel)
        # print(local_root_vel[0])

        # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
        # local_root_vel = quat_rotate(heading_rot, root_vel)
        # print(local_root_vel[0], "\n")

        # root_pos_expand = root_pos.unsqueeze(-2)  # [num_envs, 1, 3]
        # local_key_body_pos = key_body_pos - root_pos_expand
        
        # heading_rot_expand = heading_rot.unsqueeze(-2)
        # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
        # flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
        # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
        # local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
        # flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)   





@torch.jit.script
def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
    dof = dof.clone()
    dof[:, indices_sim] = dof[:, indices_motion]
    return dof[:, valid_dof_body_ids]

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))