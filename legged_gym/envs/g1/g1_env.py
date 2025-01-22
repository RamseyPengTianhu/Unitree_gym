
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
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
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_vec = torch.zeros(47, device = self.device)
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

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    def compute_obs_buf(self):
        # imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        imu_obs = self.rpy[:,0:1]
        self.yaw = self.rpy[:,2]
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            self.rpy[:,0:1],    #[1,2]
                            torch.sin(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            torch.cos(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            # self.target_pos_rel,  
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            self.reindex_feet(self.contact_filt.float()*0-0.5),
                            ),dim=-1)

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
        # Indices for knee joints (e.g., 2nd and 3rd joints for each leg)
        knee_indices = [3, 9]  # Assuming these are the knee joint indices
        # Penalize deviation from the desired straight knee position
        straight_knee_error = torch.square(self.dof_pos[:, knee_indices])
        return -torch.sum(straight_knee_error, dim=1)
    



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
        self.dof_indices_sim = torch.tensor([0, 1, 2, 5, 6, 7, 11, 12, 13, 16, 17, 18], device=self.device, dtype=torch.long)
        self.dof_indices_motion = torch.tensor([2, 0, 1, 7, 5, 6, 12, 11, 13, 17, 16, 18], device=self.device, dtype=torch.long)
        
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
        if cfg.motion.motion_type == "single":
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/retarget_npy/{cfg.motion.motion_name}.npy")
        else:
            assert cfg.motion.motion_type == "yaml"
            motion_file = os.path.join(ASE_DIR, f"ase/poselib/data/configs/{cfg.motion.motion_name}")
        
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