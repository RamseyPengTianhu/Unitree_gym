from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
        #    'left_knee_joint' : 0.3,       
           'left_knee_joint' : 0.0,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
        #    'right_knee_joint' : 0.3,                                             
           'right_knee_joint' : 0.0,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'waist_yaw_joint' : 0.,
           'waist_roll_joint' : 0.,
           'waist_pitch_joint' : 0.,
           'left_shoulder_pitch_joint': 0.0,
           'left_shoulder_roll_joint': 0.0,
           'left_shoulder_yaw_joint': 0.,
           'left_elbow_joint': 0.6,
           'right_shoulder_pitch_joint': 0.0,
           'right_shoulder_roll_joint': 0.0,
           'right_shoulder_yaw_joint': 0.,
           'right_elbow_joint': 0.6
        }
    
    class env(LeggedRobotCfg.env):
        # num_fixed_joint = -1

        num_observations = 11 + 3*num_actions
        # num_observations = 47 + 26 + 187
        # num_observations = 77
        num_privileged_obs = 14 + 26 + 3*num_actions
        # num_privileged_obs = 50

        num_actions = 15

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
        # class ranges:
        #     lin_vel_x = [0.0, 0] # min max [m/s]
        #     lin_vel_y = [0.0, 0]   # min max [m/s]
        #     ang_vel_yaw = [0, 0]    # min max [rad/s]
        #     heading = [0, 0]



    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'waist_yaw': 400,
                     'waist_roll': 90,
                     'waist_pitch': 80,
                     'knee': 150,
                     'ankle': 40,
                     'shoulder_pitch': 90,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 20.,
                     'elbow': 60
                    #  'ankle_pitch': 35,
                    #  'ankle_roll': 30,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'waist_yaw': 2.5,
                     'waist_roll': 0.8,
                     'waist_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 1,
                     'shoulder_yaw': 0.4,
                     'elbow': 1


                    #  'ankle_pitch': 4,
                    #  'ankle_roll': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_15dof.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_15dof_fixed.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        # base_height_target = 0.78
        base_height_target = 0.78
        locomotion_max_contact_force = 300.0
        
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.2
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18
            
            straight_knee = 4
            feet_drag = -0.01
            tracking_torso_roll= 0.8
            tracking_torso_pitch= 0.5
            # stabilize_waist_yaw = 0.0001
            minimize_torso_angular_velocity = 0.4
            # penalty_ang_vel_xy_torso = -0.1
            minimize_waist_pitch_deviation = 0.3
            # penalty_feet_slippage = -0.01
            # penalty_feet_contact_forces = -0.01

            # minimize_whole_body_angular_momentum = 0.1
            # minimize_angular_momentum_rate = 0.05
            # minimize_arm_torque = 0.02
            # stable_standing = 0.01
            # heel_toe_walking = 0.01
            # minimize_waist_pitch_deviation = 0.5
            # tracking_roll= 1.5
            # tracking_pitch = 0.5

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'
