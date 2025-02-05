import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import matplotlib.pyplot as plt


import numpy as np
import torch
import json
import pandas as pd


def save_states_to_csv(state_log, dt, output_dir):
    """
    Save each state in the state log to a separate CSV file.
    
    Args:
        state_log (dict): Dictionary containing state logs.
        dt (float): Time step for the simulation.
        output_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create time vector
    max_len = max(len(v) for v in state_log.values())
    time = np.linspace(0, max_len * dt, max_len)
    pd.DataFrame(time).to_csv(os.path.join(output_dir, 'time.csv'), index=False, header=['time'])

    # Save each state to a separate CSV file
    for key, value in state_log.items():
        df = pd.DataFrame(value)
        if df.shape[1] == 1:
            df.columns = [key]
        df.to_csv(os.path.join(output_dir, f'{key}.csv'), index=False)

def update_camera_position(env, robot_index, camera_offset):
    """Update the camera position to track the robot."""
    # Get the robot's current position
    robot_position = env.root_states[robot_index, :3].cpu().numpy()

    # Calculate the new camera position
    new_camera_position = robot_position + camera_offset

    # Update the camera position and look-at target
    env.set_camera(new_camera_position, robot_position)

def _check_command_interface():
    """Check for command updates from the shared file."""
    try:
        with open('command_interface.json', 'r') as file:
            command_interface = json.load(file)
        return command_interface
    except FileNotFoundError:
        return {}

def _update_command_ranges(env, command_interface):
    """Update command ranges in the environment."""
    if 'lin_vel_x' in command_interface:
        env.command_ranges["lin_vel_x"] = command_interface['lin_vel_x']
    if 'lin_vel_y' in command_interface:
        env.command_ranges["lin_vel_y"] = command_interface['lin_vel_y']
    if 'ang_vel_yaw' in command_interface:
        env.command_ranges["ang_vel_yaw"] = command_interface['ang_vel_yaw']
    if 'heading' in command_interface:
        env.command_ranges["heading"] = command_interface['heading']



def stabilize_robot(env):
    # Define the fixed root state for the robot
    fixed_position = [0.0, 0.0, 1.0]  # x, y, z position
    fixed_orientation = [0.0, 0.0, 0.0, 1.0]  # Quaternion: x, y, z, w

    # Create a tensor for the root states
    root_states = torch.zeros((env.num_envs, 13), device=env.device, dtype=torch.float32)
    
    # Set position (indices 0, 1, 2)
    root_states[:, 0:3] = torch.tensor(fixed_position, device=env.device)

    # Set orientation (indices 3, 4, 5, 6)
    root_states[:, 3:7] = torch.tensor(fixed_orientation, device=env.device)

    # Set the root state for all environments
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(root_states))



def plot_logged_data(state_log, save_path=None):
    """
    Plot the logged data for waist roll and pitch in separate graphs.

    Args:
        state_log (dict): Dictionary containing the logged data.
        save_path (str): Path to save the plot. If None, the plot will just be shown.
    """
    # Plot for Waist Roll
    plt.figure(figsize=(10, 6))
    # if 'waist_roll_torque' in state_log:
    #     plt.plot(state_log['waist_roll_torque'], label='Waist Roll Torque')
    if 'waist_roll_position' in state_log:
        plt.plot(state_log['waist_roll_position'], label='Waist Roll Position')
    # if 'waist_roll_velocity' in state_log:
    #     plt.plot(state_log['waist_roll_velocity'], label='Waist Roll Velocity')
    if 'desired_position' in state_log:
        plt.plot(state_log['desired_position'], label='desired_position')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.title("Waist Roll PD Tuning")
    if save_path:
        plt.savefig(save_path + "_waist_roll.png")
        print(f"Waist Roll plot saved to {save_path}_waist_roll.png")
    else:
        plt.show()

    # Plot for Waist Pitch
    plt.figure(figsize=(10, 6))
    if 'waist_pitch_torque' in state_log:
        plt.plot(state_log['waist_pitch_torque'], label='Waist Pitch Torque')
    if 'waist_pitch_position' in state_log:
        plt.plot(state_log['waist_pitch_position'], label='Waist Pitch Position')
    if 'waist_pitch_velocity' in state_log:
        plt.plot(state_log['waist_pitch_velocity'], label='Waist Pitch Velocity')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.title("Waist Pitch PD Tuning")
    if save_path:
        plt.savefig(save_path + "_waist_pitch.png")
        print(f"Waist Pitch plot saved to {save_path}_waist_pitch.png")
    else:
        plt.show()
        
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False


    env_cfg.env.test = True
    robot_index = 0  # Index of the robot to track
    stop_state_log = 200  # Number of steps for logging
    joint_index = 4



    camera_offset = np.array([0.0, -3.0, 1.0])  # Adjust this offset as needed

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    logger = Logger(env.dt)
    stop_rew_log = env.max_episode_length + 1  # Steps for average reward calculation





    # Load policy
    # train_cfg.runner.resume = True
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     print('Exported policy as jit script to: ', path)
    # Define the sine wave parameters
    amplitude = 0.2  # Amplitude of the sine wave
    frequency = 0.05  # Frequency of the sine wave (Hz)
    time_step = env.dt  # Time step of the simulation
    time = 0  # Initialize time

    for i in range(2 * int(env.max_episode_length)):
        # Check and update command ranges
        # command_interface = _check_command_interface()
        # _update_command_ranges(env, command_interface)
        # actions = policy(obs.detach())
        # obs, _, rews, dones, infos = env.step(actions.detach())

        # Correct initialization of custom_actions
        custom_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

        # Example: Set specific values for waist_roll and waist_pitch joints
        waist_roll_index = 8  # Index for waist_roll_joint
        # waist_pitch_index = 13  # Index for waist_pitch_joint

        # Custom PD tuning values (these can be tuned interactively or systematically)
        desired_positions = torch.zeros((env.num_envs, env.num_dofs), device=env.device)
        desired_velocities = torch.zeros_like(desired_positions)
        # Convert time to a torch.Tensor
        time_tensor = torch.tensor(time, device=env.device)

        # Compute the desired position as a sine wave
        desired_positions[:, [waist_roll_index]] = amplitude * torch.sin(2 * np.pi * frequency * time_tensor)

        # Increment time
        time += time_step
        # p_gains = torch.tensor([100.0, 50], device=env.device)  # PD position gains for roll and pitch
        # d_gains = torch.tensor([6.0, 4.0], device=env.device)    # PD velocity gains for roll and pitch
        p_gains = torch.tensor([10], device=env.device)  # PD position gains for roll and pitch
        d_gains = torch.tensor([1], device=env.device)    # PD velocity gains for roll and pitch

        # Calculate torques for waist joints
        # joint_positions = env.dof_pos[:, [waist_roll_index, waist_pitch_index]]
        # joint_velocities = env.dof_vel[:, [waist_roll_index, waist_pitch_index]]

        joint_positions = env.dof_pos[:, [waist_roll_index]]
        print('joint_positions:',joint_positions)
        print('joint_positions[:, 0]:',joint_positions[:, 0])
        joint_velocities = env.dof_vel[:, [waist_roll_index]]
        # torques = p_gains * (desired_positions[:, [waist_roll_index, waist_pitch_index]] - joint_positions) \
        #         - d_gains * joint_velocities
        torques = p_gains * (desired_positions[:, [waist_roll_index]] - joint_positions) \
                - d_gains * joint_velocities
        

        # Apply calculated torques
        # custom_actions[:, [waist_roll_index, waist_pitch_index]] = torques
        custom_actions[:, [waist_roll_index]] = torques
        print('custom_actions:',custom_actions)

        # Step the environment with custom actions
        obs, _, rews, dones, infos = env.step(custom_actions)
        # Attach the robot to a fixed point in the simulation
        base_handle = env.gym.get_actor_handle(env.envs[0], 0)

        # Define a fixed root state
        fixed_root_state = gymapi.Transform(
            p=gymapi.Vec3(0.0, 0.0, 1.0),  # Position above the ground
            r=gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # Neutral orientation
        )

        # Apply the fixed root state
        stabilize_robot(env)


        # Update the camera position dynamically
        if MOVE_CAMERA:
            update_camera_position(env, robot_index, camera_offset)

        # Optional: Add logic for rendering or saving frames if needed
        if RECORD_FRAMES and i % 2 == 0:
            filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'frames', f'{i}.png')
            env.gym.write_viewer_image_to_file(env.viewer, filename)



        # Logging states
        if i < stop_state_log:
            logger.log_states(
            {
                'waist_roll_torque': torques[:, 0].cpu().numpy(),
                # 'waist_pitch_torque': torques[:, 1].cpu().numpy(),
                'waist_roll_position': joint_positions[:, 0].cpu().numpy(),
                # 'waist_pitch_position': joint_positions[:, 1].cpu().numpy(),
                'waist_roll_velocity': joint_velocities[:, 0].cpu().numpy(),
                'desired_position': desired_positions[:, waist_roll_index].cpu().numpy(),

                # 'waist_pitch_velocity': joint_velocities[:, 1].cpu().numpy(),
            }
        )
        elif i == stop_state_log:
            # logger.plot_states()
            plot_logged_data(logger.state_log)

            pass

        # Logging rewards
        if 0 < i < stop_rew_log and infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    # Save logged states
    save_states_to_csv(logger.state_log, env.dt, '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee')


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
