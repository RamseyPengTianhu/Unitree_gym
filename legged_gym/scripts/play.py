import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt



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


def plot_deviations(state_log, save_path=None):
    """
    Plot pitch, roll, and yaw deviations.

    Args:
        state_log (dict): Dictionary containing logged deviations.
        save_path (str): Path to save the plots (optional).
    """
    plt.figure(figsize=(10, 6))
    if 'pitch_dev' in state_log:
        plt.plot(state_log['pitch_dev'], label='Pitch Deviation')
    if 'roll_dev' in state_log:
        plt.plot(state_log['roll_dev'], label='Roll Deviation')
    if 'yaw_dev' in state_log:
        plt.plot(state_log['yaw_dev'], label='Yaw Deviation')

    plt.xlabel('Time Steps')
    plt.ylabel('Deviation')
    plt.title('Upper Body Stability Deviations')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_upper_body_orientation(state_log, save_path=None):
    """
    Plot pitch, roll, and yaw for the upper body.

    Args:
        state_log (dict): Dictionary containing the logged data (tensors or arrays).
        save_path (str): Path to save the plot. If None, the plot will just be shown.
    """
    plt.figure(figsize=(10, 6))

    # Safely handle tensors on GPU or CPU
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()  # Move to CPU and convert to NumPy
        return np.array(tensor)  # Convert list of arrays to NumPy array

    # Define body part labels and line styles
    body_parts = ["Pelvis", "Waist", "Torso"]
    line_styles = ['-', '--', ':']  # Different line styles for clarity

    # # Convert and plot pitch
    # if 'pitch' in state_log:
    #     pitch = to_numpy(state_log['pitch'])
    #     pitch = np.array(pitch)  # Ensure proper shape
    #     print("Pitch shape:", pitch.shape)  # Debugging
        
    #     # Plot each body part separately
    #     for i in range(min(3, pitch.shape[1])):  # Avoid index errors
    #         plt.plot(pitch[:, i], linestyle=line_styles[i % len(line_styles)], label=f'Pitch - {body_parts[i]}')

    # Convert and plot roll
    if 'roll' in state_log:
        roll = to_numpy(state_log['roll'])
        roll = np.array(roll)
        print("Roll shape:", roll.shape)
        
        # Plot each body part separately
        for i in range(min(3, roll.shape[1])):
            plt.plot(roll[:, i], linestyle=line_styles[i % len(line_styles)], label=f'Roll - {body_parts[i]}')

    # Yaw is optional, uncomment if needed
    # if 'yaw' in state_log:
    #     yaw = to_numpy(state_log['yaw'])
    #     plt.plot(yaw, label='Yaw')

    # Add labels and title
    plt.xlabel('Time Steps')
    plt.ylabel('Orientation (Radians)')
    plt.title('Upper Body Orientation (Pitch & Roll)')
    plt.legend()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_com(state_log, save_path=None):
    """
    Plot the Center of Mass (CoM) trajectory over time for X and Y directions.

    Args:
        state_log (dict): Dictionary containing logged CoM values.
                          Expected keys: 'com_x', 'com_y'
        save_path (str): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))

    # Ensure keys exist in state_log
    if 'com_x' not in state_log or 'com_y' not in state_log:
        print("Error: Missing 'com_x' or 'com_y' in state_log.")
        return

    # Convert tensors to NumPy arrays (if necessary)
    com_x = state_log['com_x']
    com_y = state_log['com_y']

    if isinstance(com_x, torch.Tensor):
        com_x = com_x.detach().cpu().numpy()
    if isinstance(com_y, torch.Tensor):
        com_y = com_y.detach().cpu().numpy()

    # Generate time steps based on data length
    time_steps = np.arange(len(com_x))

    # Plot CoM X and Y trajectories
    plt.plot(time_steps, com_x, label='CoM X', color='b', linestyle='-', linewidth=2)
    plt.plot(time_steps, com_y, label='CoM Y', color='r', linestyle='--', linewidth=2)

    # Add inline labels at the last point
    # plt.text(time_steps[-1], com_x[-1], f'X: {com_x[-1]:.3f}', fontsize=10, color='b')
    # plt.text(time_steps[-1], com_y[-1], f'Y: {com_y[-1]:.3f}', fontsize=10, color='r')

    # Add labels, title, and legend
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('CoM Position (m)', fontsize=12)
    plt.title('Center of Mass (CoM) Trajectory Over Time', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
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
    stop_state_log = 1000  # Number of steps for logging
    joint_index = 4
    # Indices for the upper body links

    camera_offset = np.array([0.0, -3.0, 1.0])  # Adjust this offset as needed

    # Prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    logger = Logger(env.dt)
    stop_rew_log = env.max_episode_length + 1  # Steps for average reward calculation





    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(1 * int(env.max_episode_length)):
        # Check and update command ranges
        command_interface = _check_command_interface()
        _update_command_ranges(env, command_interface)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        


        upper_body_rpy = env._extract_upper_body_rpy()
        upper_body_rpy_numpy = upper_body_rpy.detach().cpu().numpy()
        upper_roll = upper_body_rpy_numpy[:,0]
        print('upper_roll.shape:',upper_roll)
        upper_pitch = upper_body_rpy_numpy[:,1]
        upper_yaw = upper_body_rpy_numpy[:,2]
        com = env.calculate_center_of_mass()
        com = com.detach().cpu().numpy()
        com_x = com[:,0]
        com_y = com[:,1]



        # Logging deviations

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

                    # 'pitch_dev': pitch_dev,
                    # 'roll_dev': roll_dev,
                    # 'yaw_dev': yaw_dev,
                    'pitch': upper_pitch,
                    'roll': upper_roll,
                    'yaw': upper_yaw,
                    'com_x':com_x,
                    'com_y':com_y,
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'Left_knee_pos': env.dof_pos[robot_index, 3].item(),
                    'Right_knee_pos': env.dof_pos[robot_index, 9].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            # logger.plot_states()
            # plot_deviations(logger.state_log, save_path='/home/tianhu/unitree_rl_gym/logs/upper_body_deviations.png')
            plot_upper_body_orientation(
            logger.state_log, 
            save_path='/home/tianhu/unitree_rl_gym/logs/upper_body_orientation.png'
    )
            plot_com(
            logger.state_log, 
            save_path='/home/tianhu/unitree_rl_gym/logs/com.png'
    )
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

