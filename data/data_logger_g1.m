% Define the file paths
timeFile = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/time.csv';
Left_knee_pos_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/Left_knee_pos.csv';
Right_knee_pos_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/Right_knee_pos.csv';
contact_force_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/contact_forces_z.csv';
base_linear_x_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/base_vel_x.csv';
base_linear_y_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/base_vel_y.csv';
base_yaw_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/base_vel_yaw.csv';
Command_base_linear_x_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/command_x.csv';
Command_base_linear_y_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/command_y.csv';
Command_base_yaw_File = '/home/tianhu/unitree_rl_gym/data/g1_sim/straight_knee/command_yaw.csv';

% Load the data
time = readmatrix(timeFile);
Left_knee_pos = readmatrix(Left_knee_pos_File);
Right_knee_pos = readmatrix(Right_knee_pos_File);
contact_force = readmatrix(contact_force_File); % Expecting two columns: [Left_Foot, Right_Foot]
base_linear_x = readmatrix(base_linear_x_File);
base_linear_y = readmatrix(base_linear_y_File);
base_yaw = readmatrix(base_yaw_File);
Command_base_linear_x = readmatrix(Command_base_linear_x_File);
Command_base_linear_y = readmatrix(Command_base_linear_y_File);
Command_base_yaw = readmatrix(Command_base_yaw_File);

% Split contact force data into left and right foot
Left_foot_force = contact_force(:, 1);
Right_foot_force = contact_force(:, 2);

% Remove data for the first 0.5 seconds
valid_indices = time >= 5;
time = time(valid_indices);
Left_knee_pos = Left_knee_pos(valid_indices);
Right_knee_pos = Right_knee_pos(valid_indices);
Left_foot_force = Left_foot_force(valid_indices);
Right_foot_force = Right_foot_force(valid_indices);
base_linear_x = base_linear_x(valid_indices);
base_linear_y = base_linear_y(valid_indices);
base_yaw = base_yaw(valid_indices);
Command_base_linear_x = Command_base_linear_x(valid_indices);
Command_base_linear_y = Command_base_linear_y(valid_indices);
Command_base_yaw = Command_base_yaw(valid_indices);

% Create plots
figure;

% Plot knee positions
subplot(2, 2, 1);
plot(time, Left_knee_pos, 'b', 'DisplayName', 'Left Knee');
hold on;
plot(time, Right_knee_pos, 'r', 'DisplayName', 'Right Knee');
xlabel('Time (s)');
ylabel('Knee Position');
title('Knee Positions');
legend;
grid on;

% Plot contact forces for left and right foot
subplot(2, 2, 2);
plot(time, Left_foot_force, 'b', 'DisplayName', 'Left Foot');
hold on;
plot(time, Right_foot_force, 'r', 'DisplayName', 'Right Foot');
xlabel('Time (s)');
ylabel('Contact Force (z)');
title('Contact Forces');
legend;
grid on;

% Plot base linear x velocity vs command x
subplot(2, 2, 3);
plot(time, base_linear_x, 'b', 'DisplayName', 'Actual Vel X');
hold on;
plot(time, Command_base_linear_x, '--r', 'DisplayName', 'Target Vel X');
xlabel('Time (s)');
ylabel('Linear Velocity X (m/s)');
title('Base Linear Velocity X');
legend;
grid on;

% Plot base linear y velocity vs command y
subplot(2, 2, 4);
plot(time, base_linear_y, 'b', 'DisplayName', 'Actual Vel Y');
hold on;
plot(time, Command_base_linear_y, '--r', 'DisplayName', 'Target Vel Y');
xlabel('Time (s)');
ylabel('Linear Velocity Y (m/s)');
title('Base Linear Velocity Y');
legend;
grid on;

% New figure for base yaw and command yaw
figure;
plot(time, base_yaw, 'b', 'DisplayName', 'Actual Yaw Vel');
hold on;
plot(time, Command_base_yaw, '--r', 'DisplayName', 'Target Yaw Vel');
xlabel('Time (s)');
ylabel('Yaw Velocity (rad/s)');
title('Base Yaw Velocity vs Target');
legend;
grid on;