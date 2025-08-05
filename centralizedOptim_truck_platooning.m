%% Optimal Trajectory Planning for Centralized Truck Platooning using fmincon
% DESCRIPTION:
% This script solves a trajectory optimization problem for a platoon of
% trucks. The goal is to find optimal acceleration profiles for each truck
% over a finite time horizon to achieve two main objectives:
%   1. The leading truck tracks a predefined reference velocity profile.
%   2. The total fuel consumption of the entire platoon is minimized.
%
% The optimization is subject to several constraints:
%   - Individual truck dynamics (position and velocity based on acceleration).
%   - Limits on each truck's acceleration and velocity.
%   - Inter-truck spacing constraints to ensure safety, typically based on
%     a constant minimum gap and a time-headway policy (the gap increases
%     with the follower's speed).
%
% METHODOLOGY:
% The problem is formulated as a nonlinear constrained optimization problem
% and solved using MATLAB's `fmincon` solver.
%
%   - Decision Variables: The acceleration `a_i(k)` for each truck `i` at
%     each discrete time step `k` over the horizon. These are flattened
%     into a single vector for `fmincon`.
%
%   - Objective Function: A weighted sum of several cost components:
%       - Leader's velocity tracking error (squared difference between
%         leader's actual velocity and reference velocity).
%       - Followers' velocity tracking error (squared difference between
%         a follower's velocity and the leader's velocity). This encourages
%         platoon cohesion.
%       - Total platoon fuel consumption, estimated using a simplified
%         physics-based model (aerodynamic drag, rolling resistance,
%         inertial forces, drivetrain efficiency, idle power). Aerodynamic
%         shielding for following trucks is considered.
%       - Gap control error (squared difference between the actual
%         bumper-to-bumper gap and a target bumper-to-bumper gap for
%         followers). This provides a "soft" constraint to maintain desired
%         spacing.
%
%   - Constraints Handled by `fmincon`:
%       - Bound Constraints: Limits on acceleration values (`a_min`, `a_max`)
%         are directly provided to `fmincon`.
%       - Nonlinear Constraints:
%           - Velocity limits (`v_min`, `v_max`) for each truck at each
%             time step.
%           - Minimum inter-truck spacing (safety constraint): The actual
%             front-to-front distance between consecutive trucks must be
%             greater than or equal to a desired minimum separation, which
%             includes the length of the truck ahead, a constant safety
%             margin, and a speed-dependent term (time headway).
%
% "DISTRIBUTED OPTIMIZATION" CONTEXT (Clarification):
% It is important to note that *this specific script implements a
% CENTRALIZED optimization approach*. All information (vehicle parameters,
% initial states, reference profiles, constraint details) is known by a
% central solver (`fmincon`), which then determines the optimal acceleration
% profiles for ALL trucks simultaneously.
%
% The term "distributed optimization" typically implies that individual agents
% (trucks, in this case) would solve parts of the problem locally, using
% only local information and information exchanged with their immediate
% neighbors, and then coordinate to reach a global or near-global optimum.
% Examples of distributed optimization algorithms include ADMM (Alternating
% Direction Method of Multipliers), DGD (Distributed Gradient Descent),
% consensus algorithms, etc.
%
% While the problem *could* be reformulated for a distributed solution (e.g.,
% using MPC where each truck solves its own optimization, or by decomposing
% the centralized problem), this script does NOT currently employ such
% distributed techniques. The previous examples involving DGD and ADMM were
% for a simpler static resource allocation problem. This script tackles a
% more complex dynamic trajectory optimization.
%
% CURRENT IMPLEMENTATION: Centralized Optimal Control.
%
% USAGE:
% 1. Define simulation, platoon, vehicle, constraint, and objective parameters.
% 2. `fmincon` is called with the objective function and nonlinear constraints.
% 3. Results (trajectories, costs) are post-processed and plotted.
%
% KEY PARAMETERS FOR TUNING:
% - `w_v_leader`, `w_v_follower`, `w_f`, `w_gap_control`: Weights in the
%   objective function, critical for balancing conflicting goals.
% - `fmincon` options (`MaxIterations`, `MaxFunctionEvaluations`, tolerances).
% - Time horizon (`T_horizon_sec`) and time step (`dt`).

% Copyright 2025, The MathWorks, Inc.

clear; clc; close all;
rng(1); % for reproducibility if any stochastic elements were present (not here)

%  Simulation Parameters 
T_horizon_sec = 60; % Simulation horizon (seconds)
dt = 1;             % Time step (seconds)
T_steps = T_horizon_sec / dt; % Number of time steps

%  Platoon Parameters 
N = 3;              % Number of trucks
L_truck = 18;       % Length of each truck (m) - for spacing

%  Vehicle Parameters (assuming identical trucks for simplicity) 
m = 20000 * ones(N,1); % Mass of each truck (kg)
C_d_truck = 0.6;    % Base aerodynamic drag coefficient
A_f = 10;           % Frontal area (m^2)
C_r = 0.007;        % Rolling resistance coefficient
rho_air = 1.225;    % Air density (kg/m^3)
g = 9.81;           % Gravity (m/s^2)
eta_drivetrain = 0.85; % Drivetrain efficiency
P_idle = 5000;      % Idle power consumption (Watts) - e.g. 5 kW
k_fuel = 7e-8;      % Fuel consumption factor (e.g., kg_fuel/Joule or L_fuel/Joule)
% Typical diesel: ~200-220 g/kWh. 210 g/kWh = 210/(1000*3600) g/Ws = 5.83e-5 g/Ws
% If k_fuel is set to get roughly Liters, 1L diesel ~ 0.85kg, energy ~38MJ/L
% So 1L / 38e6 J = 2.63e-8 L/J. Let's use k_fuel in g/J
k_fuel_g_per_J = 5.83e-5; % (grams of fuel per Joule of engine energy)

% Aerodynamic shielding factors (0 = no shielding, 1 = perfect shielding)
% These are highly dependent on spacing, but we simplify for now
shielding_factors = zeros(N,1);
if N > 1, shielding_factors(2) = 0.3; end % 2nd truck gets 30% drag reduction
if N > 2, shielding_factors(3) = 0.4; end % 3rd truck gets 40%
if N > 3, shielding_factors(4:N) = 0.45; end % Others
C_d_eff = C_d_truck * (1 - shielding_factors);

%  Constraints 
v_min = 10;         % Minimum velocity (m/s) approx 36 km/h
v_max = 25;         % Maximum velocity (m/s) approx 90 km/h
a_min = -2.0;       % Maximum deceleration (m/s^2)
a_max = 1.0;        % Maximum acceleration (m/s^2)

% Spacing policy: s_front_bumper(i-1) - s_front_bumper(i) >= L_truck + s_min_gap + t_h * v_i(k)
s_min_gap_const = 5; % Minimum constant gap between rear of leader and front of follower (m)
t_h = 1.0;          % Time headway (s)

%  Objective Function Weights 
w_v_leader = 1;     % Weight for leader velocity tracking error
w_v_follower = 0.5; % Weight for follower velocity tracking error (NEW)
w_f = 0.01;         % Weight for total fuel consumption
% Fuel is in grams, velocity error squared is (m/s)^2.
% A fuel value of 1000g vs vel error of 1 (m/s)^2
% w_f might need to be much smaller if fuel values are large.
w_gap_control = 0.2; % weight for gap control

% Reference Velocity Profile for the Leader (m/s) 
% Example: Trapezoidal profile
%  Reference Velocity Profile for the Leader (m/s) 
v_ref_profile = ones(T_steps, 1) * 20; % Start with a default
if T_steps >= 20 % Ensure enough steps for trapezoid
    v_ref_profile(1:10) = linspace(15, 22, 10);
    v_ref_profile(11:T_steps-10) = 22;
    v_ref_profile(T_steps-9:T_steps) = linspace(22, 18, 10);
elseif T_steps >= 10
    v_ref_profile(1:10) = linspace(15,20,10);
    if T_steps > 10
        v_ref_profile(11:T_steps) = 20;
    end
else % If very few steps, just make it constant or simple ramp
    v_ref_profile(:) = 15; % Or some other simple profile
end

v_ref_profile = min(max(v_ref_profile, v_min), v_max); % Ensure ref is within bounds

%  Initial Conditions 
v_initial = v_ref_profile(1) * ones(N,1); % All start at leader's initial reference speed
s_initial = zeros(N,1);
s_initial(1) = 0; % Leader starts at 0 m
for i = 2:N
    % Start trucks spaced according to policy at initial speed
    desired_gap_initial = L_truck + s_min_gap_const + t_h * v_initial(i);
    s_initial(i) = s_initial(i-1) - desired_gap_initial;
end

%% Optimization Problem

%  Optimization Variables 
% Accelerations a_i(k) for i=1..N, k=1..T_steps
% We flatten this into a single vector for fmincon:
% [a_1(1)...a_1(T_steps), a_2(1)...a_2(T_steps), ..., a_N(1)...a_N(T_steps)]'
num_decision_vars = N * T_steps;

% Bounds for decision variables (accelerations)
lb_a = ones(num_decision_vars, 1) * a_min;
ub_a = ones(num_decision_vars, 1) * a_max;

% Initial guess for accelerations (e.g., zero acceleration)
a_guess_flat = zeros(num_decision_vars, 1);
% Slightly better guess: leader tries to maintain v_ref(1) if v_initial(1) is different
% For simplicity, keep zero accel guess.

%  fmincon Options 
options = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ... % 'interior-point' or 'sqp'
    'Display', 'iter-detailed', ...
    'MaxIterations', 500, ... % Might need more for convergence
    'MaxFunctionEvaluations', 20000, ... % Might need more
    'ConstraintTolerance', 1e-4, ...
    'OptimalityTolerance', 1e-4, ...
    'StepTolerance', 1e-7, ...
    'UseParallel', false); % Set to true if you have Parallel Computing Toolbox

%  Call fmincon 
fprintf('Starting optimization for truck platooning...\n');
objFun = @(a_flat) objectiveFunctionPlatoon(a_flat, N, T_steps, dt, ...
    s_initial, v_initial, v_ref_profile, ...
    m, C_d_eff, A_f, C_r, rho_air, g, ...
    eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, ...
    s_min_gap_const, t_h, ...
    w_v_leader, w_v_follower, w_f, w_gap_control);

nonlconFun = @(a_flat) nonlinearConstraintsPlatoon(a_flat, N, T_steps, dt, ...
    s_initial, v_initial, ...
    v_min, v_max, L_truck, ...
    s_min_gap_const, t_h);
tic;
[a_optimal_flat, J_optimal, exitflag, output] = fmincon(objFun, a_guess_flat, ...
    [], [], [], [], lb_a, ub_a, ...
    nonlconFun, options);
toc;

fprintf('Optimization finished.\n');
fprintf('Exit flag: %d\n', exitflag);
disp(output.message);
fprintf('Optimal objective function value: %.4e\n', J_optimal);

%%  Post-process and Plot Results 
% Reshape optimal accelerations
a_optimal = reshape(a_optimal_flat, T_steps, N)'; % N x T_steps

% Simulate with optimal accelerations to get s and v trajectories
s_trajectory = zeros(N, T_steps + 1);
v_trajectory = zeros(N, T_steps + 1);
s_trajectory(:,1) = s_initial;
v_trajectory(:,1) = v_initial;

for k = 1:T_steps
    v_trajectory(:,k+1) = v_trajectory(:,k) + a_optimal(:,k) * dt;
    % Apply velocity constraints (clipping, though fmincon should respect them via nonlcon)
    v_trajectory(:,k+1) = max(v_min, min(v_max, v_trajectory(:,k+1)));

    s_trajectory(:,k+1) = s_trajectory(:,k) + v_trajectory(:,k) * dt + 0.5 * a_optimal(:,k) * dt^2;
end

% Calculate individual cost components with optimal trajectory
% Make sure the number of '1's here matches the number of weight inputs
% in your objectiveFunctionPlatoon (w_v_leader, w_v_follower, w_f, w_gap_control if used)
% Assuming you have w_v_leader, w_v_follower, w_f (and no w_gap_control yet)
% In the post-processing section:
if N > 1
    [~, fuel_total_optimal_raw, vel_track_leader_cost_optimal_raw, vel_track_follower_cost_optimal_raw, gap_control_cost_raw] = ...
        objectiveFunctionPlatoon(a_optimal_flat, N, T_steps, dt, ...
        s_initial, v_initial, v_ref_profile, ...
        m, C_d_eff, A_f, C_r, rho_air, g, ...
        eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, ...
        s_min_gap_const, t_h, ... 
        1, 1, 1, 1); % Unit weights for w_v_leader, w_v_follower, w_f, w_gap_control
else % Only a leader
    [~, fuel_total_optimal_raw, vel_track_leader_cost_optimal_raw, ~, ~] = ...
        objectiveFunctionPlatoon(a_optimal_flat, N, T_steps, dt, ...
        s_initial, v_initial, v_ref_profile, ...
        m, C_d_eff, A_f, C_r, rho_air, g, ...
        eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, ...
        s_min_gap_const, t_h, ... 
        1, 0, 1, 0); % Unit weights, follower and gap weights are 0
    vel_track_follower_cost_optimal_raw = 0;
    gap_control_cost_raw = 0;
end

fprintf('Optimal Total Objective Value (J_optimal from fmincon): %.4e\n', J_optimal);
fprintf(' Raw Cost Components (calculated with unit weights) \n');
fprintf('Optimal Leader Velocity Tracking Cost (sum sq error): %.4e\n', vel_track_leader_cost_optimal_raw);
if N > 1
    fprintf('Optimal Follower(s) Velocity Tracking Cost (sum sq error): %.4e\n', vel_track_follower_cost_optimal_raw);
end
fprintf('Optimal Total Fuel Consumption (grams): %.2f g\n', fuel_total_optimal_raw);

% If you add gap control cost:
[~, fuel_total_optimal_raw, vel_track_leader_cost_optimal_raw, vel_track_follower_cost_optimal_raw, gap_control_cost_raw] = ...
    objectiveFunctionPlatoon(a_optimal_flat, N, T_steps, dt, ...
    s_initial, v_initial, v_ref_profile, ...
    m, C_d_eff, A_f, C_r, rho_air, g, ...
    eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, ...
    s_min_gap_const, t_h, ... 
    1, 1, 1, 1);
fprintf('Optimal Gap Control Cost (sum sq error): %.4e\n', gap_control_cost_raw);


time_axis = (0:T_steps) * dt;
time_axis_accel = (0:T_steps-1) * dt;

% Plot Velocities
figure;
plot(time_axis, v_trajectory', 'LineWidth', 1.5);
hold on;
plot(time_axis_accel, v_ref_profile, 'k--', 'LineWidth', 1.5);
title('Truck Velocities');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend_entries_v = cell(N+1,1);
for i=1:N, legend_entries_v{i} = sprintf('Truck %d', i); end
legend_entries_v{N+1} = 'Reference (Leader)';
legend(legend_entries_v, 'Location', 'best');
grid on;

% Add an annotation with key cost values to the velocity plot
annotation_str = {
    sprintf('J_{optimal}: %.2e', J_optimal),
    sprintf('Fuel_{total}: %.0f g', fuel_total_optimal_raw),
    sprintf('Cost_{V,Leader}: %.2e', vel_track_leader_cost_optimal_raw)
    };
if N > 1
    annotation_str{end+1} = sprintf('Cost_{V,Follower}: %.2e', vel_track_follower_cost_optimal_raw);
end
% Add other costs if you have them, e.g., gap_control_cost_raw

% Plot Accelerations
figure;
plot(time_axis_accel, a_optimal', 'LineWidth', 1.5);
title('Truck Accelerations');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
yline(a_max, 'r--', 'Max Accel');
yline(a_min, 'b--', 'Min Accel');
legend_entries_a = cell(N,1);
for i=1:N, legend_entries_a{i} = sprintf('Truck %d', i); end
legend(legend_entries_a, 'Location', 'best');
grid on;

% Plot Positions
figure;
plot(time_axis, s_trajectory', 'LineWidth', 1.5);
title('Truck Positions');
xlabel('Time (s)');
ylabel('Position (m)');
legend(legend_entries_a, 'Location', 'best'); % Same legend as accel
grid on;

% Plot Inter-truck Gaps
if N > 1
    figure;
    gaps_actual = zeros(N-1, T_steps + 1);
    gaps_desired = zeros(N-1, T_steps + 1); % For desired minimum based on policy

    % If you have gap control cost, you might also want to plot the target gap used in the objective
    % gaps_target_objective = zeros(N-1, T_steps + 1);

    for i = 2:N
        gaps_actual(i-1, :) = (s_trajectory(i-1,:) - L_truck) - s_trajectory(i,:); % Rear of (i-1) to front of i
        gaps_desired(i-1, :) = s_min_gap_const + t_h * v_trajectory(i,:); % This is the Desired *Minimum* Gap

        % If you had a target gap for objective (from w_gap_control part)
        % target_bumper_to_bumper_gap = s_min_gap_const + t_h * v_trajectory(i,2:end);
        % gaps_target_objective(i-1, 2:end) = target_bumper_to_bumper_gap;
        % gaps_target_objective(i-1, 1) = s_min_gap_const + t_h * v_trajectory(i,1); % for t=0

        subplot(N-1, 1, i-1);
        plot(time_axis, gaps_actual(i-1,:), 'b-', 'LineWidth', 1.5);
        hold on;
        plot(time_axis, gaps_desired(i-1,:), 'r--', 'LineWidth', 1.5);
        % if you plotted gaps_target_objective, add it here:
        % plot(time_axis, gaps_target_objective(i-1,:), 'g:', 'LineWidth', 1.5);

        title(sprintf('Gap between Truck %d and Truck %d', i-1, i));
        xlabel('Time (s)');
        ylabel('Gap (m)');
        legend_gap = {'Actual Bumper-to-Bumper Gap', 'Desired Min Gap (Constraint)'};
        % if you plotted gaps_target_objective, add to legend:
        % legend_gap = [legend_gap, {'Target Gap (Objective)'}];
        legend(legend_gap, 'Location', 'best');
        grid on;

        % Adjust ylim to better see the region of interest
        min_plot_gap = min([gaps_actual(i-1,:), gaps_desired(i-1,:)]) - 5;
        max_plot_gap = max([gaps_actual(i-1,:), gaps_desired(i-1,:)]) + 10;
        ylim([min_plot_gap, max_plot_gap]);
    end
    sgtitle('Inter-truck Gaps');
end

%%  Objective Function 
function [J, total_fuel_g, vel_tracking_cost_leader, vel_tracking_cost_followers, gap_control_cost] = ...
    objectiveFunctionPlatoon(a_flat, N, T_steps, dt, ...
    s0, v0, v_ref, ...
    m, C_d_eff, A_f, C_r, rho_air, g, ...
    eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, ...
    s_min_gap_const, t_h, ...
    w_v_leader, w_v_follower, w_f, w_gap_control)

a = reshape(a_flat, T_steps, N)'; % N x T_steps

s = zeros(N, T_steps + 1);
v = zeros(N, T_steps + 1);
s(:,1) = s0;
v(:,1) = v0;

total_fuel_g = 0;

%  Main Simulation Loop to get s and v trajectories and fuel 
for k = 1:T_steps
    v(:,k+1) = v(:,k) + a(:,k) * dt;
    s(:,k+1) = s(:,k) + v(:,k) * dt + 0.5 * a(:,k) * dt^2;

    for i = 1:N % Fuel for each truck
        current_v_fuel = v(i,k); % Use v(i,k) or v(i,k+1) or avg? Consistent is key. Using v(i,k) (start of interval)
        if current_v_fuel < 0.1, current_v_fuel = 0.1; end

        F_aero = 0.5 * rho_air * C_d_eff(i) * A_f * current_v_fuel^2;
        F_roll = m(i) * g * C_r;
        F_inertia = m(i) * a(i,k);
        F_total = F_aero + F_roll + F_inertia;
        P_wheel = F_total * current_v_fuel;

        if P_wheel >= 0
            P_engine = P_wheel / eta_drivetrain + P_idle;
        else
            P_engine = P_idle;
        end
        fuel_rate_g_per_s = k_fuel_g_per_J * P_engine;
        fuel_interval_g = fuel_rate_g_per_s * dt;
        total_fuel_g = total_fuel_g + fuel_interval_g;
    end
end

%  Calculate Cost Components (after trajectories are known) 
% Leader velocity tracking (velocities at k=1 to T_steps, which is v(:,2) to v(:,T_steps+1))
vel_tracking_cost_leader = sum((v(1, 2:end)' - v_ref).^2);

vel_tracking_cost_followers = 0;
gap_control_cost = 0;

if N > 1
    for i = 2:N
        % Follower velocity tracking (compare v_follower with v_leader over the horizon)
        vel_tracking_cost_followers = vel_tracking_cost_followers + ...
            sum((v(i, 2:end)' - v(1, 2:end)').^2);
        % Alternative: Followers track v_ref
        % vel_tracking_cost_followers = vel_tracking_cost_followers + sum((v(i, 2:end)' - v_ref).^2);

        % Gap control cost (compare actual gap with target gap over the horizon)
        % s values are s(time_index), v values are v(time_index)
        % We care about gaps at time steps k=1...T_steps (indices 2 to T_steps+1 in s and v)
        actual_bumper_to_bumper_gap = (s(i-1, 2:end) - L_truck) - s(i, 2:end); % Row vector
        target_bumper_to_bumper_gap = s_min_gap_const + t_h * v(i, 2:end);    % Row vector for v at end of step
        gap_control_cost = gap_control_cost + sum((actual_bumper_to_bumper_gap - target_bumper_to_bumper_gap).^2);
    end
end

J = w_v_leader * vel_tracking_cost_leader + ...
    w_v_follower * vel_tracking_cost_followers + ...
    w_f * total_fuel_g +...
    w_gap_control * gap_control_cost;
end

%%  Nonlinear Constraint Function 
function [c, ceq] = nonlinearConstraintsPlatoon(a_flat, N, T_steps, dt, ...
    s0, v0, v_min, v_max, L_truck, ...
    s_min_gap_const, t_h)

a = reshape(a_flat, T_steps, N)'; % N x T_steps

s = zeros(N, T_steps + 1);
v = zeros(N, T_steps + 1);
s(:,1) = s0;
v(:,1) = v0;

% Simulate dynamics based on accelerations 'a'
for k = 1:T_steps
    v(:,k+1) = v(:,k) + a(:,k) * dt;
    s(:,k+1) = s(:,k) + v(:,k) * dt + 0.5 * a(:,k) * dt^2;
end

% Inequality constraints c(x) <= 0
c = [];

% 1. Velocity constraints: v_i(k) <= v_max  => v_i(k) - v_max <= 0
%                       v_i(k) >= v_min  => v_min - v_i(k) <= 0
% Applied for k=1...T_steps (i.e., v(:,2:T_steps+1))
c = [c; reshape(v(:,2:end) - v_max, N*T_steps, 1)];
c = [c; reshape(v_min - v(:,2:end), N*T_steps, 1)];

% 2. Inter-truck spacing constraints
% For trucks i = 2:N
% s_{i-1}(k) - s_i(k) >= L_truck + s_min_gap_const + t_h * v_i(k)
% => (L_truck + s_min_gap_const + t_h * v_i(k)) - (s_{i-1}(k) - s_i(k)) <= 0
% Applied for k=1...T_steps (i.e., s_i(k+1), v_i(k+1))
if N > 1
    for i = 2:N
        s_leader_front = s(i-1, 2:end); % Positions of truck i-1 at k=1...T_steps
        s_follower_front = s(i, 2:end);   % Positions of truck i   at k=1...T_steps
        v_follower = v(i, 2:end);       % Velocities of truck i  at k=1...T_steps

        % Actual gap: front bumper of (i-1) to front bumper of (i)
        actual_separation = s_leader_front - s_follower_front;

        % Desired minimum separation (front-to-front)
        desired_min_separation = L_truck + s_min_gap_const + t_h * v_follower;

        c_spacing = desired_min_separation - actual_separation;
        c = [c; c_spacing(:)]; % Append as column
    end
end

% Equality constraints ceq(x) = 0 (none in this formulation)
ceq = [];
end