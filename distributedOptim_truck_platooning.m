%% Decentralized Distributed MPC for Truck Platooning 
% DESCRIPTION:
% This script simulates a platoon of trucks where each truck attempts to
% optimize its trajectory in a decentralized and distributed manner using
% Model Predictive Control (MPC). The overall goal is to have the platoon
% move cohesively, with the leader tracking a reference velocity, followers
% maintaining safe and efficient spacing, and all trucks minimizing fuel.
%
% METHODOLOGY: DECENTRALIZED DISTRIBUTED MODEL PREDICTIVE CONTROL (DMPC)
%
% 1. Agent-Based Control:
%    - Each truck `i` in the platoon acts as an independent agent equipped
%      with its own MPC controller.
%
% 2. Local Optimization at Each Time Step:
%    - At every control interval `dt_control`, each truck `i` solves a
%      finite-horizon optimal control problem for its own future actions.
%    - Decision Variables (for truck `i`): A sequence of its own future
%      accelerations `a_i(k), a_i(k+1), ..., a_i(k+Hp-1)` over a
%      prediction horizon `Hp`.
%    - Local Objective Function (for truck `i`): This is a weighted sum of
%      truck-specific goals:
%        - Leader (Truck 1):
%            - Track a global reference velocity profile (`v_ref_global`).
%            - Minimize its own estimated fuel consumption.
%            - (Optional) Smooth its acceleration profile.
%        - Followers (Truck `i > 1`):
%            - Track the predicted velocity of the preceding truck (`i-1`).
%            - Maintain a target inter-truck gap (bumper-to-bumper) with
%              the preceding truck (`i-1`), based on a constant minimum
%              and a time-headway policy.
%            - Minimize its own estimated fuel consumption.
%            - (Optional) Smooth its acceleration profile.
%    - Local Constraints (for truck `i`):
%        - Its own vehicle dynamics (predicting future position and velocity
%          based on its planned accelerations).
%        - Limits on its own acceleration (`a_min`, `a_max`).
%        - Limits on its own velocity (`v_min`, `v_max`).
%        - For followers: A hard safety constraint on the minimum spacing
%          (front-to-front separation) with the preceding truck (`i-1`).
%    - Solver: Each truck's local optimization problem, being a Nonlinear
%      Program (NLP) due to the fuel model and dynamics, is solved using
%      MATLAB's `fmincon`.
%
% 3. Information Exchange and "Distributed" Nature:
%    - The "distributed" aspect arises from how trucks share information to
%      make their local decisions.
%    - Preceding Vehicle Information: To achieve coordinated behavior, each
%      follower truck `i` requires the predicted future trajectory (position
%      and velocity over the horizon `Hp`) of the truck immediately
%      preceding it (`i-1`).
%    - Communication Assumption (Simplified in this script):
%        - This script simulates a sequential solution: the leader solves its
%          MPC first, then the first follower, then the second, and so on,
%          all within the same simulation time step `k_sim`.
%        - When truck `i-1` solves its MPC, its planned trajectory could
%          ideally be communicated to truck `i`.
%        - **Simplification in this Script:** For simplicity in this conceptual
%          outline, the prediction of the preceding truck's behavior is
%          rudimentary (e.g., assumes it continues its last applied
%          acceleration). A full DMPC would involve more sophisticated
%          management and use of these inter-agent predictions.
%    - No Central Coordinator for Trajectory Commands: Unlike the centralized
%      approach where one solver plans for everyone, here, each truck plans
%      for itself based on its local goals and information from its neighbors.
%
% 4. MPC Receding Horizon Principle:
%    - After solving its local optimization, each truck `i` applies only the
%      *first* acceleration command `a_i(k)` from its optimal sequence.
%    - At the next time step `k_sim + 1`, the entire process (state update,
%      information gathering, local optimization) is repeated with updated
%      states and a shifted prediction horizon.
%
% 5. Overall Behavior:
%    - The collective behavior of the platoon emerges from these local,
%      interacting optimizations. The effectiveness of the DMPC depends on
%      the design of local objectives, constraints, prediction accuracy of
%      neighboring vehicles, and the stability of the interactions.
%
% LIMITATIONS OF THIS CONCEPTUAL SCRIPT:
%    - Preceding Truck Prediction: The mechanism for followers to get
%      accurate predictions of the leader's future actions is highly
%      simplified. Real DMPC systems use more robust prediction/communication.
%    - Iteration/Convergence: This script uses a single-pass sequential
%      solution within each time step. More advanced DMPC might involve
%      multiple iterations between trucks at a single time step to converge
%      to a better-coordinated plan (e.g., using game theory concepts or
%      algorithms like ADMM adapted for dynamic systems).
%    - Stability: Formal stability analysis of DMPC schemes is complex.
%    - `mpcActiveSetSolver` is not directly used here because each local problem
%      is solved as an NLP with `fmincon`. To use an active-set QP solver,
%      the local problems would need to be formulated as Quadratic Programs.
%
% CURRENT IMPLEMENTATION: Decentralized Model Predictive Control with
% sequential solving and simplified inter-agent prediction.
% --

% Copyright 2025, The MathWorks, Inc.

clear; clc; close all;
rng(1);

%  Simulation Parameters
T_simulation_sec = 60; % Total simulation duration
dt = 1.0;      % Control/MPC update interval (seconds)
num_sim_steps = T_simulation_sec / dt;

%  Platoon & Vehicle Parameters
N = 3; L_truck = 18; m = 20000*ones(N,1); C_d_truck = 0.6; A_f = 10;
C_r = 0.007; rho_air = 1.225; g = 9.81; eta_drivetrain = 0.85;
P_idle = 5000; k_fuel_g_per_J = 5.83e-5;
shielding_factors = zeros(N,1);
if N > 1, shielding_factors(2) = 0.3; end
if N > 2, shielding_factors(3) = 0.4; end
C_d_eff = C_d_truck * (1 - shielding_factors);
v_min = 10; v_max = 25; a_min = -2.0; a_max = 1.0;
s_min_gap_const = 5; t_h = 1.0;

%  MPC Parameters for each truck
Hp = 10; % Prediction horizon (number of steps, e.g., 10 * dt_control seconds)
% Hc = Hp; % Control horizon (often same as prediction horizon for simplicity here)

%  Objective Weights (can be different per truck or role)
w_v_ref_leader = 1.0;       % Leader tracking v_ref
w_v_track_follower = 0.8;   % Follower tracking leader's velocity
w_gap_control = 1.0;        % Follower gap control
w_fuel_leader = 0.01;
w_fuel_follower = 0.05;
w_accel_smooth = 0.01;      % Penalize large changes in acceleration (optional)

%  Reference Velocity Profile for the Leader
v_ref_profile = ones(num_sim_steps + Hp, 1) * 20; % Extended for lookahead
% Example: Trapezoidal profile (adjust for num_sim_steps)
if num_sim_steps + Hp >= 20
    v_ref_profile(1:10) = linspace(15, 22, 10);
    v_ref_profile(11:num_sim_steps+Hp-10) = 22;
    v_ref_profile(num_sim_steps+Hp-9:num_sim_steps+Hp) = linspace(22, 18, 10);
end

v_ref_profile = min(max(v_ref_profile, v_min), v_max);

%  Initial States
s_platoon = zeros(N, num_sim_steps + 1);
v_platoon = zeros(N, num_sim_steps + 1);
a_platoon_actual = zeros(N, num_sim_steps); % Store applied accelerations

v_initial = v_ref_profile(1) * ones(N,1);
s_initial = zeros(N,1); s_initial(1) = 0;
for i = 2:N
    desired_gap_initial = L_truck + s_min_gap_const + t_h * v_initial(i);
    s_initial(i) = s_initial(i-1) - desired_gap_initial;
end
s_platoon(:,1) = s_initial;
v_platoon(:,1) = v_initial;

%  Main Simulation Loop
% Store predicted trajectories from other trucks (simplified: only from preceding truck)
predicted_s_ahead = zeros(Hp+1, 1); % Position of truck i-1
predicted_v_ahead = zeros(Hp+1, 1); % Velocity of truck i-1

fprintf('Starting Decentralized Distributed MPC Simulation...\n');
for k_sim = 1:num_sim_steps % Current simulation time step
    fprintf('Simulation Step %d/%d\n', k_sim, num_sim_steps);

    current_s = s_platoon(:, k_sim);
    current_v = v_platoon(:, k_sim);

    % Store optimal accelerations for this step
    optimal_a_this_step = zeros(N,1);

    %  Iterate through trucks (Leader solves first, then followers)
    % This is a sequential pass; more advanced schemes might iterate.
    for i_truck = 1:N
        % Get relevant portion of global reference if leader
        v_ref_horizon = [];
        if i_truck == 1
            v_ref_horizon = v_ref_profile(k_sim : k_sim + Hp -1);
        end

        a_prev_applied_this_truck = []; % Initialize for the case k_sim == 1
        if k_sim > 1
            a_prev_applied_this_truck = a_platoon_actual(i_truck, k_sim-1);
        end

        % Get predictions from truck ahead (i_truck-1)
        % In a real system, this comes via communication.
        % Here, we'd use the *planned* trajectory of truck i-1 from its MPC solve
        % at this *same* simulation step k_sim (if solved sequentially) or previous step k_sim-1.
        % For simplicity, let's assume truck i-1 has already solved its MPC for this step k_sim
        % and we can access its planned trajectory.
        if i_truck > 1
            % This is a simplification: In reality, truck i-1 would have solved its MPC
            % and communicated its plan. Here we'd need to store those plans.
            % For now, let's use its *actual* current state and assume it maintains current accel
            % or a simple prediction. This is a major simplification point.
            % A better approach: after truck i-1 solves its MPC, store its optimal_a_sequence
            % and then predict s_ahead, v_ahead using that.
            if k_sim == 1 && i_truck > 1 % First step, no prior optimal_a for truck i-1
                predicted_s_ahead(1) = current_s(i_truck-1);
                predicted_v_ahead(1) = current_v(i_truck-1);
                temp_a_ahead = 0; % Assume 0 accel for truck ahead initially
                for h=1:Hp
                    predicted_v_ahead(h+1) = predicted_v_ahead(h) + temp_a_ahead * dt;
                    predicted_v_ahead(h+1) = max(v_min, min(v_max, predicted_v_ahead(h+1)));
                    predicted_s_ahead(h+1) = predicted_s_ahead(h) + predicted_v_ahead(h)*dt + 0.5*temp_a_ahead*dt^2;
                end
            elseif i_truck > 1
                % Ideally, use the planned trajectory of truck i-1 from its MPC solve
                % at this time step k_sim. This requires careful data flow management.
                % For this conceptual code, we'll just make a crude prediction.
                % This is NOT how a proper DMPC would get this.
                last_applied_a_ahead = 0;
                if k_sim > 1, last_applied_a_ahead = a_platoon_actual(i_truck-1, k_sim-1); end

                predicted_s_ahead(1) = current_s(i_truck-1);
                predicted_v_ahead(1) = current_v(i_truck-1);
                for h=1:Hp
                    predicted_v_ahead(h+1) = predicted_v_ahead(h) + last_applied_a_ahead * dt; % Assume it continues its last accel
                    predicted_v_ahead(h+1) = max(v_min, min(v_max, predicted_v_ahead(h+1)));
                    predicted_s_ahead(h+1) = predicted_s_ahead(h) + predicted_v_ahead(h)*dt + 0.5*last_applied_a_ahead*dt^2;
                end
            end
        end

        %  Define and Solve Local MPC Problem for i_truck
        % Decision variables: a_i_sequence = [a_i(0), ..., a_i(Hp-1)]'
        num_dec_vars_local = Hp;
        lb_a_local = ones(num_dec_vars_local, 1) * a_min;
        ub_a_local = ones(num_dec_vars_local, 1) * a_max;
        a_guess_local = zeros(num_dec_vars_local, 1);
        if k_sim > 1 % Use previous applied accel as guess
            a_guess_local(:) = a_platoon_actual(i_truck, k_sim-1);
        end
        % Get the specific current state for THIS truck for THIS iteration
        s_i_current_val = current_s(i_truck); % Scalar
        v_i_current_val = current_v(i_truck); % Scalar

        % Get the specific parameters for THIS truck
        m_i_val = m(i_truck);
        C_d_eff_i_val = C_d_eff(i_truck);

        % Objective function for this truck's MPC
        objFunLocal = @(a_i_seq) mpcObjectiveTruck_i(a_i_seq, i_truck, N, Hp, dt, ...
            s_i_current_val, v_i_current_val, ...
            v_ref_horizon, predicted_s_ahead, predicted_v_ahead, ...
            m_i_val, C_d_eff_i_val, A_f, C_r, rho_air, g, ...
            eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, s_min_gap_const, t_h, ...
            a_prev_applied_this_truck, ...
            w_v_ref_leader, w_v_track_follower, w_gap_control, ...
            w_fuel_leader, w_fuel_follower, w_accel_smooth);

        % Nonlinear constraints for this truck's MPC (velocity limits, spacing for followers)
        nonlconFunLocal = @(a_i_seq) mpcConstraintsTruck_i(a_i_seq, i_truck, N, Hp, dt, ...
            s_i_current_val, v_i_current_val, ...
            predicted_s_ahead, ...
            v_min, v_max, L_truck, s_min_gap_const, t_h);

        % SOLVER:
        % If the problem is formulated as a QP (quadratic objective, linear constraints),
        % mpcActiveSetSolver or quadprog could be used.
        % If NLP (as it is now with nonlinear fuel/dynamics in objective/constraints),
        % fmincon is more appropriate for each local problem.
        options_local_mpc = optimoptions('fmincon', 'Algorithm','sqp', ...
            'Display','none', 'MaxIterations', 50, ...
            'MaxFunctionEvaluations', 5000, 'ConstraintTolerance', 1e-3, ...
            'OptimalityTolerance', 1e-3, 'StepTolerance', 1e-7);

        [a_i_optimal_sequence, J_local_opt] = fmincon(objFunLocal, a_guess_local, ...
            [], [], [], [], lb_a_local, ub_a_local, ...
            nonlconFunLocal, options_local_mpc);

        optimal_a_this_step(i_truck) = a_i_optimal_sequence(1); % Apply first step (MPC principle)

        %  Store this truck's planned trajectory if needed by truck i+1
        % This part is crucial for true DMPC. We'd predict truck i's states using
        % a_i_optimal_sequence and pass that to truck i+1 when it solves its MPC.
        % (Skipped in this simplified conceptual loop for brevity, but this is where
        % the `predicted_s_ahead` for truck i+1 would be properly generated).

    end % End loop over i_truck

    %  Apply optimal accelerations and update platoon state
    a_platoon_actual(:, k_sim) = optimal_a_this_step;
    v_platoon(:, k_sim+1) = v_platoon(:, k_sim) + optimal_a_this_step * dt;
    % Clip velocities here for safety if solver slightly violates
    v_platoon(:, k_sim+1) = max(v_min, min(v_max, v_platoon(:, k_sim+1)));

    s_platoon(:, k_sim+1) = s_platoon(:, k_sim) + v_platoon(:, k_sim) * dt ...
        + 0.5 * optimal_a_this_step * dt^2;

end % End simulation loop k_sim
fprintf('Decentralized Distributed MPC Simulation Finished.\n');

%  Plotting Results  (Similar to centralized version)
% Plot v_platoon, a_platoon_actual, s_platoon, gaps, etc.
time_axis_sim = (0:num_sim_steps) * dt;
time_axis_accel_sim = (0:num_sim_steps-1) * dt;

figure;
plot(time_axis_sim, v_platoon', 'LineWidth', 1.5);
hold on;
plot(time_axis_accel_sim, v_ref_profile(1:num_sim_steps), 'k--', 'LineWidth', 1.5);
title('Truck Velocities (Decentralized MPC)'); xlabel('Time (s)'); ylabel('Velocity (m/s)');
legend_entries_v = cell(N+1,1);
for i=1:N, legend_entries_v{i} = sprintf('Truck %d', i); end
legend_entries_v{N+1} = 'Reference (Leader)'; legend(legend_entries_v); grid on;

% ... Add other plots for acceleration, position, gaps ...
if N > 1
    figure;
    for i = 2:N
        actual_gap = (s_platoon(i-1,:) - L_truck) - s_platoon(i,:);
        desired_min_gap = s_min_gap_const + t_h * v_platoon(i,:);
        subplot(N-1,1,i-1);
        plot(time_axis_sim, actual_gap, 'b-'); hold on;
        plot(time_axis_sim, desired_min_gap, 'r--');
        title(sprintf('Gap T%d-T%d (DMPC)', i-1,i)); ylabel('Gap (m)'); grid on;
        legend('Actual Bumper-Gap', 'Desired Min Bumper-Gap');
    end
    sgtitle('Inter-truck Gaps (Decentralized MPC)');
end

%%  MPC Objective Function for Truck i
function J_i = mpcObjectiveTruck_i(a_i_seq, i_truck, N_total_trucks, Hp, dt, ...
    s_i_current, v_i_current, ...
    v_ref_leader_horizon, s_ahead_pred, v_ahead_pred, ...
    m_i, C_d_eff_i, A_f, C_r, rho_air, g, ...
    eta_drivetrain, P_idle, k_fuel_g_per_J, L_truck, s_min_gap_const_i, t_h_i, ...
    a_prev_applied_i, ...
    w_v_ref_L, w_v_track_F, w_gap_F, w_fuel_L, w_fuel_F, w_accel_smooth_i) %#ok<INUSD>

% Predict states of truck i over horizon Hp based on a_i_seq
s_i_pred = zeros(Hp+1, 1);
v_i_pred = zeros(Hp+1, 1);
s_i_pred(1) = s_i_current;
v_i_pred(1) = v_i_current;
fuel_i_total = 0;

for h = 1:Hp
    % Dynamics
    v_i_pred(h+1) = v_i_pred(h) + a_i_seq(h) * dt;
    s_i_pred(h+1) = s_i_pred(h) + v_i_pred(h) * dt + 0.5 * a_i_seq(h) * dt^2;

    % Fuel for truck i at step h
    current_v_fuel = v_i_pred(h); % Velocity at start of interval h
    if current_v_fuel < 0.1, current_v_fuel = 0.1; end
    F_aero = 0.5 * rho_air * C_d_eff_i * A_f * current_v_fuel^2;
    F_roll = m_i * g * C_r;
    F_inertia = m_i * a_i_seq(h);
    F_total = F_aero + F_roll + F_inertia;
    P_wheel = F_total * current_v_fuel;
    if P_wheel >= 0
        P_engine = P_wheel / eta_drivetrain + P_idle;
    else
        P_engine = P_idle;
    end
    fuel_i_total = fuel_i_total + (k_fuel_g_per_J * P_engine * dt);
end

% Cost components

%  Acceleration Smoothing Cost

% Option 1: Penalize squared accelerations (simpler, often effective)
cost_accel_squared = w_accel_smooth_i * sum(a_i_seq.^2); %#ok<NASGU>

% Option 2: Penalize change in acceleration (jerk) within the horizon
cost_accel_diff_horizon = 0;
if length(a_i_seq) > 1
    cost_accel_diff_horizon = w_accel_smooth_i * sum(diff(a_i_seq).^2);
end

% Option 3: Penalize change from PREVIOUS applied acceleration to the FIRST planned one
% This requires passing the previous acceleration or k_sim
% To implement this robustly, you'd need to pass a_prev_applied_i as an argument
% to mpcObjectiveTruck_i.
% Let's assume you've added a_prev_applied_i to the function arguments:
% function J_i = mpcObjectiveTruck_i(a_i_seq, ..., a_prev_applied_i, w_accel_smooth_i)

if ~isempty(a_prev_applied_i) % Check if a previous acceleration is available/passed
    cost_accel_diff_initial_step = w_accel_smooth_i * (a_i_seq(1) - a_prev_applied_i)^2;
else % First step or no previous accel provided
    cost_accel_diff_initial_step = w_accel_smooth_i * (a_i_seq(1) - 0)^2; % Penalize first accel from zero
end
% Choose one or a combination for the total smoothing cost
% For simplicity, let's combine Option 2 and Option 3 (if a_prev_applied_i is passed)
cost_accel_smooth = cost_accel_diff_horizon + cost_accel_diff_initial_step;
% Or, if you just want to penalize jerk within the horizon:
% cost_accel_smooth = cost_accel_diff_horizon;
% Or, if you only want to penalize large accelerations directly:
% cost_accel_smooth = cost_accel_squared;

if length(a_i_seq)>1, cost_accel_smooth = cost_accel_smooth + w_accel_smooth_i * sum(diff(a_i_seq).^2); end


if i_truck == 1 % Leader
    cost_v_tracking = w_v_ref_L * sum((v_i_pred(2:end) - v_ref_leader_horizon(1:Hp)).^2);
    cost_fuel = w_fuel_L * fuel_i_total;
    cost_gap_control = 0;
else % Follower
    % Velocity tracking (track vehicle ahead)
    cost_v_tracking = w_v_track_F * sum((v_i_pred(2:end) - v_ahead_pred(2:end)).^2); % v_ahead_pred is Hp+1 long

    % Gap control
    actual_bumper_gap = (s_ahead_pred(2:end) - L_truck) - s_i_pred(2:end);
    target_bumper_gap = s_min_gap_const_i + t_h_i * v_i_pred(2:end); % Target based on own speed
    cost_gap_control = w_gap_F * sum((actual_bumper_gap - target_bumper_gap).^2);

    cost_fuel = w_fuel_F * fuel_i_total;
end

J_i = cost_v_tracking + cost_gap_control + cost_fuel + cost_accel_smooth;
end

%%  MPC Constraints Function for Truck i
function [c, ceq] = mpcConstraintsTruck_i(a_i_seq, i_truck, N_total_trucks, Hp, dt, ...
    s_i_current, v_i_current, ...
    s_ahead_pred, ... % Predicted pos of truck ahead
    v_min_i, v_max_i, L_truck_i, s_min_gap_const_i, t_h_i) %#ok<INUSD>
% Predict states of truck i
s_i_pred = zeros(Hp+1, 1);
v_i_pred = zeros(Hp+1, 1);
s_i_pred(1) = s_i_current;
v_i_pred(1) = v_i_current;

for h = 1:Hp
    v_i_pred(h+1) = v_i_pred(h) + a_i_seq(h) * dt;
    s_i_pred(h+1) = s_i_pred(h) + v_i_pred(h) * dt + 0.5 * a_i_seq(h) * dt^2;
end

c = [];
% Velocity constraints for truck i (for steps 1 to Hp, i.e., v_i_pred(2:end))
c = [c; v_i_pred(2:end) - v_max_i];
c = [c; v_min_i - v_i_pred(2:end)];

% Spacing constraint if follower
if i_truck > 1
    % s_ahead_pred(h+1) - s_i_pred(h+1) >= L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(h+1)
    % => (L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(h+1)) - (s_ahead_pred(h+1) - s_i_pred(h+1)) <= 0
    desired_front_to_front_sep = L_truck_i + s_min_gap_const_i + t_h_i * v_i_pred(2:end);
    actual_front_to_front_sep = s_ahead_pred(2:end) - s_i_pred(2:end);
    c_spacing = desired_front_to_front_sep - actual_front_to_front_sep;
    c = [c; c_spacing];
end
ceq = [];
end