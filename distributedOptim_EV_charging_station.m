%% Distributed Resource Allocation for EV Charging 
%-----------------------------------------------------------------

% Copyright 2025, The MathWorks, Inc.

clear; clc; close all;
rng(1); % fix the random number generator, to obtain reproducible results

% Number of EVs (agents)
N = 10;

% Total Power Capacity of the charging station (e.g., kW)
P_total = 50; 

% Local data: Each EV i has a preferred charging rate p_pref(i)
% Let's say preferred rates are between 3 kW and 11 kW (typical Level 2 charging)
p_pref = rand(N,1) * 8 + 3; % Randomly between 3 and 11 kW

fprintf('Preferred charging rates (kW):\n');
disp(p_pref');
fprintf('Sum of preferred rates: %.2f kW\n', sum(p_pref));
fprintf('Total available power: %.2f kW\n\n', P_total);

%% Solve with DGD

% For simplicity, use complete graph avg.
fprintf('Using simplified complete graph weights for W (equal averaging).\n');
W = ones(N) / N; 

% % Communication graph: Ring topology for a slightly more realistic scenario
% % than complete graph, but still simple.
% % For a ring, each agent communicates with two neighbors.
% W = zeros(N,N);
% for i = 1:N
%     W(i,i) = 1/3; % Weight for self
%     W(i, mod(i-1-1, N)+1) = 1/3; % Weight for previous neighbor (with wrap-around)
%     W(i, mod(i, N)+1) = 1/3;     % Weight for next neighbor (with wrap-around)
% end

% Stepsize for DGD (small constant for stability)
alpha = 0.2; 

% Initialization: Each agent's estimate of mean(p_pref) starts from zero
% (or could start from its own p_pref_i)
z = zeros(N,1); % z_i will be agent i's estimate of mean(p_pref)

% Number of iterations for consensus
T = 50;

% Store history for plotting convergence of z
Zhist = zeros(N, T+1);
Zhist(:,1) = z;

% DGD for estimating mean(p_pref) 
for k = 1:T
    % Each agent updates its estimate z_i based on neighbors' estimates
    % and its own preferred rate p_pref_i.
    % The DGD update: z_new = W * z_old - alpha * (z_old - p_pref)
    % Here, 'a' from the simple example is 'p_pref'.
    % 'x' from the simple example is 'z'.
    z = W * z - alpha * (z - p_pref); 
    
    Zhist(:,k+1) = z;
end

% True mean of preferred rates
mean_p_pref_true = mean(p_pref);

% Plot convergence of z_i to mean(p_pref)
figure;
plot(0:T, Zhist', 'LineWidth', 1.2);
hold on;
yline(mean_p_pref_true, 'k--', 'LineWidth', 2);
xlabel('Iteration k');
ylabel('z_i^{(k)} (Estimate of mean p_{pref})');
title('DGD: Estimating Average Preferred Charging Rate');
legendStrings = arrayfun(@(i) sprintf('EV %d',i), 1:N, 'UniformOutput', false);
legend([legendStrings, {'True Mean p_{pref}'}],'Location','best');
grid on;

% Calculate final allocated power 
% Each EV uses its final estimate of mean(p_pref) (which is z(:,T+1))
% If consensus is perfect, all z_i(T+1) are close to mean_p_pref_true.
estimated_mean_p_pref = Zhist(:, T+1); % Each EV has its own estimate

% Optimal allocation formula: p_i = p_pref_i + P_total/N - estimated_mean_p_pref_i
p_allocated = p_pref + (P_total/N) - estimated_mean_p_pref;

% Ensure non-negativity (cannot draw negative power in this simple model)
% and also that no EV exceeds a maximum possible rate (e.g., its p_pref_i or a physical limit like 11kW)
% For simplicity, we'll just do non-negativity.
% More advanced: if an EV hits 0, its demand is removed and power redistributed among others.
p_allocated = max(0, p_allocated);

% Due to max(0,.) constraint, sum(p_allocated) might not be P_total.
% A simple proportional rescaling if total allocated power is not P_total.
% This is a heuristic step after the distributed calculation.
% A fully distributed scheme for exact sum constraint with bounds is more complex (e.g., using ADMM).
current_sum_p_allocated = sum(p_allocated);
if current_sum_p_allocated > 0 && abs(current_sum_p_allocated - P_total) > 1e-3 % Add tolerance
    fprintf('Sum of p_allocated before scaling: %.2f kW. Rescaling to match P_total = %.2f kW.\n', current_sum_p_allocated, P_total);
    p_allocated = p_allocated * (P_total / current_sum_p_allocated);
    % Re-check for max(0,...) or other individual constraints if scaling pushed values out of bounds.
    % For this example, we'll assume this simple scaling is sufficient.
    p_allocated = max(0, p_allocated); % Ensure non-negativity again after scaling
    % If scaling makes sum too low because some hit zero, another pass might be needed.
    % For simplicity, we stop here.
else
    fprintf('Sum of p_allocated after max(0,.): %.2f kW. Close enough to P_total = %.2f kW.\n', current_sum_p_allocated, P_total);
end


fprintf('\nFinal Allocations \n');
disp('EV # | Preferred (kW) | Allocated (kW) | Estimate of Mean Pref (kW)');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f | %20.2f\n', i, p_pref(i), p_allocated(i), estimated_mean_p_pref(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates: %.2f kW (Target: %.2f kW)\n', sum(p_allocated), P_total);

% Analytical optimal solution (assuming perfect knowledge, no non-negativity initially)
p_optimal_analytical_no_bounds = p_pref + (P_total - sum(p_pref)) / N;
% This analytical solution doesn't enforce p_i >= 0.
% True constrained optimum is harder to write in one line if bounds are active.

% Plot results
figure;
bar_data = [p_pref, p_allocated, p_optimal_analytical_no_bounds];
% Cap negative analytical results at 0 for fairer comparison in plot
bar_data(:,3) = max(0, bar_data(:,3)); 

b = bar(1:N, bar_data, 'grouped'); %#ok<*NASGU>
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title('Preferred vs. Allocated Charging Rates');
xticks(1:N);
legend('Preferred Rate', 'DGD Allocated Rate', 'Analytical Optimal', 'Location','best');
grid on;

if sum(p_pref) > P_total
    annotation('textbox', [0.55, 0.15, 0.1, 0.1],'FontSize', 8, 'String', ...
        sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates adjusted by DGD.', sum(p_pref), P_total), ...
        'FitBoxToText','on', 'BackgroundColor', 'yellow');
elseif sum(p_pref) < P_total
    annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'FontSize', 8, 'String',...
        sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates adjusted by DGD.', sum(p_pref), P_total), ...
        'FitBoxToText','on', 'BackgroundColor', 'cyan');
else
    annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'FontSize', 8, 'String',...
        sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total), ...
        'FitBoxToText','on', 'BackgroundColor', 'green');   
end

%% Solve with ADMM
%-----------------------------------------------------------------

% Communication Graph (Complete Graph for Simplicity) ---
% W_consensus is for averaging/summing steps within ADMM
W_consensus = ones(N) / N; 
% For DGD-style consensus on lambda, a doubly stochastic matrix is good.
% For simple averaging over a complete graph, 1/N works for each element.
% If graph is not complete, Metropolis-Hastings or similar needed.

% ADMM Parameters 
T_admm = 20;                   % Number of ADMM iterations (outer loop)
rho = 0.1;                      % ADMM penalty parameter / dual step size
beta = 0.05;                    % Proximal term coefficient for p_i update (0 for simpler update)

% Consensus Parameters (Inner Loops) 
T_consensus = 20;               % Iterations for inner consensus (sum_p and lambda)
alpha_consensus = 0.3;          % Stepsize for DGD within consensus routines

% Initialization 
p = p_pref;                     % Initial power allocations (can be p_pref, 0, or P_total/N)
% p = ones(N,1) * P_total/N;
lambda = zeros(N,1);            % Initial dual variables (price estimates)

% Store History 
Phist_admm = zeros(N, T_admm+1);
Lambdahist_admm = zeros(N, T_admm+1);
Sum_p_hist_admm = zeros(1, T_admm+1);

Phist_admm(:,1) = p;
Lambdahist_admm(:,1) = lambda;
Sum_p_hist_admm(1) = sum(p);

fprintf('Starting ADMM...\n');
% ADMM Algorithm 
for k = 1:T_admm
    % Step 1: Primal Update (Local p_i solve) 
    % p_i^{k+1} = max(0, (2*p_pref_i - lambda_i^k + beta*p_i^k) / (2 + beta))
    p_prev_iter = p; % Store p_i^k for the proximal term
    for i = 1:N
        numerator = 2 * p_pref(i) - lambda(i) + beta * p_prev_iter(i);
        denominator = 2 + beta;
        p(i) = max(0, numerator / denominator);
    end

    % Step 2a: Distributed Summation of p_i^{k+1} 
    % Each agent needs an estimate of sum(p). We use DGD to average p_i, then multiply by N.
    % Or, DGD to directly sum (more complex DGD variant). Simpler: average then scale.

    current_p_for_avg = p; % Values to be averaged
    avg_p_estimates = current_p_for_avg; % Initial estimate is local value

    % Store history of sum_p estimation for this ADMM iteration (optional, for debug)
    % AvgPHistInner = zeros(N, T_consensus+1); AvgPHistInner(:,1) = avg_p_estimates;

    for iter_c = 1:T_consensus
        % DGD for averaging: x_new = W_consensus * x_old - alpha_c * (x_old - values_to_average)
        % Here, we want x_old to converge to mean(values_to_average).
        % The standard DGD "x = Wx - alpha*(x-a)" makes x converge to mean(a).
        % So 'x' is avg_p_estimates, 'a' is current_p_for_avg
        % Using a simpler consensus averaging: new_estimates = W_consensus * old_estimates
        % This simple form requires W_consensus to be doubly stochastic and primitive.
        % For complete graph W_consensus = ones(N)/N, this works.
        avg_p_estimates = W_consensus * avg_p_estimates; % Iterative averaging
        % AvgPHistInner(:, iter_c+1) = avg_p_estimates;
    end
    % After T_consensus iterations, avg_p_estimates(i) is agent i's estimate of mean(p)
    % So, agent i's estimate of sum(p) is N * avg_p_estimates(i)
    estimated_sum_p_all_agents = N * avg_p_estimates; % Vector of N estimates, ideally all similar

    % For the lambda update, each agent uses its own estimate of the sum.
    % Or, we could pick one (e.g., agent 1's estimate) if we assume good consensus.
    % Let's use each agent's individual estimate for local lambda update.

    % Step 2b: Dual Variable Pre-Update (Local lambda_i update) 
    lambda_local_update = lambda + rho * (estimated_sum_p_all_agents - P_total);

    % Step 3: Consensus on Dual Variables (lambda_i) 
    % Agents run consensus on their lambda_local_update values.
    lambda_for_consensus = lambda_local_update;
    current_lambda_estimates = lambda_for_consensus; 

    % Store history of lambda estimation (optional)
    % LambdaHistInner = zeros(N, T_consensus+1); LambdaHistInner(:,1) = current_lambda_estimates;

    for iter_c = 1:T_consensus
        % Similar consensus averaging for lambda
        current_lambda_estimates = W_consensus * current_lambda_estimates;
        % LambdaHistInner(:, iter_c+1) = current_lambda_estimates;
    end
    lambda = current_lambda_estimates; % All lambda(i) should be close after consensus

    % Store history for ADMM iteration k 
    Phist_admm(:,k+1) = p;
    Lambdahist_admm(:,k+1) = lambda; % Store the consensus-agreed lambda
    Sum_p_hist_admm(k+1) = sum(p); % True sum for plotting global constraint satisfaction

    if mod(k, T_admm/10) == 0
        fprintf('ADMM Iter %d/%d: Sum P = %.2f (Target P_total = %.1f), Avg Lambda = %.2f\n', ...
                k, T_admm, sum(p), P_total, mean(lambda));
    end
end
fprintf('ADMM Finished.\n');

% Results 
p_allocated_admm = Phist_admm(:, T_admm+1);

fprintf('\nADMM Final Allocations \n');
disp('EV # | Preferred (kW) | Allocated (kW) | Final Lambda_i');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f | %14.4f\n', i, p_pref(i), p_allocated_admm(i), lambda(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates (ADMM): %.2f kW (Target: %.2f kW)\n', sum(p_allocated_admm), P_total);
fprintf('Final Lambda values std dev: %.4e (should be small if consensus worked)\n', std(lambda));


% Analytical Optimal Solution (for comparison) ---
% Solved via KKT conditions:
% L = sum_i (p_i - p_pref_i)^2 - nu (sum_i p_i - P_total) - sum_i mu_i p_i
% dL/dp_i = 2(p_i - p_pref_i) - nu - mu_i = 0
% If p_i > 0, then mu_i = 0 => p_i = p_pref_i + nu/2
% If p_i = 0, then mu_i >= 0 => 2(-p_pref_i) - nu - mu_i = 0 => nu <= -2*p_pref_i
% The single 'nu' here corresponds to '-lambda_star' from our ADMM.
% So, p_i_opt = p_pref_i - lambda_star/2
% We need to find lambda_star such that sum(max(0, p_pref_i - lambda_star/2)) = P_total
% This can be found by a 1D root search for lambda_star.
f_to_solve = @(l_star) sum(max(0, p_pref - l_star/2)) - P_total;
% Sensible search range for lambda_star:
% If all p_pref are high, lambda_star will be positive (to reduce allocations)
% If all p_pref are low, lambda_star will be negative (to increase allocations)
min_l = 2 * (min(p_pref) - max(p_pref) - P_total/N); % Heuristic lower bound
max_l = 2 * (max(p_pref) - min(p_pref) + P_total/N); % Heuristic upper bound
try
    lambda_star_optimal = fzero(f_to_solve, mean(lambda)); % Start search near ADMM's result
catch
    try 
        lambda_star_optimal = fzero(f_to_solve, 0); % Try starting at 0
    catch
        fprintf('fzero failed to find analytical lambda_star. Plotting without it.\n');
        lambda_star_optimal = NaN;
    end
end
p_optimal_analytical = max(0, p_pref - lambda_star_optimal/2);


% Plotting 
% 1. Convergence of allocations p_i
figure;
plot(0:T_admm, Phist_admm', 'LineWidth', 1.2);
xlabel('ADMM Iteration k');
ylabel('Allocated Power p_i^{(k)} (kW)');
title(sprintf('ADMM: EV Power Allocations (N=%d, rho=%.2f, beta=%.2f)', N, rho, beta));
legendStrings_p = arrayfun(@(i) sprintf('EV %d',i), 1:N, 'UniformOutput', false);
legend(legendStrings_p,'Location','best');
grid on;

% 2. Convergence of dual variables lambda_i
figure;
plot(0:T_admm, Lambdahist_admm', 'LineWidth', 1.2);
hold on;
if ~isnan(lambda_star_optimal)
    yline(lambda_star_optimal, 'k--', 'LineWidth', 2, 'Label', 'Optimal \lambda* (analytical)');
end
xlabel('ADMM Iteration k');
ylabel('Dual Variable \lambda_i^{(k)}');
title(sprintf('ADMM: Convergence of Dual Variables (Price Estimates)'));
legendStrings_l = arrayfun(@(i) sprintf('\\lambda_%d',i), 1:N, 'UniformOutput', false);
if ~isnan(lambda_star_optimal)
    legend([legendStrings_l, {'Optimal \lambda*'}],'Location','best');
else
    legend(legendStrings_l,'Location','best');
end
grid on;

% 3. Convergence of Sum p_i to P_total
figure;
plot(0:T_admm, Sum_p_hist_admm, 'b-', 'LineWidth', 1.5);
hold on;
yline(P_total, 'r--', 'LineWidth', 2);
xlabel('ADMM Iteration k');
ylabel('Total Allocated Power \Sigma p_i^{(k)} (kW)');
title('ADMM: Convergence of Total Power to P_{total}');
legend('Sum p_i^{(k)}', 'P_{total}', 'Location','best');
grid on; ylim([P_total*0.8, P_total*1.2]); % Adjust ylim if needed

% 4. Final Bar Chart Comparison
figure;
bar_data = [p_pref, p_allocated_admm];
legend_items = {'Preferred Rate', 'ADMM Allocated Rate'};
if ~isnan(lambda_star_optimal)
    bar_data = [bar_data, p_optimal_analytical];
    legend_items = [legend_items, {'Analytical Optimal'}];
end

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title('Preferred vs. Allocated Charging Rates (ADMM)');
xticks(1:N);
legend(legend_items, 'Location', 'best');
grid on;

if sum(p_pref) > P_total
    annotation_text = sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates adjusted by ADMM.', sum(p_pref), P_total);
    annotation_color = 'yellow';
elseif sum(p_pref) < P_total
    annotation_text = sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates adjusted by ADMM.', sum(p_pref), P_total);
    annotation_color = 'cyan';
else
    annotation_text = sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total);
    annotation_color = 'green';
end
annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'String', annotation_text, ...
        'FitBoxToText','on', 'BackgroundColor', annotation_color, 'FontSize', 8);

%% Solve with PSO
%-----------------------------------------------------------------

% PSO Parameters
nvars = N;                      % Number of optimization variables (power for each EV)

% Lower bounds (p_i >= 0)
lb = zeros(N,1);

% Upper bounds (0 <= p_i <= P_total theoretically, or max individual charge rate)
% For simplicity, let's use P_total as a loose upper bound for each.
% A tighter bound could be max(p_pref) or a physical EV charging limit.
ub = P_total * ones(N,1);
% ub = min(P_total, max(15, p_pref * 1.5)); % Example of slightly tighter UB

% Penalty coefficient for the sum constraint
penalty_coefficient = 1000; % This might need tuning

%  Objective Function for PSO (with penalty) 
% The input 'p' to this function will be a row vector (PSO standard)
objectiveFunction = @(p) sum((p' - p_pref).^2) + ...
                         penalty_coefficient * (sum(p) - P_total)^2;
% Note: p' is used because p_pref is a column vector.
% PSO passes 'p' as a row vector to the objective function.

%  PSO Options 
options = optimoptions('particleswarm', ...
    'SwarmSize', max(50, 10*nvars), ... % Number of particles
    'MaxIterations', 200*nvars, ...    % Max iterations
    'Display', 'iter', ...           % Show iteration progress
    'ObjectiveLimit', 1e-4, ...      % Stop if objective is good enough
    'FunctionTolerance', 1e-6, ...   % Stop if objective change is small
    'UseParallel', false);            % Set to true if you have Parallel Computing Toolbox
                                      % and problem is large enough.

fprintf('Starting PSO...\n');
%  Run PSO 
% particleswarm(objectiveFunction, nvars, lb, ub, options)
[p_optimal_pso_raw, fval_pso, exitflag, output_pso] = particleswarm(objectiveFunction, nvars, lb, ub, options);

fprintf('PSO Finished.\n');
fprintf('PSO Exit Flag: %d\n', exitflag);
disp(output_pso.message);
fprintf('Objective function value at solution: %.4e\n', fval_pso);
fprintf('Constraint term value: %.4e * (%.4f - %.1f)^2 = %.4e\n', ...
    penalty_coefficient, sum(p_optimal_pso_raw), P_total, ...
    penalty_coefficient * (sum(p_optimal_pso_raw) - P_total)^2);

%  Post-processing the PSO solution 
% PSO solution is a row vector, convert to column for consistency
p_allocated_pso = p_optimal_pso_raw';

% Check the sum constraint satisfaction
sum_p_pso = sum(p_allocated_pso);
fprintf('Sum of allocated rates (PSO raw): %.4f kW (Target: %.2f kW)\n', sum_p_pso, P_total);

% If the sum is not exactly P_total due to penalty approximation,
% we can apply a simple scaling as a heuristic, provided the solution is "close".
% This is common practice if the penalty method gets us near the feasible region.
if abs(sum_p_pso - P_total) > 1e-3 && sum_p_pso > 1e-6 % Add tolerance and check if sum is not zero
    fprintf('Normalizing PSO solution to meet sum constraint exactly.\n');
    p_allocated_pso_normalized = p_allocated_pso * (P_total / sum_p_pso);
    % Ensure non-negativity again after scaling (though unlikely to be an issue if lb was 0 and scaling factor > 0)
    p_allocated_pso_normalized = max(0, p_allocated_pso_normalized);
    % If this re-normalization drastically changes the solution or violates other complex constraints
    % (not present here), then the penalty method or PSO tuning needs more work.
else
    p_allocated_pso_normalized = p_allocated_pso;
end
% Re-check sum after potential normalization
sum_p_pso_normalized = sum(p_allocated_pso_normalized);

fprintf('\nPSO Final Allocations (after potential normalization)\n');
disp('EV # | Preferred (kW) | Allocated (kW)');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f\n', i, p_pref(i), p_allocated_pso_normalized(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates (PSO normalized): %.2f kW (Target: %.2f kW)\n', sum_p_pso_normalized, P_total);
fprintf('Original objective value (without penalty) for PSO solution: %.4f\n', sum((p_allocated_pso_normalized - p_pref).^2));


%  Analytical Optimal Solution (for comparison, same as in ADMM example) 
f_to_solve = @(lambda_star) sum(max(0, p_pref - lambda_star/2)) - P_total;
try
    lambda_star_optimal = fzero(f_to_solve, 0); 
    p_optimal_analytical = max(0, p_pref - lambda_star_optimal/2);
    analytical_available = true;
    fprintf('Analytical optimal objective value: %.4f\n', sum((p_optimal_analytical - p_pref).^2));
catch
    fprintf('fzero failed to find analytical lambda_star. Plotting without it.\n');
    p_optimal_analytical = nan(N,1);
    analytical_available = false;
end

%  Plotting 
figure;
bar_data = [p_pref, p_allocated_pso_normalized];
legend_items = {'Preferred Rate', 'PSO Allocated Rate'};
if analytical_available
    bar_data = [bar_data, p_optimal_analytical];
    legend_items = [legend_items, {'Analytical Optimal'}];
end

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title(sprintf('Preferred vs. Allocated Charging Rates (PSO, Penalty Coeff: %g)', penalty_coefficient));
xticks(1:N);
legend(legend_items, 'Location', 'best');
grid on;

if sum(p_pref) > P_total
    annotation_text = sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates adjusted by PSO.', sum(p_pref), P_total);
    annotation_color = 'yellow';
elseif sum(p_pref) < P_total
    annotation_text = sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates adjusted by PSO.', sum(p_pref), P_total);
    annotation_color = 'cyan';
else
    annotation_text = sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total);
    annotation_color = 'green';
end
annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'String', annotation_text, ...
        'FitBoxToText','on', 'BackgroundColor', annotation_color, 'FontSize', 8);

%  Convergence Plot (if you want to see PSO progress, requires custom output function) 
% To get iteration-wise best fitness for plotting convergence:
% 1. Define an output function:
%    function [state,options,optchanged] = psoIterDisplay(options,state,flag)
%        optchanged = false;
%        if strcmp(flag,'iter')
%            if ~isfield(state,'BestFitnessHistory')
%                state.BestFitnessHistory = [];
%            end
%            state.BestFitnessHistory = [state.BestFitnessHistory, state.BestFitness];
%        end
%    end
% 2. Add 'OutputFcn', @psoIterDisplay to optimoptions
% 3. After running PSO:
%    figure;
%    plot(output_pso.BestFitnessHistory); % Assuming BestFitnessHistory was populated
%    xlabel('Iteration'); ylabel('Best Objective Value (incl. penalty)');
%    title('PSO Convergence'); grid on;
% Note: The standard 'Display','iter' option gives some info, but for a plot,
% a custom output function or capturing iteration data is needed.
% The 'output_pso' struct doesn't store the full history by default.
% Let's try to use a built-in plot function if available or make a simple one.
if isfield(options, 'PlotFcn') && ~isempty(options.PlotFcn)
    % If a plot function was specified and ran, figures might already exist.
    % For example, 'PlotFcns', @psoplotswarm shows particle movement.
    % @psoplotbestf shows best fitness.
    fprintf("If 'PlotFcns' was used in options, relevant plots might already be displayed.\n")
else
    % If no PlotFcn was used, we can try to plot the objective value from fval_pso
    % (which is only the final value). To get a convergence plot, you'd typically
    % need to set options.PlotFcns = @psoplotbestf;
    fprintf("To see PSO convergence plot, set options.PlotFcns = @psoplotbestf;\n");
    fprintf("Example: options = optimoptions(options, 'PlotFcns', @psoplotbestf);\n");
end

%% Distributed Resource Allocation for EV Charging via Genetic Algorithm (GA)
%-----------------------------------------------------------------

% GA Parameters
nvars = N;                      % Number of optimization variables (power for each EV)

% Objective function: GA passes 'x' as a row vector.
% p_pref is a column vector.
objectiveFunction = @(p_row) sum((p_row' - p_pref).^2);

% Bound constraints
lb = zeros(1, nvars);           % Lower bounds (p_i >= 0), GA expects row vector
% Upper bounds (0 <= p_i <= P_total, or max individual charge rate)
ub = P_total * ones(1, nvars);  % GA expects row vector
% ub = min(P_total, max(15, p_pref' * 1.5)); % Example of slightly tighter UB (ensure row vector)


% Linear equality constraint: sum(p_i) = P_total
Aeq = ones(1, nvars);           % Coefficients for p1 + p2 + ... + pN
beq = P_total;                  % The sum should equal P_total

% No linear inequality constraints
A = [];
b = [];

% No nonlinear constraints function
nonlcon = [];

% No integer constraints in this version
IntCon = [];

%  GA Options 
options = optimoptions('ga', ...
    'PopulationSize', max(50, 10*nvars), ... % Number of individuals in population
    'MaxGenerations', 50 * nvars / 10, ...  % Max number of generations (scaled by nvars)
    'Display', 'iter', ...                 % Show iteration progress
    'PlotFcn', {@gaplotbestf, @gaplotstopping}, ... % Plot best fitness and stopping criteria
    'FunctionTolerance', 1e-7, ...         % Stop if objective change is small
    'ConstraintTolerance', 1e-7, ...       % Tolerance for constraint satisfaction
    'UseParallel', false);                  % Set to true for parallel computation

fprintf('Starting GA...\n');
%  Run GA 
[p_optimal_ga_row, fval_ga, exitflag_ga, output_ga, population_ga, scores_ga] = ...
    ga(objectiveFunction, nvars, A, b, Aeq, beq, lb, ub, nonlcon, IntCon, options);

fprintf('GA Finished.\n');
fprintf('GA Exit Flag: %d\n', exitflag_ga);
disp(output_ga.message);
fprintf('Objective function value at solution: %.4e\n', fval_ga);

%  Post-processing the GA solution 
% GA solution is a row vector, convert to column for consistency
p_allocated_ga = p_optimal_ga_row';

% Check the sum constraint satisfaction (should be very good)
sum_p_ga = sum(p_allocated_ga);
fprintf('Sum of allocated rates (GA): %.4f kW (Target: %.2f kW)\n', sum_p_ga, P_total);
fprintf('Deviation from sum constraint: %.4e\n', sum_p_ga - P_total);


fprintf('\nGA Final Allocations\n');
disp('EV # | Preferred (kW) | Allocated (kW)');
for i=1:N
    fprintf('%4d | %14.2f | %14.2f\n', i, p_pref(i), p_allocated_ga(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf('Sum of Preferred Rates: %.2f kW\n', sum(p_pref));
fprintf('Sum of Allocated Rates (GA): %.2f kW (Target: %.2f kW)\n', sum(p_allocated_ga), P_total);
fprintf('Objective value (sum of squared differences) for GA solution: %.4f\n', fval_ga);


%  Analytical Optimal Solution (for comparison, same as in ADMM/PSO example) 
% Solved via KKT conditions: p_i_opt = max(0, p_pref_i - lambda_star/2)
% Find lambda_star such that sum(max(0, p_pref_i - lambda_star/2)) = P_total
f_to_solve = @(lambda_s) sum(max(0, p_pref - lambda_s/2)) - P_total;
try
    % Provide a search interval for fzero if needed, e.g., based on mean(lambda) from ADMM
    % or a wider range.
    lambda_star_optimal = fzero(f_to_solve, [-100, 100]); % Search in a reasonable range
    if isnan(lambda_star_optimal) || abs(f_to_solve(lambda_star_optimal)) > 1e-3
        lambda_star_optimal = fzero(f_to_solve, 0); % Try another starting point if first fails
    end
    p_optimal_analytical = max(0, p_pref - lambda_star_optimal/2);
    analytical_available = true;
    fprintf('Analytical optimal objective value: %.4f\n', sum((p_optimal_analytical - p_pref).^2));
    fprintf('Analytical sum: %.4f\n', sum(p_optimal_analytical));

catch ME
    fprintf('fzero failed to find analytical lambda_star: %s. Plotting without it.\n', ME.message);
    p_optimal_analytical = nan(N,1);
    analytical_available = false;
end


%  Plotting 
figure; % New figure for the bar chart
bar_data = [p_pref, p_allocated_ga];
legend_items = {'Preferred Rate', 'GA Allocated Rate'};
if analytical_available && ~any(isnan(p_optimal_analytical))
    bar_data = [bar_data, p_optimal_analytical];
    legend_items = [legend_items, {'Analytical Optimal'}];
end

b = bar(1:N, bar_data, 'grouped');
xlabel('Electric Vehicle (Agent)');
ylabel('Charging Rate (kW)');
title(sprintf('Preferred vs. Allocated Charging Rates (GA, N=%d)', N));
xticks(1:N);
legend(legend_items, 'Location', 'northwest');
grid on;

if sum(p_pref) > P_total
    annotation_text = sprintf('Demand (%.1fkW) > Supply (%.1fkW)\nRates adjusted by GA.', sum(p_pref), P_total);
    annotation_color = 'yellow';
elseif sum(p_pref) < P_total
    annotation_text = sprintf('Demand (%.1fkW) < Supply (%.1fkW)\nRates adjusted by GA.', sum(p_pref), P_total);
    annotation_color = 'cyan';
else
    annotation_text = sprintf('Demand (%.1fkW) == Supply (%.1fkW)', sum(p_pref), P_total);
    annotation_color = 'green';
end
annotation('textbox', [0.55, 0.15, 0.1, 0.1], 'String', annotation_text, ...
        'FitBoxToText','on', 'BackgroundColor', annotation_color, 'FontSize', 8);

% GA plots (Best Fitness and Stopping Criteria) would have been generated by PlotFcn.
% We can ensure they are displayed by pausing or by managing figure handles if needed.
% For a script, they usually appear automatically.
disp('GA plots for best fitness and stopping criteria should have been displayed in separate figures.');