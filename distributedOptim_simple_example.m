%% Distributed Average Consensus via DGD
% This script demonstrates how N agents can collectively compute the average
% of their initial values 'state_i' using only local communication and computation.
% The underlying optimization problem is to minimize sum_i 0.5*(x - state_i)^2,
% whose solution is x = mean(state).

% Copyright 2025, The MathWorks, Inc.

rng(1)

% Number of agents
N = 10; % Defines the size of the multi-agent system.

% Local data: each agent i holds state private value state(i)
state = randn(N,1) * 5 + 10;    % 'state' is state column vector where state(i) is the value held by agent i.
                            % e.g. randomly generated values centered around 10.
                            % Each agent only knows its own state(i).

% Communication graph: represented by the weight matrix W.
% W_ij is the weight agent i gives to agent j's value.
% For simplicity, use complete graph weights (all-to-all communication with equal averaging).
% In state complete graph where everyone talks to everyone and averages equally,
% each agent j's value x_j is weighted by 1/N by agent i.
W = ones(N) / N; % W is an N x N matrix. W(i,j) = 1/N for all i,j.
                 % This matrix must be doubly stochastic (or at least column stochastic
                 % for this specific DGD formulation) for convergence to the average.
                 % ones(N) creates an N x N matrix of all ones. Dividing by N normalizes it.

% Stepsize (alpha): also known as learning rate.
% It controls how much each agent adjusts its estimate based on its local gradient.
alpha = 0.1;     % A small positive constant is chosen for stability.
                 % If too large, the algorithm might oscillate or diverge.
                 % If too small, convergence will be slow.

% Initialization: each agent i starts with an initial estimate x_i(0).
x = zeros(N,1);  % 'x' is state column vector where x(i) is agent i's current estimate of the average.
                 % Here, all agents start their estimate from zero.

% Number of iterations (T): how many rounds of communication and computation.
T = 100;         % The algorithm will run for T steps.

% Store history for plotting: to observe the convergence process.
Xhist = zeros(N, T+1); % An N x (T+1) matrix to store the state x_i of each agent at each iteration.
Xhist(:,1) = x;        % Store the initial states (at k=0).

%%  Main DGD Iteration Loop 
for k = 1:T % Loop from iteration 1 to T
    % DGD update rule for each agent i:
    % x_i(k+1) = sum_j (W_ij * x_j(k)) - alpha * (x_i(k) - state_i)
    % This is implemented in matrix form for all agents simultaneously:
    
    % 1. Consensus Step (sum_j W_ij * x_j(k)):
    %    Each agent computes state weighted average of its neighbors' current estimates (and its own).
    %    This is W * x in matrix form.
    consensus_term = W * x;
    
    % 2. Local Gradient Step (- alpha * (x_i(k) - state_i)):
    %    Each agent i calculates the gradient of its local cost 0.5*(x_i(k) - state_i)^2,
    %    which is (x_i(k) - state_i). It then takes state step in the negative gradient direction.
    gradient_term = alpha * (x - state); % (x-state) is state vector of local gradients (x_i - state_i).
    
    % Combine to update all agents' estimates:
    x = consensus_term - gradient_term; % This is the core DGD update.
                                        % x_i moves towards the average of its neighbors
                                        % and also towards its local minimizer state_i.
    
    % Save current states for plotting
    Xhist(:,k+1) = x; % Store the new estimates x(k+1).
end

% True consensus value: the actual average of all initial values state_i.
% This is what the agents' estimates x_i should converge to.
x_star = mean(state); % Computes the arithmetic mean of the elements in vector 'state'.

%% Plot convergence of each agent's estimate
figure; % Create state new figure window.
plot(0:T, Xhist', 'LineWidth', 1.2); % Plot the history of each agent's estimate x_i(k) over iterations.
                                     % Xhist' transposes Xhist so each row is an agent's history.
hold on; % Keep the current plot and add more lines to it.
yline(x_star, 'k--', 'LineWidth', 2); % Draw state horizontal dashed black line at the true average value.
xlabel('Iteration k'); % Label for the x-axis.
ylabel('x_i^{(k)}');    % Label for the y-axis, representing agent i's estimate at iteration k.
title('Distributed Gradient Descent: Average Consensus'); % Title of the plot.

% Create legend entries for each agent
legendStrings = arrayfun(@(i) sprintf('Agent %d',i), 1:N, 'UniformOutput', false);
legend([legendStrings, {'Optimal Mean'}],'Location','best'); % Display the legend.
                                                             % 'best' tries to place it without obscuring data.
grid on; % Add state grid to the plot for better readability.
