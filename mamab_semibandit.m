function [final_policy, voi, reward, energy, goodput, channel_use, reward_history, policy_history] = mamab_semibandit(cdf, values, thetas, success, psi, epsilon, kappa, memory, M)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                       function: mamab_semibandit                        %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs the BETA MAMAB training using semi-bandit estimation (collision    %
% channel only)                                                           %
%                                                                         %
% Inputs:                                                                 %
% -cdf:             the CDF of values for each node [N x V]               %
% -values:          the possible values for all nodes [1 x V]             %
% -thetas:          the possible BETA thresholds [1 x T]                  %
% -success:         the success probability vector [1 x N]                %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -epsilon:         exploration rate of epsilon-hedge [scalar, 0-1]       %
% -kappa:           learning rate exponential factor [scalar, 0-1]        %
% -memory:          steps to estimate semi-bandit reward [scalar, int]    %
% -M:               number of training steps [scalar, int]                %
%                                                                         %
% Outputs:                                                                %
% -final_policy:    the final policy of the MAMAB system [1 x N]          %
% -voi:             the average VoI over training [scalar]                %
% -reward:          the average reward over training [scalar]             %
% -energy:          the energy use for each node over training [1 x N]    %
% -goodput:         the goodput over training [scalar, 0-1]               %
% -channel_use:     the channel use fraction over training [scalar, 0-1]  %
% -reward_history:  the step-by-step reward over training [1 x M]         %
% -policy_history:  the step-by-step policy over training [N x M]         %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Utility variables
N = size(cdf, 1);
T = length(thetas);
energy = zeros(1, N);
reward_history = zeros(1, M);
policy_history = zeros(N, M);
exp_values = zeros(N, T);
alphas = ones(memory, N);
betas = zeros(memory, N);
lambdas = zeros(memory, N);
rhos = zeros(memory, N);
voi = 0;
goodput = 0;
channel_use = 0;

% Generate observed values for each node
sampled_values = zeros(N, M);
for n = 1 : N
    sampled_values(n, :) = datasample(values, M, 'Replace', true, 'Weights', diff(cdf(n, :)));
end

% Simulate the scenario
for trial = 1 : M
    % Run epsilon hedge and generate transmissions
    [theta_thr, greedy] = epsilon_hedge(thetas, exp_values, epsilon);
    policy_history(:, trial) = thetas(greedy);
    tx = zeros(1, N);
    outcome = 0;
    tx = sampled_values(:, trial) > theta_thr';
    % Nodes transmit
    if (sum(tx) > 0)
        success_prob = success(sum(tx));
        outcome = find(tx & (rand(N, 1) <= success_prob));
        if (~isempty(outcome))
            reward_history(trial) = sum(sampled_values(outcome, trial));
            voi = voi + reward_history(trial) / M;
            goodput = goodput + sum(outcome) / M;
        else
            outcome = -1;
        end
    end
    % There is at least one transmission
    if (sum(tx) > 0)
        channel_use = channel_use + 1 / M;
        energy = energy + tx' / M;
        reward_history(trial) = reward_history(trial) - sum(tx) * psi;
    end

    % Compute estimates
    alphas(2 : end, :) = alphas(1 : end - 1, :);
    rhos(2 : end, :) = rhos(1 : end - 1, :);
    for n = 1 : N
        alphas(1, n) = sum(tx) > tx(n);
        if (tx(n) == 0)
            betas(2 : end, n) = betas(1 : end - 1, n);
            betas(1, n) = outcome(1) > 0;
            if (alphas(1, n) > 0)
                lambdas(2 : end, n) = lambdas(1 : end - 1, n);
                if (outcome == -1)
                    lambdas(1, n) = 0;
                else
                    lambdas(1, n) = mean(sampled_values(outcome, trial));
                end
            end
        end
    end
    rhos(1, :) = tx;
    success_values = sampled_values(:, trial);
    % Update epsilon hedge
    if (trial > memory)
        exp_values = semibandit_update(exp_values, success, thetas, kappa ^ (trial - memory), psi, success_values, tx, outcome, mean(alphas), mean(betas), mean(lambdas), mean(rhos));
    end
end


[~, final_idx] = max(exp_values');

final_policy = thetas(final_idx);

reward = mean(reward_history);