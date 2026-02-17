function [voi, reward, energy, goodput, channel_use, reward_history] = montecarlo(cdf, values, psi, success, thresholds, M)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                          function: montecarlo                           %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation using the provided transmission VoI       %
% thresholds                                                              %
%                                                                         %
% Inputs:                                                                 %
% -cdf:             the CDF of values for each node [N x V]               %
% -values:          the possible values for all nodes [1 x V]             %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -success:         the success probability vector [1 x N]                %
% -thresholds:      the VoI quantile thresholds for transmission [1 x N]  %
% -M:               the number of steps to simulate [scalar, int]         %
%                                                                         %
% Outputs:                                                                %
% -voi:             the average VoI over the simulation [scalar]          %
% -reward:          the average reward over the simulation [scalar]       %
% -energy:          the energy use per node over the simulation [1 x N]   %
% -channel_use:     the channel use over the simulation [scalar, 0-1]     %
% -reward_history:  the step-by-step reward over the simulation [1 x M]   %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Utility variables
N = size(cdf, 1);
voi = 0;
goodput = 0;
channel_use = 0;
energy = zeros(1, N);
reward_history = zeros(1, M);


% Translate quantiles to actual values
value_thresholds = ones(1, N) * max(values);
for n = 1 : N
    threshold_value = find(cdf(n, :) >= thresholds(n), 1);
    if (~isempty(threshold_value))
        value_thresholds(n) = values(threshold_value);
    end
end

% Generate observed values for each node
sampled_values = zeros(N, M);
for n = 1 : N
    sampled_values(n, :) = datasample(values, M, 'Replace', true, 'Weights', diff(cdf(n, :)));
end

% Simulate the scenario
for trial = 1 : M
    tx = sampled_values(:, trial) > value_thresholds';
    % Nodes transmit
    if (sum(tx) > 0)
        success_prob = success(sum(tx));
        successful = tx & (rand(N, 1) <= success_prob);
        if (sum(successful) > 0)
            reward_history(trial) = sum(sampled_values(successful, trial));
            voi = voi + reward_history(trial) / M;
            goodput = goodput + sum(successful) / M;
        end
    end
    % There is at least one transmission
    if (sum(tx) > 0)
        channel_use = channel_use + 1 / M;
        energy = energy + tx' / M;
        reward_history(trial) = reward_history(trial) - sum(tx) * psi;
    end
end

reward = mean(reward_history);