%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                       script: sensors_asymm_cdf                         %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation with random asymmetric sensor networks    %
% using chi-squared value distributions over K parallel channels          %
%                                                                         %
% Parameters:                                                             %
% -N:           the  number of nodes [scalar, int]                        %
% -T:           the number of network realizations [scalar, int]          %
% -M:           the number of steps to simulate [scalar, int]             %
% -epsilon:     the NE approximation error [scalar, R+]                   %
% -delta:       the VoI quantization step [scalar, R+]                    %
% -Vmax:        the maximum possible VoI [scalar, R+]                     %
% -psi:         the transmission attempt cost [scalar, R+]                %
% -sigma:       the variation in the average VoI [scalar, R+]             %
% -max_iter:    the maximum number of IBR iterations [scalar, int]        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clearvars

% Simulation parameters
N = 10;
T = 100;
M = 1e6;
epsilon = 1e-3;
delta = 1e-4;
Vmax = 50;
psi = 0.25;
sigma = 0.5;
max_iter = 1000;

% Success probability
success = zeros(1, N);
success(1) = 1;

% Auxiliary variables
all_mus = zeros(T, N);
exp_reward = zeros(T, N);
pull_rewards = zeros(1, T);
pull_vois = zeros(1, T);
pull_channel_uses = zeros(1, T);
pull_energies = zeros(T, N);
pull_goodputs = zeros(1, T);
pull_thresholds = ones(T, N);
veq_thresholds = ones(T, N);
th_thresholds = zeros(T, N);
th_rewards = zeros(1, T);
th_energies = zeros(T, N);
mc_rewards = zeros(1, T);
mc_vois = zeros(1, T);
mc_energies = zeros(T, N);
mc_channel_uses = zeros(1, T);
mc_goodputs = zeros(1, T);
values = 0 : delta : 50;

% Run LIBRA
for t = 1 : T
    t
    % Generate random realization
    mus = ones(1, N) + (rand(1, N) - 0.5) * 2 * sigma;
    all_mus(t, :) = mus;
    % Compute expected rewards and VoI CDFs for all nodes
    exp_reward(t, :) = mus;
    cdf = zeros(N, length(values) + 1);
    for n = 1 : N
        cdf(n, :) = [0, 1 - exp(-(values + delta / 2) / mus(n))];
    end
    % Compute pull-based solution
    [~, nb] = max(mus);
    exp_values = diff(cdf(nb(1), :)) * values';
    pull_rew = 0;
    pull_val = 0;
    pull_tx = 0;
    for v = 1 : length(values)
        if (v > 1)
            exp_values = exp_values - values(v) * (cdf(nb(1), v) - cdf(nb(1), v - 1));
        end
        v_thr = values(v) - delta / 2;
        v_rew = exp_values - psi * (1 - cdf(nb(1), v));
        if (v_rew > pull_rew)
            pull_tx = 1 - cdf(nb(1), v);
            pull_rew = v_rew;
            pull_val = exp_values;
        end
    end

    % Compute pull-based performance
    pull_thresholds(t, nb) = 1 - pull_tx;
    pull_rewards(t) = pull_rew;
    pull_vois(t) = pull_val;
    pull_channel_uses(t) = pull_tx;
    pull_energies(t, nb(1)) = pull_tx;
    pull_goodputs(t) = pull_tx;

    % Determine LIBRA solution
    [v_eq, voi_0, initial_thresholds] = equal_value_initialization(cdf, values, psi, success);
    [thresholds, reward_iter] = iterated_best_response(cdf, psi, epsilon,success, values, initial_thresholds, max_iter);
    reward = max(reward_iter);
    th_rewards(t) = reward;
    th_energies(t, :) = 1 - thresholds;
    veq_thresholds(t, :) = initial_thresholds;
    th_thresholds(t, :) = thresholds;
    % Monte Carlo check
    [mc_voi, mc_reward, energy, goodput, channel_use, ~] = montecarlo(cdf, values, psi, success, thresholds, M);
    mc_rewards(t) = mc_reward;
    mc_vois(t) = mc_voi;
    mc_energies(t, :) = energy;
    mc_channel_uses(t) = channel_use;
    mc_goodputs(t) = goodput;
end