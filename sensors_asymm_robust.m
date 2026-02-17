%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         script: sensors_asymm                           %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation with random asymmetric sensor networks    %
% using chi-squared value distributions and an estimation error over the  %
% mean value of each node (individual, i.e., independent errors for each  %
% node, and synchronized, i.e., the same error for all nodes)             %
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
% -nu:          the variation in the average VoI estimate [scalar, R+]    %       
% -max_iter:    the maximum number of IBR iterations [scalar, int]        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all
clearvars


% Simulation parameters
N = 10;
T = 200;
M = 1e6;
epsilon = 1e-3;
delta = 1e-4;
Vmax = 50;
psi = 0.25;
sigma = 0;
nu = 0.25;
max_iter = 1000;

% Success probability calculation
success = zeros(1, N);
success(1) = 1;


% Utilivty variables
values = 0 : delta : Vmax;
pull_rewards = zeros(1, T);
th_rewards = zeros(1, T);
syn_rewards = zeros(1, T);
ind_rewards = zeros(1, T);


% Run LIBRA
for t = 1 : T
    % Compute real and estimated CDF (individual and synchronized errors)
    mus = ones(1, N) + (rand(1, N) - 0.5) * 2 * sigma;
    syn_mus = mus + (rand(1, N) - 0.5) * 2 * nu;
    ind_mus = ones(N);
    for n = 1 : N
        ind_mus(n,:) = mus + (rand(1, N) - 0.5) * 2 * nu;
    end
    cdf = zeros(N, length(values) + 1);
    syn_cdf = zeros(N, length(values) + 1);
    ind_cdf = zeros(N, N, length(values) + 1);
    for n = 1 : N
        cdf(n, :) = [0, 1 - exp(-(values + delta / 2) / mus(n))];
        syn_cdf(n, :) = [0, 1 - exp(-(values + delta / 2) / syn_mus(n))];
        for m = 1 : N
            ind_cdf(m, n, :) = [0, 1 - exp(-(values + delta / 2) / ind_mus(m, n))];
        end
    end    


    % Compute pull-based solution
    [~, nb] = max(mus);
    exp_values = diff(cdf(nb(1), :)) * values';
    pull_rew = 0;
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
        end
    end
    pull_rewards(t) = pull_rew;

    % Real solution with LIBRA
    [~, ~, initial_thresholds] = equal_value_initialization(syn_cdf, values, psi, success);
    [~, iter_rewards] = iterated_best_response(cdf, psi, epsilon, success, values, initial_thresholds, max_iter);
    th_rewards(t) = max(iter_rewards);
    % Noisy (synchronized) solution with LIBRA
    [~, ~, syn_initial_thresholds] = equal_value_initialization(syn_cdf, values, psi, success);
    [est_thresholds] = iterated_best_response(syn_cdf, psi, epsilon, success, values, syn_initial_thresholds, max_iter);
    threshold_indices = ones(1, N) * 1e9;
    syn_thresholds = ones(1, N);
    for m = 1 : N
        threshold_index = find(cdf(m, :) >= est_thresholds(m), 1);
        if (~isempty(threshold_index))
            threshold_indices(m) = threshold_index;
            syn_thresholds(m) = cdf(m, threshold_index);
        end
    end
    [~, mc_reward, ~, ~, ~, ~] = montecarlo(cdf, values, psi, success, syn_thresholds, M);
    syn_rewards(t) = mc_reward;

    % Noisy (individual) solution with LIBRA
    ind_thresholds = ones(1, N);
    for m = 1 : N
        [~, ~, ind_initial_thresholds] = equal_value_initialization(squeeze(ind_cdf(m, :, :)), values, psi, success);
        [est_thresholds] = iterated_best_response(squeeze(ind_cdf(m, :, :)), psi, epsilon, success, values, syn_initial_thresholds, max_iter);
        threshold_index = find(cdf(m, :) >= est_thresholds(m), 1);
        if (~isempty(threshold_index))
            ind_thresholds(m) = cdf(m, threshold_index);
        end
    end
    [~, mc_reward, ~, ~, ~, ~] = montecarlo(cdf, values, psi, success, ind_thresholds, M);
    ind_rewards(t) = mc_reward;

end
