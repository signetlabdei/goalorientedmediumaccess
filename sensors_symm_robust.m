%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                       script: sensors_symm_robust                       %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation with random asymmetric sensor network     %
% using chi-squared value distributions and a symmetric belief (i.e., all %
% sensors believe they have the same mean)                                %
%                                                                         %
% Parameters:                                                             %
% -N:           the  number of nodes [scalar, int]                        %
% -T:           the number of network realizations [scalar, int]          %
% -M:           the number of steps to simulate [scalar, int]             %
% -epsilon:     the NE approximation error [scalar, R+]                   %
% -delta:       the VoI quantization step [scalar, R+]                    %
% -Vmax:        the maximum possible VoI [scalar, R+]                     %
% -psi:         the transmission attempt cost [scalar, R+]                %
% -nus:          the variation in the average VoI estimate [1 x L, R+]    %       
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
sigma = 0;
nus = 0: 0.05: 0.5;
max_iter = 1000;

% Success probability calculation
success = zeros(1, N);
success(1) = 1;


% Utilivty variables
L = length(nus);
values = 0 : delta : Vmax;
pull_rewards = zeros(L, T);
th_rewards = zeros(L, T);
syn_rewards = zeros(L, T);


% Run LIBRA
for ell = 1 : L
    nu = nus(ell);
    for t = 1 : T
        nu, t
        % Compute real and estimated CDF (individual and synchronized errors)
        syn_mus = ones(1, N);    
        mus = syn_mus + (rand(1, N) - 0.5) * 2 * nu;
        cdf = zeros(N, length(values) + 1);
        syn_cdf = zeros(N, length(values) + 1);
        for n = 1 : N
            cdf(n, :) = [0, 1 - exp(-(values + delta / 2) / mus(n))];
            syn_cdf(n, :) = [0, 1 - exp(-(values + delta / 2) / syn_mus(n))];
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
        pull_rewards(ell, t) = pull_rew;
    
        % Real solution with LIBRA
        [~, ~, initial_thresholds] = equal_value_initialization(cdf, values, psi, success);
        [~, iter_rewards] = iterated_best_response(cdf, psi, epsilon, success, values, initial_thresholds, max_iter);
        th_rewards(ell, t) = max(iter_rewards);
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
        syn_rewards(ell, t) = mc_reward;
    end

end
