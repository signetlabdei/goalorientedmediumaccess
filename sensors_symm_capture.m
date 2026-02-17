%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                      script: sensors_symm_capture                       %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation with a symmetric sensor network using     %
% chi-squared value distributions as a function of the capture            %
% probability                                                             %
%                                                                         %
% Parameters:                                                             %
% -N:           the  number of nodes [scalar, int]                        %
% -M:           the number of steps to simulate [scalar, int]             %
% -epsilon:     the NE approximation error [scalar, R+]                   %
% -delta:       the VoI quantization step [scalar, R+]                    %
% -Vmax:        the maximum possible VoI [scalar, R+]                     %
% -psi:         the transmission attempt cost [scalar, R+]                %
% -max_iter:    the maximum number of IBR iterations [scalar, int]        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clearvars

% Simulation parameters
N = 10;
M = 1e6;
epsilon = 1e-3;
delta = 1e-4;
Vmax = 50;
psi = 0;
capture = 0 : 0.01 : 0.25;
max_iter = 1000;

% Success probability
success = zeros(1, N);
success(1) = 1;
L = length(capture);

% Chi-squared distribution
values = 0 : delta : Vmax;
cdf = [0, 1 - exp(-(values + delta / 2))];

% Auxiliary variables
pull_rewards = zeros(L, 1);
pull_vois = zeros(L, 1);
pull_energies = zeros(L, 1);
pull_fairness = zeros(L, 1);
th_rewards = zeros(L, 1);
th_energies = zeros(L, 1);
th_fairness = zeros(L, 1);
mc_rewards = zeros(L, 1);
mc_vois = zeros(L, 1);
mc_energies = zeros(L, 1);
cap_rewards = zeros(L, 1);
cap_vois = zeros(L, 1);
cap_energies = zeros(L, 1);
values = 0 : delta : 50;

% Compute pull-based solution
nb = 1;
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

cdf = repmat(cdf, N, 1);


% Pre-compute transmission values
tx_values = zeros(size(cdf));
for n = 1 : N
    pmf = diff(cdf(n, :));
    prob_val = pmf .* values;
    tx_values(n, 1 : length(values)) = flip(cumsum(flip(prob_val)));
    tx_values(n, :) = tx_values(n, :) ./ (1 - cdf(n, :));
end


% Compute capture LIBRA solution
[v_eq, voi_0, initial_zero_thresholds] = equal_value_initialization(cdf, values, psi, success);
[zero_thresholds, reward_iter] = iterated_best_response(cdf, psi, epsilon, success, values, initial_zero_thresholds, max_iter);

% Run LIBRA
for ell = 1 : L

    success(2) = capture(ell);
    ell, success(2)

    % Copy pull-based performance
    pull_rewards(ell) = pull_rew;
    pull_vois(ell) = pull_val;
    pull_energies(ell) = pull_tx;
    pull_fairness(ell) = 1 / N;

    % Determine LIBRA solution
    [v_eq, voi_0, initial_thresholds] = equal_value_initialization(cdf, values, psi, success);
    [thresholds, reward_iter] = iterated_best_response(cdf, psi, epsilon, success, values, initial_thresholds, max_iter);
    reward = max(reward_iter);
    th_rewards(ell) = reward;
    th_energies(ell) = sum(1 - thresholds);
    % Compute fairness
    th_fairness(ell) = (sum(1 - thresholds)) ^ 2 / N / sum((1 - thresholds) .^ 2);

    % Monte Carlo check
    [mc_voi, mc_reward, energy, goodput, channel_use, ~] = montecarlo(cdf, values, psi, success, thresholds, M);
    mc_rewards(ell) = mc_reward;
    mc_vois(ell) = mc_voi;
    mc_energies(ell) = sum(energy);

    % Collision LIBRA
    % Compute theoretical value
    tx_num = find(success > 0);
    success = [success, 0];
    for tx = tx_num
        % Tx transmitters (consider possible combinations)
        possible_tx = nchoosek(1 : N, tx);
        for comb = 1 : size(possible_tx, 1)
            prob_vec = zero_thresholds;
            for m = 1 : N
                if (any(possible_tx(comb, :) == m))
                    prob_vec(m) = 1 - zero_thresholds(m);
                end
            end
            prob_comb = prod(prob_vec);
            for m = possible_tx(comb, :)
                threshold_index = find(cdf(m, :) >= zero_thresholds(m), 1);
                if (~isempty(threshold_index))
                    cap_rewards(ell) = cap_rewards(ell) + prob_comb * success(tx) * tx_values(m, threshold_index);
                end
            end
        end
    end
    for m = 1 : N
        cap_rewards(ell) = cap_rewards(ell) - psi * (1 - zero_thresholds(m));
    end
    [mc_voi, mc_reward, energy, goodput, channel_use, ~] = montecarlo(cdf, values, psi, success, zero_thresholds, M);
    cap_vois(ell) = mc_voi;
    cap_energies(ell) = sum(1 - zero_thresholds);
end