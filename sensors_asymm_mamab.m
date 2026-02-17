%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                      script: sensors_asymm_mamab                        %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation with random asymmetric sensor networks    %
% using chi-squared value distributions, comparing BETA and LIBRA over a  %
% collision channel                                                       %
%                                                                         %
% Simulation parameters:                                                  %
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
% Beta parameters:                                                        %
% -thetas:      the possible BETA thresholds [1 x V]                      %
% -exploration: exploration rate of epsilon-hedge [scalar, 0-1]           %
% -memory:      steps to estimate the semi-bandit reward [scalar, int]    %
% -kappa:       learning rate exponential factor [scalar, 0-1]            %
% -mab_steps:   number of training steps [scalar, int]                    %
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
psi = 0.25;
sigma = 0.5;
max_iterations = 1000;
values = 0 : delta : 50;

% BETA parameters
thetas = 0 : 0.1 : 20;
exploration = 0.01;
memory = 25;
kappa = 0.99995;
mab_steps = 25000;

% Success probability
success = zeros(1, N);
success(1) = 1;
success(2) = 0.1;

% Auxiliary variables
exp_reward = zeros(T, N);
pull_rewards = zeros(1, T);
pull_channel = zeros(1, T);
pull_energies = zeros(T, N);
pull_goodputs = zeros(1, T);
pull_thresholds = ones(T, N);
veq_thresholds = ones(T, N);
th_thresholds = zeros(T, N);
th_rewards = zeros(1, T);
th_energies = zeros(T, N);
mc_rewards = zeros(1, T);
mc_energies = zeros(T, N);
mc_channel_uses = zeros(1, T);
mc_goodputs = zeros(1, T);
mab_thresholds = zeros(T, N);
mab_rewards = zeros(T, mab_steps / 100);
mab_reward = zeros(1, T);
mc_mab_reward = zeros(1, T);
mc_mab_voi = zeros(1, T);
mc_mab_energies = zeros(T, N);
mc_mab_channel_uses = zeros(1, T);
mc_mab_goodputs = zeros(1, T);

% Run LIBRA and BETA
for t = 1 : T
    t
    mus = ones(1, N) + (rand(1, N) - 0.5) * 2 * sigma;
    exp_reward(t, :) = mus;
    cdf = zeros(N, length(values) + 1);
    for n = 1 : N
        cdf(n, :) = [0, 1 - exp(-(values + delta / 2) / mus(n))];
    end

    % Pre-compute transmission values
    tx_values = zeros(size(cdf));
    for n = 1 : N
        pmf = diff(cdf(n, :));
        prob_val = pmf .* values;
        tx_values(n, 1 : length(values)) = flip(cumsum(flip(prob_val)));
        tx_values(n, :) = tx_values(n, :) ./ (1 - cdf(n, :));
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
    pull_channel(t) = pull_tx;
    pull_energies(t, nb(1)) = pull_tx;
    pull_goodputs(t) = pull_tx;

    % Determine LIBRA solution
    [~, ~, initial_thresholds] = equal_value_initialization(cdf, values, psi, success);
    [thresholds, reward_iter] = iterated_best_response(cdf, psi, epsilon, success, values, initial_thresholds, max_iterations);
    reward = max(reward_iter);
    th_rewards(t) = reward;
    th_energies(t, :) = 1 - thresholds;
    veq_thresholds(t, :) = initial_thresholds;
    th_thresholds(t, :) = thresholds;
    % % Monte Carlo check
    % [mc_voi, mc_reward, energy, goodput, channel_use, ~] = montecarlo(cdf, values, psi, success, thresholds, M);
    % mc_rewards(t) = mc_reward;
    % mc_energies(t, :) = energy;
    % mc_channel_uses(t) = channel_use;
    % mc_goodputs(t) = goodput;
    % Run BETA
    [mab_policy, mab_voi_mean, mab_reward_mean, mab_energy, mab_goodput, mab_channel_use, mab_reward_history, mab_policy_history] = mamab_semibandit(cdf, values, thetas, success, psi, exploration, kappa, memory, mab_steps);
    % Evaluate BETA solution convergence
    for s = 100 : 100 : mab_steps
        mab_threshold_values = mab_policy_history(:, s);
        mab_threshold_indices = zeros(1, N);
        mab_thresholds = zeros(1, N);
        for m = 1 : N
            mab_threshold_indices(m) = find(abs(values - mab_threshold_values(m)) < delta / 10, 1);
            mab_thresholds(m) = cdf(m, mab_threshold_indices(m));
        end
        % Compute value
        mab_next_reward = 0;
        tx_num = find(success > 0);
        for tx = tx_num
            % Tx transmitters (consider possible combinations)
            possible_tx = nchoosek(1 : N, tx);
            for comb = 1 : size(possible_tx, 1)
                prob_vec = mab_thresholds;
                for m = 1 : N
                    if (any(possible_tx(comb, :) == m))
                        prob_vec(m) = 1 - mab_thresholds(m);
                    end
                end
                prob_comb = prod(prob_vec);
                for m = possible_tx(comb, :)
                    threshold_index = find(cdf(m, :) >= mab_thresholds(m), 1);
                    if (~isempty(threshold_index))
                        mab_next_reward = mab_next_reward + prob_comb * success(tx) * tx_values(m, threshold_index);
                    end
                end
            end
        end
        for m = 1 : N
            mab_next_reward = mab_next_reward - psi * (1 - mab_thresholds(m));
        end
        mab_rewards(t, s / 100) = mab_next_reward;
    end
    for m = 1 : N
        mab_threshold_indices(m) = find(abs(values - mab_policy(m)) < delta / 10, 1);
        mab_thresholds(m) = cdf(m, mab_threshold_indices(m));
    end
    % % BETA Monte Carlo check
    % [mc_mab_voi(t), mc_mab_reward(t), mc_mab_energies(t, :), mc_mab_goodputs(t), mc_mab_channel_uses(t), ~] = montecarlo(cdf, values, psi, success, mab_thresholds, M);
    % mab_reward(t) = mab_rewards(t, end);
end

