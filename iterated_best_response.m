function [thresholds, reward_history] = iterated_best_response(cdf, psi, epsilon, success, values, init_thresholds, max_iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    function: iterated_best_response                     %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Computes the LIBRA thresholds for a given initial policy by running the %
% Iterated Best Response (IBR) part of the algorithm                      %
%                                                                         %
% Inputs:                                                                 %
% -cdf:             the CDF of values for each node [N x V]               %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -epsilon:         error threshold to stop IBR [scalar, 0-1]             %
% -success:         the success probability vector [1 x N]                %
% -values:          the possible values for all nodes [1 x V]             %
% -init_thresholds: the initial thresholds to start IBR [1 x N]           %
% -max_iter:        the maximum number of IBR iterations [scalar, int]    %
%                                                                         %
% Outputs:                                                                %
% -thresholds:      the final thresholds at convergence [1 x N]           %
% -reward_history:  the reward for each IBR iteration [1 x max_iter]      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Utility variables
N = length(init_thresholds);
V = size(values, 2);
nodes = 1 : N;
reward_history = zeros(1, N * max_iter + 1);
tx_num = find(success > 0);
success = [success, 0];

% Pre-compute transmission values
tx_values = zeros(size(cdf));
for n = 1 : N
    pmf = diff(cdf(n, :));
    prob_val = pmf .* values;
    tx_values(n, 1 : V) = flip(cumsum(flip(prob_val)));
    tx_values(n, :) = tx_values(n, :) ./ (1 - cdf(n, :));
end


previous_thresholds = -ones(1, N);
thresholds = init_thresholds;
iter = 1;
% Compute initial value
threshold_index = find(cdf(n, :) >= thresholds(n), 1);
if (~isempty(threshold_index))
    for tx = tx_num
        % Tx transmitters (consider possible combinations)
        possible_tx = nchoosek(1 : N, tx);
        for comb = 1 : size(possible_tx, 1)
            prob_vec = thresholds;
            for m = 1 : N
                if (any(possible_tx(comb, :) == m))
                    prob_vec(m) = 1 - thresholds(m);
                end
            end
            prob_comb = prod(prob_vec);
            for n = possible_tx(comb, :)
                reward_history(1) = reward_history(1) + prob_comb * success(tx) * tx_values(n, threshold_index);
            end
        end
    end
    for n = 1 : N
        reward_history(1) = reward_history(1) - psi * (1 - thresholds(n));
    end
end


% Iterated best response
while(iter < max_iter && max(abs(previous_thresholds - thresholds)) >= epsilon)
    previous_thresholds = thresholds;
    % Iterate over the nodes
    for n = nodes
        % Find best response
        interferers = 1 : N;
        interferers(n) = [];
        theta = 0;
        zeta = prod(thresholds(interferers)) * success(1);        
        % Compute denominator zeta
        for tx = tx_num
            % Tx interferers (consider possible combinations)
            possible_tx = nchoosek(interferers, tx);
            for comb = 1 : size(possible_tx, 1)
                prob_vec = thresholds(interferers);
                for m = interferers
                    if (any(possible_tx(comb, :) == m))
                        prob_vec(find(interferers == m, 1)) = 1 - thresholds(m);
                    end
                end
                zeta = zeta + prod(prob_vec) * success(tx + 1);
            end
        end

        % Compute numerator theta
        for tx = tx_num
            if (tx == N)
                success_int = success(tx);
            else
                success_int = (success(tx) - success(tx + 1));
            end
            % Some interferers (consider possible combinations)
            possible_tx = nchoosek(interferers, tx);
            for comb = 1 : size(possible_tx, 1)
                prob_vec = thresholds(interferers);
                for m = interferers
                    if (any(possible_tx(comb, :) == m))
                        prob_vec(find(interferers == m, 1)) = 1 - thresholds(m);
                    end
                end
                interf_value = 0;
                for m = possible_tx(comb, :)
                    threshold_index = find(cdf(m, :) >= thresholds(m), 1);
                    if (~isempty(threshold_index))
                        interf_value = interf_value + tx_values(m, threshold_index);
                    end
                end
                theta = theta + prod(prob_vec) * success_int * interf_value;
            end
        end

        theta = theta + psi;

        theta = theta / zeta;

        value_index = find(values >= theta, 1);
        if (isempty(value_index))
            thresholds(n) = 1;
        else
            thresholds(n) = cdf(n, value_index);
        end
        % Compute value
        for tx = tx_num
            % Tx transmitters (consider possible combinations)
            possible_tx = nchoosek(1 : N, tx);
            for comb = 1 : size(possible_tx, 1)
                prob_vec = thresholds;
                for m = 1 : N
                    if (any(possible_tx(comb, :) == m))
                        prob_vec(m) = 1 - thresholds(m);
                    end
                end
                prob_comb = prod(prob_vec);
                for m = possible_tx(comb, :)
                    threshold_index = find(cdf(m, :) >= thresholds(m), 1);
                    if (~isempty(threshold_index))
                        reward_history(N * (iter - 1) + n + 1) = reward_history(N * (iter - 1) + n + 1) + prob_comb * success(tx) * tx_values(m, threshold_index);
                    end
                end
            end
        end
        for m = 1 : N
            reward_history(N * (iter - 1) + n + 1) = reward_history(N * (iter - 1) + n + 1) - psi * (1 - thresholds(m));
        end
    end
    iter = iter + 1;
end

reward_history(iter : end) = max(reward_history);