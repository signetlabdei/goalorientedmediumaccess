function [v_eq, reward, thresholds] = equal_value_initialization(cdf, values, psi, success)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                  function: equal_value_initialization                   %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Determines the VoI quantile thresholds considering the same actual VoI  %
% as the threshold for all nodes                                          %
%                                                                         %
% Inputs:                                                                 %
% -cdf:             the CDF of values for each node [N x V]               %
% -values:          the possible values for all nodes [1 x V]             %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -success:         the success probability vector [1 x N]                %
%                                                                         %
% Outputs:                                                                %
% -v_eq:            the value of the transmission threshold [scalar]      %
% -reward:          the reward of the initialization policy [scalar]      %
% -thresholds:      the threshold quantiles [1 x N]                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Utility variables
N = size(cdf, 1);
V = length(values);
% Outputs
v_eq = -1;
reward = 0;
thresholds = zeros(1, N);
success = [success, 0];

tx_quantile = zeros(1, N);
exp_values = zeros(1, N);

% Pre-compute transmission values
tx_values = zeros(size(cdf));
for n = 1 : N
    pmf = diff(cdf(n, :));
    prob_val = pmf .* values;
    tx_values(n, 1 : V) = flip(cumsum(flip(prob_val)));
    tx_values(n, :) = tx_values(n, :) ./ (1 - cdf(n, :));
end


tx_num = find(success > 0);



% Iterate over possible values
for v = 1 : V - 1
    value = 0;
    tx_quantile = cdf(:, v)';

    for tx = tx_num
        % Tx transmitters (consider possible combinations)
        possible_tx = nchoosek(1 : N, tx);
        for comb = 1 : size(possible_tx, 1)
            prob_vec = tx_quantile;
            for m = 1 : N
                if (any(possible_tx(comb, :) == m))
                    prob_vec(m) = 1 - tx_quantile(m);
                end
            end
            prob_comb = prod(prob_vec);
            for n = possible_tx(comb, :)
                value = value + prob_comb * success(tx) * tx_values(n, v);
            end
        end
    end
    for n = 1 : N
        value = value - psi * (1 - tx_quantile(n));
    end

    % Update best value
    if (value > reward)
        reward = value;
        v_eq = values(v);
        thresholds = tx_quantile;
    end
end