function [new_exp_values] = semibandit_update(exp_values, success, thetas, gamma, psi, vois, actions, outcome, alphas, betas, lambdas, rhos)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                       function: semibandit_update                       %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Computes a MAMAB update by using counterfactual reasoning               %
%                                                                         %
% Inputs:                                                                 %
% -exp_values:      the current arm rewards [N x V]                       %
% -success:         the success probability vector [1 x N]                %
% -thetas:          the possible BETA thresholds [1 x V]                  %
% -gamma:           the learning rate for this step [scalar, 0-1]         %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -vois:            the VoI for this step for all nodes [N x 1]           %
% -actions:         whether each node transmitted [1 x N, bool]           %
% -outcome:         the IDs of the successful nodes, 0 for silence, or -1 %
%                   for a collision [1 x S, int]                          %
% -alphas:          ratio of occupied slots if node n is silent [1 x N]   %
% -betas:           ratio of successful slots if node n is silent [1 x N] %
% -lambdas:         average VoI if node n is silent [1 x N]               %
% -rhos:            estimated transmission rates [1 x N]                  %
%                                                                         %
% Outputs:                                                                %
% -new_exp_values:  the updated arm rewards [N x V]                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Utility variables
N = size(exp_values, 1);
V = size(exp_values, 2);
new_exp_values = zeros(N, V);

coll_success = zeros(1, N);
coll_success(1) = 0;

% Outcome: silence
if (outcome == 0)
    for n = 1 : N
        tx_idx = find(thetas > vois(n), 1);
        if (~isempty(tx_idx))
            % The node transmits: success
            new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) + gamma * (vois(n) - psi);
            % The node is silent: silence
            new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end);
        else
            new_exp_values(n, :) = exp_values(n, :) + gamma * (vois(n) - psi);
        end
    end
else
    % Pure collision channel
    if (success == coll_success)
        % Outcome: success
        if (outcome > 0)
            % Successful node
            tx_idx = find(thetas > vois(outcome), 1);
            if (~isempty(tx_idx))
                % The node transmits: success
                new_exp_values(outcome, 1 : tx_idx - 1) = exp_values(outcome, 1 : tx_idx - 1) + gamma * (vois(outcome) - psi);
                % The node is silent: silence
                new_exp_values(outcome, tx_idx : end) = exp_values(outcome, tx_idx : end);
            else
                new_exp_values(outcome, :) = exp_values(outcome, :) + gamma * (vois(outcome) - psi);
            end
            % Other nodes
            for n = setdiff(1 : N, outcome)
                tx_idx = find(thetas > vois(n), 1);
                if (~isempty(tx_idx))
                    % The node transmits: collision
                    new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) - gamma * 2 * psi;
                    % The node is silent: success (same VoI as the real outcome)
                    new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) + gamma * (vois(outcome) - psi);
                else
                    new_exp_values(n, :) = exp_values(n, :) - gamma * 2 * psi;
                end
            end
        end
    
        % Outcome: collision
        if (outcome == -1)
            for n = setdiff(1 : N, outcome)
                if (actions(n) == 0)
                    % The node was not a part of the collision set: collision is
                    % unavoidable
                    tx_idx = find(thetas > vois(n), 1);
                    activity = sum(rhos) - rhos(n) - betas(n);
                    collisions = alphas(n) - betas(n);
                    if (collisions > 0)
                        if (~isempty(tx_idx))
                            % The node transmits: collision (involving the node!)
                            new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) - gamma * psi * (activity + 1) / collisions;
                            % The node is silent: collision
                            new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) - gamma * psi * activity / collisions;
                        else
                            new_exp_values(n, :) = exp_values(n, :) - gamma * psi * (activity + 1) / collisions;
                        end
                    else
                        new_exp_values(n, :) = exp_values(n, :);
                    end
                else
                    % The node was a part of the collision set: collision might be
                    % avoidable if it is silent
                    tx_idx = find(thetas > vois(n), 1);
                    activity = sum(rhos) - rhos(n);
                    if (~isempty(tx_idx))
                        % The node transmits: collision (involving the node!)
                        new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) - gamma * psi * (activity + 1) / alphas(n);
                        % The node is silent: collision or success
                        new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) +  gamma * (lambdas(n) - psi * activity / alphas(n));
                    else
                        new_exp_values(n, :) = exp_values(n, :) - gamma * psi * (activity + 1) / alphas(n);
                    end
                end
            end
        end
    else
        % Capture channel
        for n = 1 : N
            p_tx = zeros(1, N);
            ps_tx = zeros(1, N);
            received_reward = 0;
            % At least one successful node
            if (outcome(1) > 0)
                tx_max = find(success, 1, 'last');
                n_succ = length(outcome);
                % Check if node n was successful
                ns = ~isempty(find(outcome == n, 1));
                os = n_succ - ns;
                received_reward = sum(vois(outcome));
                others = 1 : N;
                others(outcome) = [];
                p_tx(1) = prod(1 - rhos(others));
                ps_tx(1) = p_tx(1) * success(n_succ) ^ n_succ;
                for failed = 1 : tx_max - n_succ
                    % Tx other transmitters (consider possible combinations)
                    possible_tx = nchoosek(others, failed);
                    for comb = 1 : size(possible_tx, 1)
                        prob_vec = 1 - rhos(others);
                        for m = others
                            if (any(possible_tx(comb, :) == m))
                                prob_vec(find(others == m, 1)) = rhos(m);
                            end
                        end
                        p_tx(failed + 1) = p_tx(failed + 1) + prod(prob_vec);
                        tx_tot = failed + n_succ - ns + actions(n);
                        ps_tx(failed + 1) = ps_tx(failed + 1) + prod(prob_vec) * success(tx_tot) ^ (n_succ) * (1 - success(tx_tot)) ^ (tx_tot - n_succ);
                    end
                end
            else
                others = 1 : N;
                others(n) = [];
                os = 0;
                for failed = 1 : N - 1
                    % Tx other transmitters (consider possible combinations)
                    possible_tx = nchoosek(others, failed);
                    for comb = 1 : size(possible_tx, 1)
                        prob_vec = 1 - rhos(others);
                        for m = others
                            if (any(possible_tx(comb, :) == m))
                                prob_vec(find(others == m, 1)) = rhos(m);
                            end
                        end
                        p_tx(failed + 1) = p_tx(failed + 1) + prod(prob_vec);
                        tx_tot = failed + actions(n);
                        ps_tx(failed + 1) = ps_tx(failed + 1) + prod(prob_vec) * (1 - success(tx_tot)) ^ tx_tot;
                    end
                end
            end
            % Compute the posterior distribution of the number of
            % transmitters other than n
            ps_tx = ps_tx ./ max(1e-9, p_tx);
            ps = ps_tx * p_tx';
            ptx_s = zeros(1, N);
            for failed = 1 : N
                ptx_s(failed) = ps_tx(failed) * p_tx(failed) / max(ps, 1e-9);
            end

            reward_tx = 0;
            reward_silence = 0;
            if (actions(n) == 1)
                % Node n was involved
                reward_tx = received_reward - psi * (os + 1);
                for otx = 0 : N - os - 1
                    reward_tx = reward_tx - otx * psi * ptx_s(otx + 1);
                    if (otx + os > 0)
                        reward_silence = reward_silence + ptx_s(otx + 1) * (otx + os)  * (lambdas(n) / max(1e-9, betas(n)) * alphas(n) * success(otx + os) - psi);
                    end
                end
            else
                % Node n was not involved
                reward_silence = received_reward;
                for otx = 0 : N - os - 1
                    reward_tx = reward_tx + ptx_s(otx + 1) * ((vois(n) + (otx + os)  * (lambdas(n) / max(1e-9, betas(n)) * alphas(n))) * success(otx + os + 1) - (otx + os + 1) * psi);
                    if (otx + os > 0)
                        reward_silence = reward_silence - (otx + os) * psi * ptx_s(otx + 1);
                    end
                end
            end
            tx_idx = find(thetas > vois(n), 1);
            if (~isempty(tx_idx))
                % The node transmits
                new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) + gamma * reward_tx;
                % The node is silent
                new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) + gamma * reward_silence;
            else
                new_exp_values(n, :) = exp_values(n, :) + gamma * reward_tx;
            end
        end
    end
end



end