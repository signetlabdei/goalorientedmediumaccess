function [success] = success_par(N, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                          function: success_par                          %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Run the epsilon-hedge MAMAB algorithm                                   %
%                                                                         %
% Inputs:                                                                 %
% -thetas:          the possible BETA thresholds [1 x V]                  %
% -exp_values:      the current arm rewards [N x V]                       %
% -epsilon:         exploration rate of epsilon-hedge [scalar, 0-1]       %
%                                                                         %
% Outputs:                                                                %
% -actions:         the action for each node [1 x N]                      %
% -greedy:          the highest-reward option for each node [1 x N]       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

success = zeros(1, N + 1);
success(1) = 1;

if (K > 1)
    p_succ = zeros(N);
    
    for t = 1 : N
        if (t <= K)
            p_succ(t, t) = factorial(K) / factorial(K - t) / K ^ t;
        end
        for s = 1 : min(t - 1, K)
            for c = 1 : min((t - s) / 2, K - s)
                p_sel = nchoosek(t, s) * nchoosek(t - s, c) * nchoosek(t - s - c, c);
                p_chan = factorial(K) / (factorial(c) * factorial(K - s - c) * K ^ t * 2 ^ c 
                p_succ(t, s) = p_succ(t, s) + ps * pc;
            end
        end
    end
end




end