%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                           script: attractors                            %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Determines the attraction regions of LIBRA policies by iterating over   %
% all possible initial starting policies (N=3 nodes, exponential or       %
% Gaussian VoI distribution) over a pure collision channel                %
% Parameters:                                                             %
% -M:           the number of steps to simulate [scalar, int]             %
% -epsilon:     the grid approximation error [scalar, 0-1]                %
% -mu:          the VoI mean (Gaussian) or rate (exp.) [scalar, R+]       %
% -sigma:       the VoI std. deviation (Gaussian) [scalar, R+]            %
% -Vmax:        the maximum possible VoI [scalar, R+]                     %
% -psi:         the transmission attempt cost [scalar, R+]                %
% -max_iter:    the maximum number of IBR iterations [scalar, int]        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




close all
clearvars

% Simulation parameters
M = 1e6;
epsilon = 1e-3;
mu = 0.5;
sigma = 2;
Vmax = 20;
psi = 0.25;
max_iter = 1000;

% Utility variables
values = 0 : epsilon : Vmax;
cdf = zeros(3, length(values) + 1);
threshold_values = zeros(3, 1 + round(1 / epsilon), 1 + round(1 / epsilon));
rewards = zeros(1 + round(1 / epsilon));

% Exponential distribution
cdf(1, :) = [0, 1 - exp(-mu * (values + epsilon / 2))];
cdf(2, :) = [0, 1 - exp(-mu * (values + epsilon / 2))];
cdf(3, :) = [0, 1 - exp(-mu * (values + epsilon / 2))];

% Gaussian distribution
% cdf(1, :) = [0, normcdf((values + epsilon / 2) / sigma - mu)];
% cdf(2, :) = [0, normcdf((values + epsilon / 2) / sigma - mu)];
% cdf(3, :) = [0, normcdf((values + epsilon / 2) / sigma - mu)];

% Iterate over all values
for theta_1 = 0 : epsilon : 1
    theta_1
    for theta_2 =  0 : epsilon : 1
        % Compute LIBRA solution
        [thresholds, reward_history] = iterated_best_response(cdf, psi, epsilon, values, [0, theta_1, theta_2], max_iter);
        % Store reward and threshold values
        threshold_values(:, 1 + round(theta_1 / epsilon), 1 + round(theta_2 / epsilon)) = thresholds;
        rewards(1 + round(theta_1 / epsilon), 1 + round(theta_2 / epsilon)) = reward_history(end);
    end
end
