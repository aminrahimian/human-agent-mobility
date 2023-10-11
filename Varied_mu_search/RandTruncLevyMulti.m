function [Rand] = RandTruncLevyMulti(alpha, max, N) 
%RandTruncLevy: Generate a series of random numbers from truncated Levy distribution
%skewness parameter beta = 0;
%   Rand: output number;
%   alpha: Levy index on (0,2];
%   N: the number of sampling values
%   max: the cutoff or the upper limit of the distribution 
V = (rand(N,1)-0.5)*pi; % generate a random variable V distributed homogenously on (-pi/2, pi/2);
W = exprnd(N,1); % generate an exponential variable W with mean 1.
Rand = sin(alpha.*V)./(cos(V).^(1/alpha)) .* ((cos((1-alpha).*V)./W).^(1/alpha-1));
Rand(Rand>0) = min(Rand(Rand>0), max);
Rand(Rand<0) = min(-Rand(Rand<0), max);
end