function [Rand] = RandTruncLevyAlpha(alpha, max) 
%RandTruncLevy: Generate random numbers from truncated Levy distribution
%skewness parameter beta = 0;
%   Rand: output number;
%   alpha: Levy index on (0,2];
%

V = (rand(1)-0.5)*pi; % generate a random variable V distributed homogenously on (-pi/2, pi/2);
W = exprnd(1); % generate an exponential variable W with mean 1.
Rand = sin(alpha*V)/(cos(V)^(1/alpha)) * ((cos((1-alpha)*V)/W)^(1/alpha-1));
if Rand > 0
    Rand = min(Rand, max);
elseif Rand < 0
    Rand = min(-Rand, max);
end
end