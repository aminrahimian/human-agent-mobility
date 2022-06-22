function [ei]=calc_ei(xy_pede,r_target)
% calculate the unit vector pointing to the target
diff = r_target - xy_pede;
ei = diff./vecnorm(diff,2,2); % pointing from center to the target
end