function [V_alpha_beta_gradient]=calc_V_alpha_beta(xy_alpha,xy_beta,sigma_alpha,nn, epsilon)
    % potential function describing the interaction between drones
    r_alpha_beta_norm = vecnorm((xy_alpha - xy_beta),2,2);
    V_alpha_beta_gradient = -epsilon*nn*...
                 ((sigma_alpha.^nn)./(r_alpha_beta_norm.^(nn+1))).*...
                 (2*(sigma_alpha.^nn)./(r_alpha_beta_norm.^nn)-1);
    V_alpha_beta_gradient(V_alpha_beta_gradient > 0) = 0;
end