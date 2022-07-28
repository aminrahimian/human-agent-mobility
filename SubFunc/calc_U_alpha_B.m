function [U_alpha_beta_gradient]=calc_U_alpha_B(xy_pede,nearest_boundary)

    U_0 = 100; % m^2/s^2
    R = 3; %0.2;

    r_alpha_B_norm = vecnorm((xy_pede - nearest_boundary),2,2);
    
    U_alpha_beta_gradient = -U_0/R*exp(-r_alpha_B_norm./R);

end