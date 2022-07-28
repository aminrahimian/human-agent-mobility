function [F_alpha_B]=calc_F_alpha_B(xy_pede,r_target,phi,c_wall,Boundary_all)
        
    [ei]=calc_ei(xy_pede,r_target);
    nearest_boundary = zeros(size(xy_pede));
    
    % find nearest boundary points for each point
    for k = 1:size(xy_pede,1) % loop through pedestrian
        cur_loc = xy_pede(k,:);
        % calculate the distance to each point on the boundary for one pedestrian
        dist_B = vecnorm((cur_loc - Boundary_all),2,2);
        [~,I] = min(dist_B);
        nearest_boundary(k,:) = Boundary_all(I,:);
    end
    
    r_alpha_B = xy_pede - nearest_boundary;
    e_alpha_B = r_alpha_B./vecnorm(r_alpha_B,2,2); % Unit vector from boundary to the pedestrian
    
    % calculate the force
    [U_alpha_beta_gradient]=calc_U_alpha_B(xy_pede,nearest_boundary);
    F_alpha_B = vecnorm(-U_alpha_beta_gradient,2,2).*e_alpha_B;
    
    % add sight effect
    r_alpha_B_dot_e_beta = sum(-e_alpha_B.*ei,2,'omitnan');
    r_alpha_beta_cos_phi = vecnorm(e_alpha_B,2,2)*cos(phi);
    logic_great = r_alpha_B_dot_e_beta >= r_alpha_beta_cos_phi;
    logic_small = r_alpha_B_dot_e_beta < r_alpha_beta_cos_phi;
    F_alpha_B = F_alpha_B.*logic_great + F_alpha_B.*logic_small.*c_wall;
end