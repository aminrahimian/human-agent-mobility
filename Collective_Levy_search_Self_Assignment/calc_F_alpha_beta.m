function [F_alpha_beta]=calc_F_alpha_beta(xy_drone_1,sigma_alpha, nn, epsilon)
    % calculate the force between drones
    XY_drone=xy_drone_1;
    F_alpha_beta = zeros(size(XY_drone)); % Stores the force for drones
    
    % each loop centers at one pedestrian
    for n = 1:size(XY_drone,1) % loop through particles
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%% separate alpha and beta %%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        xy_alpha = XY_drone(n,:);
        xy_beta = XY_drone; xy_beta(n,:) = []; % remove the center point        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%% calculate the force %%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % calculate the norm of the gradient
        [V_alpha_beta_gradient]=calc_V_alpha_beta(xy_alpha,xy_beta,sigma_alpha,nn, epsilon);
         
        % gradient in direction of e_alpha_beta
        r_alpha_beta = xy_alpha - xy_beta; % Vector from beta to alpha
        e_alpha_beta = r_alpha_beta./vecnorm(r_alpha_beta,2,2); % unit vector from beta to alpha
        F_a_b_1 = vecnorm(-V_alpha_beta_gradient,2,2).*e_alpha_beta;
        
        F_alpha_beta(n,:) = sum(F_a_b_1,1);      
    end         
end













