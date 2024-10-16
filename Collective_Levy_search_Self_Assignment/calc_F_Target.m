function [F_target]=calc_F_Target(xy_drone,xy_target, Sigma, V_0)
    % force magnitude depends on the distance between targets and drones
    F_target = zeros(size(xy_target,1),2,size(xy_drone,1));
    for i = 1:size(xy_drone,1)
        % calculate the unit vector pointing to the target
        diff = xy_target - xy_drone(i,:);
        ei = diff./vecnorm(diff,2,2); % pointing from center to the target
        V_target_gradient = -V_0/Sigma*exp(-vecnorm(diff,2,2)./Sigma);
        F_target(:,:,i) = vecnorm(-V_target_gradient,2,2).*ei;
    end
end
