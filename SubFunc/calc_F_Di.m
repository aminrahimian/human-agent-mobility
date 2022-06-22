function [F_Di]=calc_F_Di(xy_pede,uv_pede,r_target,v_desire,tau)
% forcing due to pedestrian's intension to r_target
[ei]=calc_ei(xy_pede,r_target);
F_Di = (v_desire.*ei - uv_pede)./tau;
end
