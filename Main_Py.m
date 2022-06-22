%% import image and create environmental condition layer
im = imread('Environ2k2k.png');
im = rgb2gray(im);
im = medfilt2(im,[15 15]);
im = imbinarize(im,'adaptive','Sensitivity',0.5,'ForegroundPolarity','bright');
imshow(im);

%% run the simulation
clear;
addpath('SubFunc');

% set the environmental condition
Dim1 = 2000;
Dim2 = 2000;
dx = Dim1/100;
[X,Y] = meshgrid(dx/2:dx:Dim1-dx/2,dx/2:dx:Dim2-dx/2);
v_scale_factor = round(ones(size(X)),2);

% agent parameters
tau = 0.5; % relaxation time
dt = tau; % time for velocity adjustment
dT = 60; % time for movement after velocity adjustment
R = 50; % detection radius parameter
theta_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75] * pi; % the space of moving directions
Epsilon = 0.9;

NN = 1;

for nn = 1:NN
    
    % delete all files from past loop
    delete input_*.csv

    % randomly generate the location of the real target
    Real_Target = [rand(1) * Dim1, rand(1) * Dim2];

    % initial location
    xy_pede = [0,0]; % initial location
    xy_pede_save = xy_pede; % save all location information

    % initial moving direction
    v_desire = 2.8 * interp2(X,Y,v_scale_factor,xy_pede(1),xy_pede(2),'nearest',1); % desired velocity [dependent on environment]
    vel_pede = v_desire; % initial velocity
    [theta_pede,idx] = datasample(theta_list,1); % randomly sample a initial direction
    u_pede = vel_pede.*cos(theta_pede); % u, v component of pedestrian
    v_pede = vel_pede.*sin(theta_pede);
    uv_pede = [u_pede, v_pede]; % initial velocity

    % probability of receiving a signal
    radius_factor = 2 * interp2(X,Y,v_scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
    Likely = exp(-1*norm(xy_pede - Real_Target)./(radius_factor * R));

    % output map to feed to python code, each row
    % [x_loc, y_loc, indx_right, indx_up, indx_left, indx_down, reward]
    output_mat = zeros(1,2);
    output_mat(1,1:2) = xy_pede;
    output_mat(1,3) = binornd(1,Likely);

     % initial time index
    t = 1;

    % generate the initial matrix to feed in python
    % for t = 2:20
    %
    %     % randomly sample the fake target
    %     l_target = 100 * rand(1); % distance to target
    %     [theta_target,idx] = datasample(theta_list,1);
    %     r_target_x = xy_pede(1) + l_target .* cos(theta_target);
    %     r_target_y = xy_pede(2) + l_target .* sin(theta_target);
    %     r_target = [r_target_x, r_target_y];
    %
    %     % update the movement of the agent
    %     [F_Di]=calc_F_Di(xy_pede,uv_pede,r_target,v_desire,tau);
    %     uv_pede_old = uv_pede; % save the previous step velocity
    %     uv_pede = uv_pede + F_Di*dt; % update velocity
    %     xy_pede = xy_pede + uv_pede*dT;
    %     xy_pede = round(xy_pede);
    %     v_desire = 2.8 * interp2(X,Y,v_scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
    %
    %     disp(v_desire);
    %
    %     xy_pede_save(t,:) = xy_pede;
    %
    %     % update the likelyhood equation
    %     radius_factor = 2 * interp2(X,Y,v_scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
    %     Likely = exp(-1*norm(xy_pede - Real_Target)./(radius_factor * R));
    %
    %     % output map to feed to python code, each row
    %     % [x_loc, y_loc, indx_right, indx_up, indx_left, indx_down, reward]
    %     output_mat(t,1:2) = xy_pede;
    %     output_mat(t,2+idx) = 1;
    %     output_mat(t,7) = binornd(1,Likely);
    %
    %     % judge if pass the target during each time step
    %     v1 = xy_pede_save(t,:); % the first vertex, end of time step
    %     v2 = xy_pede_save(t-1,:); % the second vertex, start of time step
    %     vec1 = v2 - v1;
    %     vec2 = Real_Target - v1;
    %     if dot(vec1,vec2)>=0 && dot(vec1,vec2)<=dot(vec1,vec1)
    %         d = abs( det([Real_Target-v1;v2-v1]) )/norm(v2-v1); % distanct from the real target to v1-v2
    %         if d <= R
    %             break
    %         end
    %     end
    %
    %     % output the .csv file
    %     writematrix(output_mat,'py_input.csv');
    %     writematrix(xy_pede,'location.csv');
    % end
    writematrix(output_mat,['input_',num2str(t),'.csv']);

    theta_target_save = [];

    % update the result using the output from python
    while norm(xy_pede - Real_Target) >= R
        disp([num2str(t*dT/3600),' hours']);

        % use command prompt
        %     call_py_cmd = ['cd ',pwd,'& python test_actions.py'];
        %     [status,cmdout] = system(call_py_cmd);
        %     if status ~= 0
        %         disp(cmdout);
        %     end

        % use pyrunfile
        idx = pyrunfile('Q_logit_regression.py', 'theta',...
            nn = t, L = Dim1, H = Dim2, epsilon = Epsilon);
        theta_target = theta_list(idx);

        % setup the location of the fake target according to the command
        l_target = 100;
        r_target_x = xy_pede(1) + l_target .* cos(theta_target);
        r_target_y = xy_pede(2) + l_target .* sin(theta_target);
        r_target = [r_target_x, r_target_y];

        % update the movement of the agent
        [F_Di]=calc_F_Di(xy_pede,uv_pede,r_target,v_desire,tau);
        uv_pede_old = uv_pede; % save the previous step velocity
        uv_pede = uv_pede + F_Di*dt; % update velocity
        xy_pede = xy_pede + uv_pede*dT;
        v_desire = 2.8 * interp2(X,Y,v_scale_factor,xy_pede(1),xy_pede(2),'nearest',1);

        % updata time step
        t = t+1;

        xy_pede_save(t,:) = xy_pede;

        % update the likelyhood equation
        radius_factor = 2 * interp2(X,Y,v_scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
        Likely = exp(-1*norm(xy_pede - Real_Target)./(radius_factor * R));

        % output map to feed to python code, each row
        % [x_loc, y_loc, indx_right, indx_up, indx_left, indx_down, reward]
        output_mat(t,1:2) = xy_pede;
        output_mat(t,3) = binornd(1,Likely);

        % judge if pass the target during each time step
        v1 = xy_pede_save(t,:); % the first vertex
        v2 = xy_pede_save(t-1,:); % the second vertex
        vec1 = v2 - v1;
        vec2 = Real_Target - v1;
        if dot(vec1,vec2)>=0 && dot(vec1,vec2)<=dot(vec1,vec1)
            d = abs( det([Real_Target-v1;v2-v1]) )/norm(v2-v1); % distanct from the real target to v1-v2
            if d <=50
                break
            end
        end

        delete input_*.csv
        writematrix(output_mat,['input_',num2str(t),'.csv']);

        if t*dT/3600> 24
            break
        end
    end

end

%%
% imagesc(dx/2:dx:L-dx/2,dx/2:dx:H-dx/2,v_scale_factor);
% colorbar
hold on
plot(xy_pede_save(:,1),xy_pede_save(:,2),'m-');
plot(Real_Target(1),Real_Target(2),'r.','markersize',15);
%%
test = pyrunfile('testing.py','b',L=2000,H=2000);


