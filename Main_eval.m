% This file evaluates different python algorithoms under the same
% rule.

% To do: enable parfor

clear;
addpath('SubFunc');

% Select the case that you want to use as 1, others as 0;
Shortest = 0;
ThroughSearch = 0;
RandomWalk = 0;
LevyFlight = 0;

% set the environmental condition
Dim1 = 2000;
Dim2 = 2000;
dx = Dim1/200;
x1 = 0:dx:Dim1;
x2 = 0:dx:Dim2;
[X,Y] = meshgrid(x1,x2);
board = [0,0;Dim1,0;Dim1,Dim2;0,Dim2;0,0];

% agent parameters
V_best = 2.8; % the desired moving velocity under the best environmental condition
tau = 0.5; % relaxation time
dt = tau; % time for velocity adjustment
dT = 60; % time for movement after velocity adjustment
R = 100; % detection radius parameter
radius_factor_best = 2; % The radius factor of receiving signal under best env condition

N_arm = 8; % number of arms used
theta_list = linspace(0,2*(1-1/N_arm),N_arm) * pi; % the space of moving directions


% determine whether add environmental condition
ENV = 0; % if 1, heterogeneous environmental condition is used
if ENV == 1
    load('roi.mat');
    inroi = inROI(roi1,X(:),Y(:)) | inROI(roi2,X(:),Y(:));
    inroi = reshape(inroi,length(x1),length(x2));
else
    inroi = false(length(x1),length(x2));
end
scale_factor = ones(length(x1),length(x2));
scale_factor(inroi) = 0.3; % change this for scaling of the agent's movement

% parameter for python code, if any
Epsilon = 0.9;

% set total number of loops
NN = 1;

% The Main Loop
for nn = 1:NN

    % delete all files from past loop
    delete input_*.csv

    % randomly generate the location of the real target
    Real_Target = [rand(1) * Dim1, rand(1) * Dim2];

    % initial location
    xy_pede = [0,0]; % initial location
    xy_pede_save = xy_pede; % save all location information

    % initial moving direction
    v_desire = V_best * interp2(X,Y,scale_factor,xy_pede(1),xy_pede(2),'nearest',1); % desired velocity [dependent on environment]
    vel_pede = v_desire; % initial velocity
    [theta_pede,idx] = datasample(theta_list,1); % randomly sample a initial direction
    u_pede = vel_pede.*cos(theta_pede); % u, v component of pedestrian
    v_pede = vel_pede.*sin(theta_pede);
    uv_pede = [u_pede, v_pede]; % initial velocity

    % probability of receiving a signal
    radius_factor = radius_factor_best * interp2(X,Y,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
    Likely = exp(-1*norm(xy_pede - Real_Target)./(radius_factor * R));

    % output map to feed to python code
    output_mat = zeros(1,2);
    output_mat(1,1:2) = xy_pede;
    output_mat(1,3) = binornd(1,Likely);

    % initial time index
    t = 1;

    % write output matrix to .csv file
    writematrix(output_mat,['input_',num2str(t),'.csv']);

    % initial some variables to record value
    T_total = 0;
    TimeLength_record = zeros(NN,1); % record the leng of time needed

    % update the result using the output from python
    while norm(xy_pede - Real_Target) >= R

        % if don't want to display
        disp([num2str(T_total/3600),' hours']);

        % use pyrunfile
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Edit here according to different python core %%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        idx = pyrunfile('Random_walk.py', 'theta',...
            nn = t, L = Dim1, H = Dim2);
        theta_target = theta_list(idx);

        % setup the location of the fake target according to the command
        l_target = 100;
        r_target_x = xy_pede(1) + l_target .* cos(theta_target);
        r_target_y = xy_pede(2) + l_target .* sin(theta_target);
        r_target = [r_target_x, r_target_y];

        % update the movement and location of the agent
        [F_Di]=calc_F_Di(xy_pede,uv_pede,r_target,v_desire,tau);
        uv_pede_old = uv_pede; % save the previous step velocity
        uv_pede = uv_pede + F_Di*dt; % update velocity
        xy_pede = xy_pede + uv_pede*dT;

        % apply reflective boundary condition
        if ~inpolygon(xy_pede(1),xy_pede(2),board(:,1),board(:,2))
            xy_pede(xy_pede < 0) = abs(xy_pede(xy_pede < 0));
            if xy_pede(1) > Dim1
                xy_pede(1) = 2*Dim1 - xy_pede(1);
            end
            if xy_pede(2) > Dim2
                xy_pede(2) = 2*Dim2 - xy_pede(2);
            end
        end

        % update the moving velocity according to the env condition
        v_desire = V_best * interp2(X,Y,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);

        % updata time step
        t = t+1;
        
        % record the full moving history for this realization
        xy_pede_save(t,:) = xy_pede;

        % update the likelyhood equation
        radius_factor = radius_factor_best * interp2(X,Y,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
        Likely = exp(-1*norm(xy_pede - Real_Target)./(radius_factor * R));

        % output map to feed to python code, each row
        output_mat(t,1:2) = xy_pede;
        output_mat(t,3) = binornd(1,Likely);

        % judge if pass the target during each time step
        v1 = xy_pede_save(t,:); % the first vertex
        v2 = xy_pede_save(t-1,:); % the second vertex
        vec1 = v2 - v1;
        vec2 = Real_Target - v1;
        if dot(vec1,vec2)>=0 && dot(vec1,vec2)<=dot(vec1,vec1)
            d = abs( det([Real_Target-v1;v2-v1]) )/norm(v2-v1); % distanct from the real target to v1-v2
            if d <= R
                break
            end
        end
        
        % delete old .csv file
        % write updated output matrix to .csv file
        delete input_*.csv
        writematrix(output_mat,['input_',num2str(t),'.csv']);
        
        % set the upper breaking time limit
        T_total = T_total + dT;
        if T_total/3600> 72
            break
        end
    end

    % Record the total time length needed for this realization
    TimeLength_record(nn) = T_total/3600;
    disp(['Realization ',num2str(nn),' using ',num2str(T_total/3600),' hours']);
end
%% Plotting a single track
figure(2);
hold on
plot(Real_Target(1),Real_Target(2),'rp','markersize',10,'MarkerFaceColor','r');
plot(0,0,'gs','markersize',10,'MarkerFaceColor','g');
plot(xy_pede_save(:,1),xy_pede_save(:,2),'m-','LineWidth',1.5);
plot(xy_pede_save(end,1),xy_pede_save(end,2),'bo','markersize',10,'MarkerFaceColor','b');
if ENV
    load('roi.mat');
    x1 = roi1.Position;
    x2 = roi2.Position;
    fill(x1(:,1),x1(:,2),'b', 'FaceAlpha', 0.3,'edgecolor','none');
    fill(x2(:,1),x2(:,2),'b', 'FaceAlpha', 0.3,'edgecolor','none');
end
axis([0 Dim1 0 Dim2]);
% set(gca,'Ydir','reverse');
xlb = xlabel('m');
ylb = ylabel('m');
ttl = title('Random Walk');
if ENV
    lgd = legend('real target','agent initial location','agent path','agent current position','environmental condition',...
        'location','northeast');
else
    lgd = legend('real target','agent initial location','agent path','agent current position',...
        'location','northwest');
end
% legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,500,500]);
set(gca,'looseInset',[0 0 0 0]);

