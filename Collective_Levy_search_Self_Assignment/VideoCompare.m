clear; clc; close all

% make video for comparison
social_alpha_series = [50;100;150;200;250;300;350;400];
% Sigma_series = [25;50;75;100;150;200;250;300];
Sigma_series = [25;50;100;200;300;500;700;900]; % for 5000 domain

Dim = 10000; N = 200; NDrone = 4;
mu1 = 1; maxT = 1000; maxL = 1000; % cut-off length
epsilon = 30; nn = 0.5; V_0 = 100;

R = 25; % detection radius
board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0]; % boundary
% load(strcat('TarDim=',num2str(Dim),'N=', num2str(N),'.mat'))
load("TarDistLarge.mat")
xy_target_ini = [tar_x_rec, tar_y_rec];


social_alpha = 500; % L-J potential
Sigma = 300; % exponential potential
tic
for isearch = 1:1 %N_search
    % initial location
    Ini_center = [Dim/2, Dim/2];
    x_loc = [Ini_center(1)-R;Ini_center(1)+R;Ini_center(1)-R;Ini_center(1)+R]; % initial x of 4 drones
    y_loc = [Ini_center(2)+R;Ini_center(2)+R;Ini_center(2)-R;Ini_center(2)-R]; % initial y of 4 drones
    xy_drones = [x_loc, y_loc];
    xy_drones_T = zeros(maxT,2,NDrone);
    for iDrone = 1:NDrone
        xy_drones_T(1,:,iDrone) = xy_drones(iDrone,:);
    end
    
    F_total = zeros(NDrone,2);
    px = zeros(NDrone,1); py = zeros(NDrone,1);
    % steo length series for 4 drones
    step_mu1 = transpose([RandTruncLevyMulti(mu1, maxL-25,maxT)+25,RandTruncLevyMulti(mu1, maxL-25,maxT)+25, ...
        RandTruncLevyMulti(mu1, maxL-25,maxT)+25,RandTruncLevyMulti(mu1, maxL-25,maxT)+25]);
    
    Time_index = [];
    LastTar = 0;
    limitT = 600; %TASK TIME
    NumTarDrone = zeros(NDrone,1);
    TotalDist = zeros(NDrone,limitT);
    xy_target = xy_target_ini;
    for t = 1:limitT %maxT-1
        % calculate forces between drones
        F_drones = calc_F_alpha_beta(xy_drones, social_alpha, nn, epsilon); % LJ Potential
        % calculate forces between drones and targets
        F_target = calc_F_Target(xy_drones,xy_target, Sigma, V_0); % Expoenential Potential
        for idrone = 1:NDrone
            F_total(idrone,:) = sum(F_target(:,:,idrone)) + F_drones(idrone,:);
            px(idrone) = F_total(idrone,1)/vecnorm(F_total(idrone,:),2,2); % projection on x
            py(idrone) = F_total(idrone,2)/vecnorm(F_total(idrone,:),2,2); % projection on y
            xy_drones(idrone,:) = xy_drones(idrone,:) + step_mu1(idrone,t)*[px(idrone),py(idrone)];
            TotalDist(idrone, t) = step_mu1(idrone,t);
            xy = [xy_drones(idrone,1), xy_drones(idrone,2)];
            if ~inpolygon(xy(1),xy(2),board(:,1),board(:,2))
                xy(xy < 0) = abs(xy(xy < 0));
                if xy(1) > Dim
                    xy(1) = 2*Dim - xy(1);
                end
                if xy(2) > Dim
                    xy(2) = 2*Dim - xy(2);
                end
            end
            xy_drones(idrone,:) = xy; % update the location of drone using reflective BC
            xy_drones_T(t+1,:,idrone) = xy;
            dist = pdist2(xy_drones(idrone,:), xy_target) - R;
    
            if min(dist) <= 0
                NumTarDrone(idrone) = NumTarDrone(idrone) + numel(xy_target(find(dist<0),1));
                xy_target(find(dist<0),:) = []; %  remove the detected targets
                LastTar = t;
            end
        end
    end
end
toc

% Define video parameters
outputVideoFile = 'SearchLarge.mp4';  % Specify the output video file
frameRate = 30;  % Frames per second

% Create a VideoWriter object
videoWriterObj = VideoWriter(outputVideoFile, 'MPEG-4');
videoWriterObj.FrameRate = frameRate;

% Open the video file for writing
open(videoWriterObj);

% Loop through your data (in this example, we use random images)
numFrames = 500;
color_list = ["#0072BD", "#EDB120", "#7E2F8E", "#77AC30"];
for frame = 1:numFrames
    figure(1)
    hold on
    plot(tar_x_rec,tar_y_rec,'rp','markersize',10,'MarkerFaceColor','r');
    for iDrone = 1:NDrone
        plot(xy_drones_T(1:frame,1,iDrone),xy_drones_T(1:frame,2,iDrone),'Color', color_list(iDrone),'LineWidth',2);
    end
    plot(Dim/2, Dim/2,'cs','markersize',14,'MarkerFaceColor','c');
    xlb = xlabel('$x$ (m)','interpreter','Latex');
    ylb = ylabel('$y$ (m)','interpreter','Latex');
    set(gca,'TickLabelInterpreter','latex','fontsize',14);
    set([xlb,ylb],'interpreter','Latex','fontsize',18);
    title(strcat('Numframe=', num2str(frame)), 'Interpreter','latex','FontSize',18)
    % Write the frame to the video
    writeVideo(videoWriterObj, getframe(gcf));
    pause(0.01)
end

% Close the video file
close(videoWriterObj);