clear;clc;

% This code discretize the trajectories of robot and soldier into
% consecutive locations 1 second by 1 second.

load("MovementPatterm.mat") % pre-gernerate movement of soldier
Num_SingleLevy = 20;
% generate discrete locations of soldier
for i = 11:Num_SingleLevy
    StepLength = SingleLevyMove(:,1,i);
    Direc = SingleLevyMove(:,2,i);
    lag = 1;
    xy_pede_old = [0, 0];
    xy_pede_save = [];
    tic
    for j = 1:80000
        u = cos(Direc(j)); %component of the direction vector
        v = sin(Direc(j));
        xy_pede_old = xy_pede_old + (1-lag)*[u,v];
        xy_bot_new = xy_pede_old + (StepLength(j)-(1-lag))*[u,v];

        num_steps = fix(StepLength(j)-(1-lag)); %robot steps
        xy_pede_inte = xy_pede_old + num_steps*[u,v];
        xy_pede_steps = [linspace(xy_pede_old(1),xy_pede_inte(1),num_steps+1);...
            linspace(xy_pede_old(2),xy_pede_inte(2),num_steps+1)];
        xy_pede_steps = xy_pede_steps';
        xy_pede_save = [xy_pede_save; xy_pede_steps(1:end,:)];
        xy_pede_old = xy_bot_new;
        lag = (StepLength(j)-(1-lag)) - num_steps; %the remain time
    end
    toc
    save(strcat('SoldierLocation-',num2str(i),'.mat'),'xy_pede_save');
end
%% generate discrete locations of robot
clear; clc;
Num_MultiLevy = 30;
window_size_series = 500:500:5000;
mu_bot = 1;
Dim = 100000;
for k = 1:numel(window_size_series)
    x_window = window_size_series(k);
    maxRobot = x_window;
    board_window = [0,0;...
                x_window,0;...
                x_window,x_window;...
                0,x_window;...
                0,0];
    for i = 1:Num_MultiLevy
        theta_bot = rand(80000,1)*2*pi;
        step_bot = RandTruncLevyMulti(mu_bot, maxRobot, 80000)+1;
        lag = 1;
        xy_bot_old = [0, 0];
        xy_bot_save = [];
        tic
        for j = 1:80000
            u = cos(theta_bot(j)); %component of the direction vector
            v = sin(theta_bot(j));
            xy_bot_old = xy_bot_old + (1-lag)*[u,v];
            if ~inpolygon(xy_bot_old(1),xy_bot_old(2),board_window(:,1),board_window(:,2))
                xy_bot_old(xy_bot_old < -x_window/2) = abs(xy_bot_old(xy_bot_old < -x_window/2));
                xy_bot_old(xy_bot_old > x_window/2) = x_window - xy_bot_old(xy_bot_old > x_window/2);
            end
            xy_bot_new = xy_bot_old + (step_bot(j)-(1-lag))*[u,v];
            if ~inpolygon(xy_bot_new(1),xy_bot_new(2),board_window(:,1),board_window(:,2))
                xy_bot_new(xy_bot_new < -x_window/2) = abs(xy_bot_new(xy_bot_new < -x_window/2));
                xy_bot_new(xy_bot_new > x_window/2) = x_window - xy_bot_new(xy_bot_new > x_window/2);
            end
            num_steps = fix(step_bot(j)-(1-lag)); %robot steps
            xy_bot_inte = xy_bot_old + num_steps*[u,v];
            if ~inpolygon(xy_bot_inte(1),xy_bot_inte(2),board_window(:,1),board_window(:,2))
                xy_bot_inte(xy_bot_inte < -x_window/2) = abs(xy_bot_inte(xy_bot_inte < -x_window/2));
                xy_bot_inte(xy_bot_inte > x_window/2) = x_window - xy_bot_inte(xy_bot_inte > x_window/2);
            end

            xy_bot_steps = [linspace(xy_bot_old(1),xy_bot_inte(1),num_steps+1);...
                linspace(xy_bot_old(2),xy_bot_inte(2),num_steps+1)];
            xy_bot_steps = xy_bot_steps';
            xy_bot_save = [xy_bot_save; xy_bot_steps(1:end,:)];
            xy_bot_old = xy_bot_new;
            lag = (step_bot(j)-(1-lag)) - num_steps; %the remain time
        end
        toc
        save(strcat('RobotLocation-window-',num2str(x_window), '-',num2str(i),'.mat'), 'xy_bot_save');
    end
end