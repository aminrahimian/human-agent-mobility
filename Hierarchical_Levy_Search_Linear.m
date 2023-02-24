%This file calculate the search time of single agent or single robot
%through linear addition of pre-generated positions.
%During each search, several targets can be detected.

clear; clc;

% ===================Parameters======================
Dim = 100000; %domain size
% x_window_series = 6000:1000:10000;
% x_window_series = 20:30:50;
x_window_series = 500; %window size
NumTarget = 10; %number of targets
NumSingle = 10; %number of single agent's movement
NumMulti = 10;%number of robot's movement
initial_pos = [Dim/2, Dim/2];
sec = 0;
board = [sec,sec;Dim-sec,sec;Dim-sec, Dim-sec;sec,Dim-sec;sec,sec;]; %search board
R = 50; %detection radius

%========Load files and/or Create new files=========
load("TargetDistribution.mat") % load target distribution
SingleTimeLength = cell(NumTarget, NumSingle);
MultiTimeLength = cell(NumTarget, NumSingle, NumMulti);
% save(strcat('SingleLevyTime_Linear_Multi.mat'),'SingleTimeLength');
load('SingleLevyTime_Linear_Multi.mat') %load exist single agent searcb time

%====================Calculation====================
for j = 1:1 %NumSingle
    load(strcat('SoldierLocation-',num2str(j),'.mat')); % load the movement of soldier
    for i =  1:1 %NumTarget
        Real_Target = TarDist(:,:,i);
        in = inpolygon(Real_Target(:,1),Real_Target(:,2),board(:,1),board(:,2)); %targets in the centered 1/4 area
        Num_Search = numel(Real_Target(in,1));
        Search_Target = Real_Target(in,:);
        t_index_pede = zeros(Num_Search, 1);
        xy_pede_real = initial_pos + xy_pede_save; %trajectory of soldier
        
        % find when the soldier detect the target
        for mm = 1:Num_Search
            dist = vecnorm(xy_pede_real - Search_Target(mm,:),2,2) - R; %distance between the soldier and all the targets
            if dist > 0
                t_index_pede(mm) = 1e8; %cannot detect any target
                continue
            end
            t_index_pede(mm) = find(dist <= 0, 1); %find when the soldier detect one target
        end
        t_index_pede = t_index_pede(t_index_pede<1e8); %indices of detection
        if numel(t_index_pede) == 0 %soldier cannot find any targets
            SingleTimeLength{i,j} = 0;
%             save(strcat('SingleLevyTime_Linear_Multi.mat'),'SingleTimeLength');
            continue
        end
        t_mark = sort(t_index_pede); 
        T_record_pede = [t_mark(1); t_mark(2:end) - t_mark(1:end-1)];%record the search time
        SingleTimeLength{i,j} = T_record_pede/3600;
        save(strcat('SingleLevyTime_Linear_Multi.mat'),'SingleTimeLength');
        
        % find when the robot detect the target with different window size
        for m = 1:numel(x_window_series)
            x_window = x_window_series(m);
            save(strcat('MultiLevyTimeMulti-window-',num2str(x_window),'.mat'),'MultiTimeLength');
            load(strcat('MultiLevyTimeMulti-window-', num2str(x_window),'.mat'));
            tic
            for k = 1:NumMulti
                bot_move = load(strcat('RobotLocation-window-',num2str(x_window), '-', num2str(k),'.mat'));
                xy_bot_save = bot_move.xy_bot_save;
                ending = min(numel(xy_bot_save(:,1)), numel(xy_pede_real(:,1)));
                xy_bot_real = xy_pede_real(1:ending,:) + xy_bot_save(1:ending, :);
                t_index_bot = zeros(Num_Search, 1);
                for kk = 1:Num_Search
                    dist = vecnorm(xy_bot_real - Search_Target(kk,:),2,2) - R;
                    if dist > 0
                        t_index_bot(kk) = 1e8;
                        continue
                    end
                    t_index_bot(kk) = find(dist <= 0, 1);
                end
                t_index_bot = t_index_bot(t_index_bot<1e8);
                if numel(t_index_bot) == 0 %robot cannot find any targets
                    MultiTimeLength{i,j,k} = 0;
                    continue
                end
                t_mark_bot = sort(t_index_bot);
                T_record_bot = [t_mark_bot(1); t_mark_bot(2:end) - t_mark_bot(1:end-1)];
                
                MultiTimeLength{i,j,k} = T_record_bot/3600;
            end
%             save(strcat('MultiLevyTimeMulti-window-',num2str(x_window),'.mat'),'MultiTimeLength');
            toc
        end
    end
end