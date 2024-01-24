% varied mu search based on distance
% targets moving with uniform speed
clear;clc;
load("TarDistLarge.mat")

ini_x = 5000; ini_y = 5000;
maxT = 5000; % length of trajectory
Dim = 10000; 
board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0]; %ZX: boundary
% loc = zeros(maxT+1,2);
% loc(1,:) = [ini_x, ini_y];
R = 25; % detection radius
NumLevy = 1000;
% tm = 300; % memorizing parameter
% Rm_series = [50; 200; 500; 700; 1000; 1500; 2000; 2500; 3000];
Rm_series = [50; 200; 350; 500; 650; 800; 1000; 1200; 1500; 1800];
num_det_tar = zeros(NumLevy,numel(Rm_series));
TotalDist = zeros(NumLevy,numel(Rm_series));
LastTarIndex = zeros(NumLevy,numel(Rm_series));
% translational velocity
ux = 2; uy = 2;

% Define video parameters
outputVideoFile = 'search3.mp4';  % Specify the output video file
frameRate = 60;  % Frames per second
% Create a VideoWriter object
videoWriterObj = VideoWriter(outputVideoFile, 'MPEG-4');
videoWriterObj.FrameRate = frameRate;
% Open the video file for writing
open(videoWriterObj);


for j = 3:3 %numel(Rm_series)
    Rm = Rm_series(j);
    disp('New Rm Parameter')
    tic
    for ii = 1:1 %NumLevy
        Stepfile = load(strcat('StepLength-',num2str(ii),'.mat'));
        Target = [tar_x_rec, tar_y_rec];
        DetTar_rec = [];
        Time_index = [];
        NumTar_index = []; % number of targets detected in one step
        LocDet = [];
        next_step = Stepfile.step_mu1(1);
        sum_dist = zeros(maxT,1);
        loc = zeros(maxT,2);
        sum_dist(1) = 0;
        loc(1,:) = [ini_x, ini_y];

        LastTar = 0;
        for t = 1:maxT-1
            u = cos(Stepfile.direc(t)); %component of the direction vector
            v = sin(Stepfile.direc(t));
            loc(t+1,:) = loc(t,:) + [u*next_step, v*next_step];
            sum_dist(t+1) = sum_dist(t) + next_step;
            locx = loc(t+1,1); locy = loc(t+1,2); xy = [locx, locy];
            Target = [Target(:,1) + ux*next_step, Target(:,2)+uy*next_step]; % targets with uniform speed
            tarx = Target(:,1); tary = Target(:,2);
            if ~inpolygon(xy(1),xy(2),board(:,1),board(:,2))
                xy(xy < 0) = abs(xy(xy < 0));
                if xy(1) > Dim
                    xy(1) = 2*Dim - xy(1);
                end
                if xy(2) > Dim
                    xy(2) = 2*Dim - xy(2);
                end
            end
            tarx(tarx < 0) = Dim-abs(tarx(tarx < 0));
            tary(tary < 0) = Dim-abs(tary(tary < 0));
            tarx(tarx > Dim) = tarx(tarx > Dim)-Dim;
            tary(tary > Dim) = tary(tary > Dim)-Dim;

            loc(t+1,:) = xy;
            Target = [tarx, tary];
            dist = pdist2(loc(t+1,:), Target) - R;
            % ====== varies mu with long memorization ======
            if min(dist) <= 0
                Time_index = [Time_index; t];
                NumTar_index = [NumTar_index; numel(Target(find(dist<0),1))];
                LocDet = [LocDet; loc(t+1,:)];
                NewTarDet = Target(find(dist<0),:);
                DetTar_rec = [DetTar_rec; NewTarDet];
                Target(find(dist<0),:) = []; %  remove the detected targets
                LastTar = t;
            end
            
            if numel(Time_index) >= 1 & pdist2(LocDet(end,:), loc(t+1,:)) - Rm <= 0
                next_step = Stepfile.step_mu2(t+1);
            else
                next_step = Stepfile.step_mu1(t+1);
            end
            
%             if t <= tm % has detected a target in 10 steps
%                 slice = dist_rec(1:t);
%             else
%                 slice = dist_rec(t-tm+1:t);
%             end
%             if any(slice <= 0) == 1
%                 next_step = Stepfile.step_mu2(t+1);
%             else
%                 next_step = Stepfile.step_mu1(t+1);
%             end 
            figure('Visible', 'off')
            hold on
            plot(tarx,tary,'rp','markersize',10,'MarkerFaceColor','r');
            plot(loc(1:t,1),loc(1:t,2),'b-','LineWidth',1.5);
            xlb = xlabel('$x$ (m)','interpreter','Latex');
            ylb = ylabel('$y$ (m)','interpreter','Latex');
            set([xlb,ylb],'interpreter','Latex','fontsize',18);
            xlim([0,10000]); ylim([0,10000])
            title(strcat('Numframe = ', num2str(t)), 'Interpreter','latex','FontSize',18)
            writeVideo(videoWriterObj, getframe(gcf));
%             plot(loc(t+1,1),loc(t+1,2),'b-','LineWidth',1.5);
    
        end
        % Close the video file
        close(videoWriterObj);
        
        num_det_tar(ii,j) = numel(tar_x_rec)- size(Target,1);
        LastTarIndex(ii,j) = LastTar;
        TotalDist(ii,j) = sum_dist(end);
%         if LastTar == 0
%             TotalDist(ii,j) = sum_dist(LastTar+1);
%         else
%             TotalDist(ii,j) = sum_dist(LastTar);
%         end
    end
    toc
%     figure(1);
%     hold on
%     plot(tar_x_rec,tar_y_rec,'rp','markersize',10,'MarkerFaceColor','r');
%     plot(loc(1:maxT,1),loc(1:maxT,2),'b-','LineWidth',1.5);
%     plot(ini_x, ini_y,'cs','markersize',14,'MarkerFaceColor','c');
%     plot(loc(maxT,1),loc(maxT,2),'mo','markersize',10,'MarkerFaceColor','b');
%     xlb = xlabel('$x$ (m)','interpreter','Latex');
%     ylb = ylabel('$y$ (m)','interpreter','Latex');
%     
%     lgd = legend('Targets','Trajectory','Initial location','Last location',...
%      'location','northwest');
%     % legend boxoff
%     set(gca,'TickLabelInterpreter','latex','fontsize',14);
%     set([xlb,ylb],'interpreter','Latex','fontsize',18);
%     set(lgd,'interpreter','Latex','fontsize',12);
end
% save('SearchEffiTarMove2.mat','num_det_tar', 'TotalDist', 'LastTarIndex')
%% search efficiency
NumLevy = size(num_det_tar,1);
Numtm = size(num_det_tar,2);
% Rm_series = [50; 200; 500; 700; 1000; 1500];
Rm_series = [50; 200; 350; 500; 650; 800; 1000; 1200; 1500; 1800];
avg_effi = zeros(Numtm,1); err_effi = zeros(Numtm,1);
TotalDist(TotalDist == 0) = TotalDist(TotalDist == 0)+1;
for i = 1:Numtm
    avg_effi(i) = mean(num_det_tar(:,i)./TotalDist(:,i));
    err_effi(i) = 1.96*std(num_det_tar(:,i)./TotalDist(:,i))/sqrt(1000);
end
figure(1)
errorbar(Rm_series, avg_effi, err_effi, linewidth = 3);
xlabel("$R_m$", 'Interpreter','latex', 'FontSiz',18);
ylabel("$\eta$", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'SearchEfficiencyDist2TarMove2.png')