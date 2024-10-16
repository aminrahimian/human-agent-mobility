% varied mu search with memorizing parameter tm
clear;clc;
% levy search with two mu
mu1 = 1; mu2 = 2;
maxL = 500; % cut-off length
maxT = 100000; % length of trajectory
step_mu1 = zeros(maxT, 1);
step_mu2 = zeros(maxT, 1);
for i = 1:maxT
    step_mu1(i) = RandTruncLevyAlpha(mu1, maxL-25)+25;
    step_mu2(i) = RandTruncLevyAlpha(mu2, maxL-25)+25;
end
direc = rand(maxT,1)*2*pi; %random direction

% [bin_hight,bin_center]=pdf_plotter_new(1000,step_mu1);
% y=bin_hight';
% x=bin_center';
% % normalize pdf to sum to 1.
% dx = x(2) - x(1);
% sum_pdf = sum(dx.* y);
% y = y./sum_pdf;
% 
% figure(1)
% hold on
% plot(x,y,LineWidth=3)
% [bin_hight,bin_center]=pdf_plotter_new(1000,step_mu2);
% y=bin_hight';
% x=bin_center';
% % normalize pdf to sum to 1.
% dx = x(2) - x(1);
% sum_pdf = sum(dx.* y);
% y = y./sum_pdf;
% plot(x,y, LineWidth=3)
% set(gca,'yscale','log')
% set(gca,'xscale','log')
% xlb = xlabel('Step length','interpreter','Latex');
% ylb = ylabel('PDF','interpreter','Latex');
% lgd = legend('$\mu$ = 2','$\mu$ = 3','location','southwest');
% % legend boxoff
% set(gca,'TickLabelInterpreter','latex','fontsize',14);
% set([xlb,ylb],'interpreter','Latex','fontsize',18);
% set(lgd,'interpreter','Latex','fontsize',14);
% saveas(gcf,'mu12.png')
save('StepLength-1.mat','step_mu1','step_mu2', 'direc');
%% introduce target distribution
load("TarDist3.mat")
load("StepLength-1.mat")
% use mu=2 when the agent does not detect a target;
% use mu=1 after detecting a target;
% when the agent does not detect another target in 10 steps, mu will change
% from 2 to 1.

ini_x = 5000; ini_y = 5000;
maxT = 100000; % length of trajectory
loc = zeros(maxT+1,2);
loc(1,:) = [ini_x, ini_y];
R = 25;
Target = [tar_x_rec, tar_y_rec];
next_step = step_mu1(1);
dist_rec = zeros(maxT,1);
tm = 300; % memorizing parameter
tm_series = [2;5;7;10; 20; 50; 100; 200; 300; 500; 700; 1000; 1500; 2000;2500];
num_det_tar = zeros(numel(tm_series),1);

for k = 1:numel(tm_series)
    tm = tm_series(k);
    Target = [tar_x_rec, tar_y_rec];
    dist_rec = zeros(maxT,1);
    next_step = step_mu1(1);
    loc = zeros(maxT+1,2);
    loc(1,:) = [ini_x, ini_y];
for t = 1:maxT-1
    u = cos(direc(t)); %component of the direction vector
    v = sin(direc(t));
    loc(t+1,:) = loc(t,:) + [u*next_step, v*next_step];
    dist = pdist2(loc(t+1,:), Target) - R;
    dist_rec(t) = min(dist);
    % ====== varies mu with long memorization ======
    if min(dist) <= 0
        Target(find(dist<0),:) = [];
    end

    if t <= tm % has detected a target in 10 steps
        slice = dist_rec(1:t);
    else
        slice = dist_rec(t-tm+1:t);
    end
    if any(slice <= 0) == 1
        next_step = step_mu2(t+1);
    else
        next_step = step_mu1(t+1);
    end
%     
% %     ====== varies mu with short memorization ======
%     if min(dist) <= 0
%         Target(find(dist<0),:) = [];
%         next_step = step_mu2(t+1);
%     else
%         next_step = step_mu1(t+1);
%     end
%     % ====== mu = 1 ======
%     if min(dist) <= 0
%         Target(find(dist<0),:) = [];
%     end
%     next_step = step_mu1(t+1);

%     % ====== mu = 2 ======
%     if min(dist) <= 0
%         Target(find(dist<0),:) = [];
%     end
%     next_step = step_mu2(t+1);
end

figure('visible','off');
hold on
plot(Target(:,1),Target(:, 2),'rp','markersize',10,'MarkerFaceColor','r');
plot(loc(1:maxT,1),loc(1:maxT,2),'b-','LineWidth',1.5);
plot(ini_x, ini_y,'cs','markersize',14,'MarkerFaceColor','c');
plot(loc(maxT,1),loc(maxT,2),'mo','markersize',10,'MarkerFaceColor','m');

xlb = xlabel('$x$ (m)','interpreter','Latex');
ylb = ylabel('$y$ (m)','interpreter','Latex');

lgd = legend('Targets','Trajectory','Initial location','Last location',...
 'location','northwest');
% legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set([xlb,ylb],'interpreter','Latex','fontsize',18);
set(lgd,'interpreter','Latex','fontsize',12);
xlim([1000, 13500]);
ylim([-500,9500]);
% saveas(gcf, 'CombinedLevy-mu1.png')
saveas(gcf,strcat('CombinedLevy-tm-',num2str(tm),'.png'))
num_det_tar(k) = numel(tar_x_rec)- size(Target,1);
end 
%% variation of search efficiency with memorizing parameter tm
tm_series = [1; tm_series];
num_det_tar = [17; num_det_tar];
figure(1)
hold on
plot(tm_series, num_det_tar,'b-', LineWidth=3);
yline(22, 'r--', LineWidth=3);
xlb = xlabel('$t_m$','interpreter','Latex');
ylb = ylabel('Number of detected targets','interpreter','Latex');
lgd = legend('varied $\mu$','$\mu = 2$',...
 'location','southeast');
% legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set([xlb,ylb],'interpreter','Latex','fontsize',18);
set(lgd,'interpreter','Latex','fontsize',14);
set(gca,'xscale','log')
saveas(gcf,'tm-num-target.png')
%% fix tm, calculate search efficiency of varied \mu search
% Create 100 groups of step length
NumLevy = 1000;
mu1 = 1; mu2 = 2;
maxL = 500; % cut-off length
maxT = 100000; % length of trajectory
for ii = 1:NumLevy
    tic
    step_mu1 = zeros(maxT, 1);
    step_mu2 = zeros(maxT, 1);
    for i = 1:maxT
        step_mu1(i) = RandTruncLevyAlpha(mu1, maxL-25)+25;
        step_mu2(i) = RandTruncLevyAlpha(mu2, maxL-25)+25;
    end
    direc = rand(maxT,1)*2*pi; %random direction
    save(strcat('StepLength-',num2str(ii+2000),'.mat'),'step_mu1','step_mu2', 'direc');
    toc
end
%%
clear;clc;
load("TarDistLarge.mat")
ini_x = 5000; ini_y = 5000;
maxT = 50000; % length of trajectory
Dim = 10000;
board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0]; %ZX: boundary
% loc = zeros(maxT+1,2);
% loc(1,:) = [ini_x, ini_y];
R = 25;
NumLevy = 1000;
% tm = 300; % memorizing parameter
tm_series = [50; 200; 500; 700; 1000; 1500; 2000; 2500; 3000];
num_det_tar = zeros(NumLevy,numel(tm_series));
Last_tar_index = zeros(NumLevy,numel(tm_series));
for j = 1:numel(tm_series)
    tm = tm_series(j);
    disp('New tm Parameter')
    parfor ii = 1:NumLevy
        tic
        Stepfile = load(strcat('StepLength-',num2str(ii+2000),'.mat'));
        Target = [tar_x_rec, tar_y_rec];
        dist_rec = zeros(maxT,1);
        next_step = Stepfile.step_mu1(1);
        loc = zeros(maxT+1,2);
        loc(1,:) = [ini_x, ini_y];
        LastTar = 0;
        for t = 1:maxT-1
            u = cos(Stepfile.direc(t)); %component of the direction vector
            v = sin(Stepfile.direc(t));
            loc(t+1,:) = loc(t,:) + [u*next_step, v*next_step];
            locx = loc(t+1,1); locy = loc(t+1,2); xy = [locx, locy];
            if ~inpolygon(xy(1),xy(2),board(:,1),board(:,2))
                xy(xy < 0) = abs(xy(xy < 0));
                if xy(1) > Dim
                    xy(1) = 2*Dim - xy(1);
                end
                if xy(2) > Dim
                    xy(2) = 2*Dim - xy(2);
                end
            end
            loc(t+1,:) = xy;
            dist = pdist2(loc(t+1,:), Target) - R;
            dist_rec(t) = min(dist);
            % ====== varies mu with long memorization ======
            if min(dist) <= 0
                Target(find(dist<0),:) = [];
                LastTar = t;
            end
        
            if t <= tm % has detected a target in 10 steps
                slice = dist_rec(1:t);
            else
                slice = dist_rec(t-tm+1:t);
            end
            if any(slice <= 0) == 1
                next_step = Stepfile.step_mu2(t+1);
            else
                next_step = Stepfile.step_mu1(t+1);
            end
        %     
        % %     ====== varies mu with short memorization ======
        %     if min(dist) <= 0
        %         Target(find(dist<0),:) = [];
        %         next_step = step_mu2(t+1);
        %     else
        %         next_step = step_mu1(t+1);
        %     end

        %     % ====== mu = 1 ======
        %     if min(dist) <= 0
        %         Target(find(dist<0),:) = [];
        %     end
        %     next_step = step_mu1(t+1);
        
        %     % ====== mu = 2 ======
        %     if min(dist) <= 0
        %         Target(find(dist<0),:) = [];
        %     end
        %     next_step = step_mu2(t+1);
        end
    
        num_det_tar(ii,j) = numel(tar_x_rec)- size(Target,1);
        Last_tar_index(ii,j) = LastTar;
        toc
    end

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
%     saveas(gcf, strcat('tm = ', num2str(tm),'-trajectories.png'))
%     close(gcf)
end
save('SearchEffiLargeMore5.mat','num_det_tar', 'Last_tar_index')


%% search efficiency
NumLevy = size(num_det_tar,1);
Numtm = size(num_det_tar,2);
tm_series = [50; 200; 500; 700; 1000; 1500; 2000; 2500; 3000];
avg_effi = zeros(Numtm,1); err_effi = zeros(Numtm,1);
Last_tar_index(Last_tar_index == 0) = Last_tar_index(Last_tar_index == 0)+1;
for i = 1:Numtm
    avg_effi(i) = mean(num_det_tar(:,i)./Last_tar_index(:,i));
    err_effi(i) = 1.96*std(num_det_tar(:,i)./Last_tar_index(:,i))/sqrt(1000);
end
figure(1)
errorbar(tm_series, avg_effi, err_effi, linewidth = 3);
xlabel("$t_m$", 'Interpreter','latex', 'FontSiz',18);
ylabel("$\eta$", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'SearchEfficiencyLargeMore.png')
%% MSD
load("TarDist3.mat")
load("SearchEffi3.mat")
ini_x = 5000; ini_y = 5000;
loc = zeros(maxT+1,2);
loc(1,:) = [ini_x, ini_y];
R = 25;
Target = [tar_x_rec, tar_y_rec];

% tm = 300; % memorizing parameter
tm_series = [200; 300; 500; 700; 1000; 1500; 2000; 2500];
NumLevy = 20;
num_det_tar = zeros(NumLevy,numel(tm_series));
MSD = cell(NumLevy,numel(tm_series));
for j = 1:numel(tm_series)
    tm = tm_series(j);
    for ii = 1:NumLevy
        tic
        load(strcat('StepLength-',num2str(ii),'.mat'));
        Target = [tar_x_rec, tar_y_rec];
        maxT = Last_tar_index(ii,j+7);
        dist_rec = zeros(cdfre54,1);
        next_step = step_mu1(1);
        loc = zeros(maxT+1,2);
        loc(1,:) = [ini_x, ini_y];
        for t = 1:maxT-1
            u = cos(direc(t)); %component of the direction vector
            v = sin(direc(t));
            loc(t+1,:) = loc(t,:) + [u*next_step, v*next_step];
            dist = pdist2(loc(t+1,:), Target) - R;
            dist_rec(t) = min(dist);
            % ====== varies mu with long memorization ======
            if min(dist) <= 0
                Target(find(dist<0),:) = [];
            end
        
            if t <= tm % has detected a target in 10 steps
                slice = dist_rec(1:t);
            else
                slice = dist_rec(t-tm+1:t);
            end
            if any(slice <= 0) == 1
                next_step = step_mu2(t+1);
            else
                next_step = step_mu1(t+1);
            end
        end
    
        num_det_tar(ii,j) = numel(tar_x_rec)- size(Target,1);
        save(strcat('Location-tm-',num2str(tm),'-Tra-',num2str(ii),'.mat'), 'loc')
        % calculate MSD based on trajectories.
        upperlimit = length(loc(:,1))-1;
        msd = zeros(floor(upperlimit/4),3); % save msd, std, and #n
        for dt = 1:floor(upperlimit/4)
            diff = loc(1+dt:end,1:2) - loc(1:end-dt,1:2);
            squaredDisp = sum(diff.^2,2); %dx^2+dy^2
            msd(dt,1) = mean(squaredDisp); % average
            msd(dt,2) = std(squaredDisp);
            msd(dt,3) = length(squaredDisp);
        end
        MSD{ii,j} = msd;
        toc
    end
end
%% plot MSD
cmap_use = jet(NumLevy);
Numtm = numel(tm_series);
para_series = zeros(Numtm,1);
for i = 1:Numtm
    para = zeros(NumLevy,2);
    for j = 1:NumLevy
        msd = MSD{j,i};
        delay = size(msd(:,1),1);
        figure(1)
        hold on
        plot(linspace(1,delay,delay),msd(:,1),Color=cmap_use(j,:),LineWidth=3)
        x = linspace(1,delay,delay); y = msd(:,1);
        p = polyfit(log10(x(10:200)), log10(y(10:200)), 1);
        para(j,:) = p;
    end
    xfit = x(10:200);
    yfit = xfit.^mean(para(:,1)) * 10^mean(para(:,2));
    para_series(i) = mean(para(:,1));
    plot(xfit, yfit, "k--", LineWidth=3)
    % name_ = strcat('slope = ', num2str(mean(para(:,1))));
    % annotation('textarrow', [0.2, 0.4], [0.5, 0.4],'String', name_);
    set(gca,'yscale','log')
    set(gca,'xscale','log')
    ylabel('$\left\langle (x(t)-x_0)^2\right\rangle$','Interpreter','Latex','FontSiz',18);
    xlabel('$t$', 'Interpreter','Latex','FontSiz',18);
    saveas(gcf, strcat('MSD-varied mu search-tm-.',num2str(tm_series(i)),'.png'));
    disp(mean(para(:,1)))
end
figure(2)
plot(tm_series, para_series, LineWidth=3)
ylabel('Diffusion Coefficient','Interpreter','Latex','FontSiz',18);
xlabel('$t_m$', 'Interpreter','Latex','FontSiz',18);
saveas(gcf, strcat('Dc-tm3.png'));