clear;clc
close all
% self-assignment task
% fix the domain size and increase the number of target
% check the optimized sigma_t or sigma_d

%% create randomly uniform target distribution within 2000*2000 m domain
N_group = 1:1:500; 
Dim = 2000; % domain size
figure(1)
hold on
AvgDist = Dim./sqrt(N_group);
plot(N_group, AvgDist, LineWidth=3)
% yline(1000./sqrt(50), LineWidth=3)
xlabel("$N_{target}$", 'Interpreter','latex', 'FontSiz',18);
ylabel('$\frac{D}{\sqrt{N_{target}}}$', 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'AvgDist-N.png')


N_series = [50;100;150;200;400];
for i = 5:numel(N_series)
    N = N_series(i); % number of targets 
    tar_x_rec = rand(1,N)*Dim;
    tar_y_rec = rand(1,N)*Dim;
    tar_x_rec = tar_x_rec';
    tar_y_rec = tar_y_rec';
    save(strcat('TarDim=',num2str(Dim),'N=', num2str(N),'.mat'),'tar_x_rec','tar_y_rec');
end
%% motion of 4 drones
NDrone = 4;  Dim = 2000; N=400; 
N_series = [50;100;150;200]; 
mu1 = 1; maxT = 1000; maxL = 400; % cut-off length
R = 25; % detection radius
board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0]; % boundary

social_alpha = 150; epsilon = 30; nn = 0.5;
social_alpha = 100;
Sigma = 75; V_0 = 100;
N_search = 1000;

social_alpha_series = [50;100;150;200;250;300;350;400];
% Sigma_series = [25;50;75;100;150;200;250;300];
Sigma_series = [25;50;100;200;300;500;700;900]; % for 5000 domain


load(strcat('TarDim=',num2str(Dim),'N=', num2str(N),'.mat'))
xy_target_ini = [tar_x_rec, tar_y_rec];

for i_social = 1:numel(social_alpha_series)
    social_alpha = social_alpha_series(i_social); % L-J potential
    for i_Sigma = 1:numel(Sigma_series)
        Sigma = Sigma_series(i_Sigma); % exponential potential
        Ntarget_det = zeros(N_search,1);
        LastTar_det = zeros(N_search,1);
        tic
        for isearch = 1:N_search
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
            limitT = 500; %TASK TIME
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
            Ntarget_det(isearch) = N - size(xy_target,1);
            LastTar_det(isearch) = sum(sum(TotalDist(:,1:LastTar)));
        end
        toc
        save(strcat('N=', num2str(N),'SigmaD=', num2str(social_alpha),'SigmaT=', num2str(Sigma),'.mat'), 'Ntarget_det', 'LastTar_det');
    end
end
%% check the efficiency
num_social = numel(Sigma_series);
avg_effi = zeros(num_social,1); 
err_effi = zeros(num_social,1);
tot_dist = zeros(num_social,1);
err_dist = zeros(num_social,1);
num_tar = zeros(num_social,1);
err_tar = zeros(num_social,1);
for i = 1:num_social
    NumCollectTar = Ntarget_det(i,:); %[NumTar(:,i); num_det_tar(:,i)];
    LastStep = LastTar_det(i,:); %[LastTar(:,i); Last_tar_index(:,i)];
    TotalTime = LastStep;

    NumCollectTar(LastStep == 0) = [];
    TotalTime(LastStep == 0) = [];
    disp(numel(TotalTime))
    avg_effi(i) = mean(NumCollectTar./TotalTime);
    err_effi(i) = 1.96*std(NumCollectTar./TotalTime)/sqrt(numel(TotalTime));
    tot_dist(i) = mean(TotalTime);
    err_dist(i) = 1.96*std(TotalTime)/sqrt(numel(TotalTime));
    num_tar(i) = mean(NumCollectTar);
    err_tar(i) = 1.96*std(NumCollectTar)/sqrt(numel(NumCollectTar));
end

figure(1)
hold on
% errorbar(social_alpha_series, avg_effi, err_effi, linewidth = 3);
errorbar(Sigma_series, avg_effi, err_effi, linewidth = 3);
xlabel("$N$", 'Interpreter','latex', 'FontSiz',18);
ylabel("$\eta$", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'SearchEfficiency-Sigma.png')

figure(2)
hold on
errorbar(Sigma_series, tot_dist, err_dist, linewidth = 3);
xlabel("$N$", 'Interpreter','latex', 'FontSiz',18);
ylabel("Total distance (m)", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'TotalDist-Sigma.png')

figure(3)
hold on
errorbar(Sigma_series, num_tar, err_tar, linewidth = 3);
xlabel("$N$", 'Interpreter','latex', 'FontSiz',18);
ylabel('Number of targets', 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'NumTar-Sigma.png')