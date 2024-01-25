% find the optimal in self-assignment tasks
N = 100;
N_series = [50;100;200];
social_alpha_series = [50;100;150;200;250;300;350;400];
Sigma_series = [25;50;100;200;300;500;700;900]; % for 5000 domain
num_Sigma = numel(Sigma_series);
num_social = numel(social_alpha_series);
avg_effi = zeros(num_Sigma,num_social); 
err_effi = zeros(num_Sigma,num_social);
tot_dist = zeros(num_Sigma,num_social);
err_dist = zeros(num_Sigma,num_social);
num_tar = zeros(num_Sigma,num_social);
err_tar = zeros(num_Sigma,num_social);

for k = 1:numel(N_series)
    N = N_series(k);
    for i = 1:num_Sigma
        Sigma = Sigma_series(i);
        for j = 1:num_social
            social_alpha = social_alpha_series(j);
            load(strcat('N=', num2str(N),'SigmaD=', num2str(social_alpha),'SigmaT=', num2str(Sigma),'.mat'), 'Ntarget_det', 'LastTar_det');
            NumCollectTar = Ntarget_det(:); %[NumTar(:,i); num_det_tar(:,i)];
            LastStep = LastTar_det(:); %[LastTar(:,i); Last_tar_index(:,i)];
            TotalTime = LastStep;
            NumCollectTar(LastStep == 0) = [];
            TotalTime(LastStep == 0) = [];
            disp(numel(TotalTime))
            avg_effi(i,j) = mean(NumCollectTar./TotalTime);
            err_effi(i,j) = 1.96*std(NumCollectTar./TotalTime)/sqrt(numel(TotalTime));
            tot_dist(i,j) = mean(TotalTime);
            err_dist(i,j) = 1.96*std(TotalTime)/sqrt(numel(TotalTime));
            num_tar(i,j) = mean(NumCollectTar);
            err_tar(i,j) = 1.96*std(NumCollectTar)/sqrt(numel(NumCollectTar));
        end
    end

figure(8)
hold on
s = surf(social_alpha_series, Sigma_series, avg_effi);
s.EdgeColor = 'none';
s.FaceColor = 'interp';
end

xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',20);
ylabel('$\sigma_t$', 'Interpreter','latex', 'FontSiz',20);
zlabel("$\eta$", 'Interpreter','latex', 'FontSiz',20);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'SearchEfficiency-surf.png')
%% plot
close all
colors = [
    0.1216, 0.4667, 0.7059; % Blue
    1, 0.4980, 0.0549; % Orange
    0.1725, 0.6275, 0.1725; % Green
    0.8392, 0.1529, 0.1569; % Red
    0.5804, 0.4039, 0.7412; % Purple
    0.5490, 0.3373, 0.2941; % Brown
    0.8902, 0.4667, 0.7608; % Pink
    0.4980, 0.4980, 0.4980; % Gray
];

figure(1)
hold on
for i = 1:num_Sigma
    errorbar(social_alpha_series, avg_effi(i,:), err_effi(i,:), color=colors(i,:), linewidth = 3);
end
lgd = legend('$\sigma_t = 25$','$\sigma_t = 50$','$\sigma_t = 100$','$\sigma_t = 200$',...
 '$\sigma_t = 300$','$\sigma_t = 500$','$\sigma_t = 700$','$\sigma_t = 900$','location','northeast', 'NumColumns',3);
set(lgd,'interpreter','Latex','fontsize',16);
% set(gca, 'YScale', 'log');
ylim([0, 1e-2])
xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel("$\eta$", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'SearchEfficiency-SigmaD.png')

figure(2)
hold on
for i = 1:num_Sigma
    errorbar(social_alpha_series, tot_dist(i,:), err_dist(i,:), color=colors(i,:), linewidth = 3);
end
lgd = legend('$\sigma_t = 25$','$\sigma_t = 50$','$\sigma_t = 100$','$\sigma_t = 200$',...
 '$\sigma_t = 300$','$\sigma_t = 500$','$\sigma_t = 700$','$\sigma_t = 900$','location','northeast', 'NumColumns',3);
set(lgd,'interpreter','Latex','fontsize',16);
ylim([2e4, 6e4])
xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel("Total distance (m)", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'TotalDist-SigmaD.png')

figure(3)
hold on
for i = 1:num_Sigma
    errorbar(social_alpha_series, num_tar(i,:), err_tar(i,:), color=colors(i,:), linewidth = 3);
end
lgd = legend('$\sigma_t = 25$','$\sigma_t = 50$','$\sigma_t = 100$','$\sigma_t = 200$',...
 '$\sigma_t = 300$','$\sigma_t = 500$','$\sigma_t = 700$','$\sigma_t = 900$','location','northeast', 'NumColumns',3);
set(lgd,'interpreter','Latex','fontsize',16);
ylim([0, 300])
xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel('Number of targets', 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'NumTar-SigmaD.png')

figure(4)
hold on
contourf(social_alpha_series',Sigma_series, avg_effi, 15, LineWidth=0.2)
% set(gca, 'YScale', 'log');
colormap('jet')
colorbar
xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel('$\sigma_t$', 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'Effi-SigmaD&T.png')

figure(5)
hold on
contourf(social_alpha_series',Sigma_series, tot_dist, 15, LineWidth=0.2)
% set(gca, 'YScale', 'log');
colormap('jet')
colorbar
xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel('$\sigma_t$', 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'Dist-SigmaD&T.png')

figure(6)
hold on
contourf(social_alpha_series',Sigma_series, num_tar, 15, LineWidth=0.2)
% set(gca, 'YScale', 'log');
colormap('jet')
colorbar
xlabel("$\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel('$\sigma_t$', 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'NumTar-SigmaD&T.png')
%% explore the ratio of sigma_t to sigma_d
sigma_ratio = zeros(num_Sigma,num_social); % sigma_t/sigma_d
for i = 1:num_Sigma
    Sigma = Sigma_series(i);
    for j = 1:num_social
        social_alpha = social_alpha_series(j);
        load(strcat('N=', num2str(N),'SigmaD=', num2str(social_alpha),'SigmaT=', num2str(Sigma),'.mat'), 'Ntarget_det', 'LastTar_det');
        NumCollectTar = Ntarget_det(:); %[NumTar(:,i); num_det_tar(:,i)];
        LastStep = LastTar_det(:); %[LastTar(:,i); Last_tar_index(:,i)];
        TotalTime = LastStep;
        NumCollectTar(LastStep == 0) = [];
        TotalTime(LastStep == 0) = [];
        disp(numel(TotalTime))
        sigma_ratio(i,j) = Sigma/social_alpha;
        avg_effi(i,j) = mean(NumCollectTar./TotalTime);
        err_effi(i,j) = 1.96*std(NumCollectTar./TotalTime)/sqrt(numel(TotalTime));
        tot_dist(i,j) = mean(TotalTime);
        err_dist(i,j) = 1.96*std(TotalTime)/sqrt(numel(TotalTime));
        num_tar(i,j) = mean(NumCollectTar);
        err_tar(i,j) = 1.96*std(NumCollectTar)/sqrt(numel(NumCollectTar));
    end
end

figure(7)
hold on
for i = 1:num_Sigma
    errorbar(sigma_ratio(i,:), avg_effi(i,:), err_effi(i,:), color=colors(i,:), linewidth = 3);
end
lgd = legend('$\sigma_t = 25$','$\sigma_t = 50$','$\sigma_t = 100$','$\sigma_t = 200$',...
 '$\sigma_t = 300$','$\sigma_t = 500$','$\sigma_t = 700$','$\sigma_t = 900$','location','northeast', 'NumColumns',3);
set(lgd,'interpreter','Latex','fontsize',16);
set(gca, 'XScale', 'log');
ylim([0, 1e-2])
xlabel("$\sigma_t/\sigma_d$", 'Interpreter','latex', 'FontSiz',18);
ylabel("$\eta$", 'Interpreter','latex', 'FontSiz',18);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
saveas(gcf, 'SearchEfficiency-SigmaRatio.png')