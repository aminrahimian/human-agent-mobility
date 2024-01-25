clear;clc
close all
% Parameters verification
V_0 = 100; %10;%300;%2.1;
Sigma_series = [25;300;700]; %0.3;%0.8;%0.3;
sigma_alpha = 150;
diff = 1:1:500;
epsilon = 30;
nn = 0.5;
for i = 1:3
    Sigma = Sigma_series(i);
    V_target_gra(i,:) = -V_0/Sigma*exp(-diff./Sigma);
    F_target(i,:) = -V_target_gra(i,:);
end
V_drones_gra = -epsilon*nn*((sigma_alpha.^nn)./(diff.^(nn+1))).*(2*(sigma_alpha.^nn)./(diff.^nn)-1);
F_drones = -V_drones_gra;

figure(1)
hold on
plot(diff, F_drones,LineWidth=3)
for i = 1:3
    plot(diff, F_target(i,:), LineWidth=3)
end
set(gca, 'YScale', 'log');
% ylim([0 100])
ylabel('Force Magnitude','Interpreter','Latex','FontSiz',20);
xlabel('$d$ (m)', 'Interpreter','Latex','FontSiz',20);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
% title(strcat('$\sigma_t=$', num2str(Sigma),',','$\sigma_d=$', num2str(sigma_alpha)), 'Interpreter','latex','FontSize',20)
xline(50, '--', LineWidth=3)
lgd = legend('$\sigma_d=100$','$\sigma_t=25$','$\sigma_t=300$','$\sigma_t=700$','location','northeast');
set(lgd,'interpreter','Latex','fontsize',16);
saveas(gcf, strcat('sigma_t=', num2str(Sigma), ' sigma_d=', num2str(sigma_alpha), '-Para.png'));
%% sigma d
colorlist = [[0 0.4470 0.7410];[0.8500 0.3250 0.0980];[0.9290 0.6940 0.1250];[0.4940 0.1840 0.5560];[0.4660 0.6740 0.1880];...
[0.3010 0.7450 0.9330];[0.6350 0.0780 0.1840]; [0.25 0.25 0.25]];	

figure(2)
hold on
sigma_alpha_series = [50;100;150;200;250;300;350;400];
for i = 1:numel(sigma_alpha_series)
    sigma_alpha = sigma_alpha_series(i);
    V_drones_gra = -epsilon*nn*((sigma_alpha.^nn)./(diff.^(nn+1))).*(2*(sigma_alpha.^nn)./(diff.^nn)-1);
    F_drones = -V_drones_gra;
    plot(diff, F_drones, Color = colorlist(i,:),LineWidth=3)
end
ylabel('Force Magnitude','Interpreter','Latex','FontSiz',20);
xlabel('$d$ (m)', 'Interpreter','Latex','FontSiz',20);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set(gca, 'YScale', 'log');
lgd = legend('$\sigma_d = 50$','$\sigma_d = 100$','$\sigma_d = 150$','$\sigma_d = 200$', ...
    '$\sigma_d = 250$','$\sigma_d = 300$','$\sigma_d = 350$','$\sigma_d = 400$','location','northeast');
set(lgd,'NumColumns',2,'interpreter','Latex','fontsize',16);
saveas(gcf, 'Parad.png');
%% sigma t
colorlist = [[0 0.4470 0.7410];[0.8500 0.3250 0.0980];[0.9290 0.6940 0.1250];[0.4940 0.1840 0.5560];[0.4660 0.6740 0.1880];...
[0.3010 0.7450 0.9330];[0.6350 0.0780 0.1840]; [0.25 0.25 0.25]];	

figure(2)
hold on
sigma_series = [25;50;75;100;150;200;250;300];
for i = 1:numel(sigma_series)
    Sigma = sigma_series(i);
    V_target_gra = -V_0/Sigma*exp(-diff./Sigma);    
    F_target = -V_target_gra;
    plot(diff, F_target, Color = colorlist(i,:),LineWidth=3)
end
ylabel('Force Magnitude','Interpreter','Latex','FontSiz',20);
xlabel('$d$ (m)', 'Interpreter','Latex','FontSiz',20);
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set(gca, 'YScale', 'log');
lgd = legend('$\sigma_t = 25$','$\sigma_t = 50$','$\sigma_t = 75$','$\sigma_t = 100$', ...
    '$\sigma_t = 150$','$\sigma_t = 200$','$\sigma_t = 250$','$\sigma_t = 300$','location','northeast');
set(lgd,'NumColumns',2,'interpreter','Latex','fontsize',16);
ylim([1e-5 8e1])
saveas(gcf, 'Parat.png');