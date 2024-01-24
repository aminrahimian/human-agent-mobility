clear;clc
% Postprocess: plot the distribution of traveling distance/lapsing time between
% two successive detected targets
load("ParSet.mat") % recorded data --- saved by CombinedLevySearch_DistPar
NumLevy = 3000;

Lapsing_time = [];
Lapsing_distance = [];

for i = 1:NumLevy
    Time_array = cell2mat(TimeLapse(i));
    Dist_array = cell2mat(TotalDist(i));
    if numel(Time_array) > 1
        Lapsing_time = [Lapsing_time; Time_array(2:end)-Time_array(1:end-1)];
        Lapsing_distance = [Lapsing_distance; Dist_array(2:end) - Dist_array(1:end-1)];
    end
end

figure(1)
[bin_hight,bin_center] = pdf_plotter_new(500,Lapsing_time(Lapsing_time<=2000));
plot(bin_center, bin_hight, LineWidth=3)
xlb = xlabel('Lapsing Timesteps','interpreter','Latex');
ylb = ylabel('PDF','interpreter','Latex');
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set([xlb,ylb],'interpreter','Latex','fontsize',18);
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
saveas(gcf, 'PDF-Lapsing_time.png')

figure(2)
hold on
[bin_hight,bin_center] = pdf_plotter_new(500,Lapsing_distance(Lapsing_distance<=1e5));
plot(bin_center, bin_hight, LineWidth=3)
plot(bin_center(10:400), 30*bin_center(10:400).^-1.65, 'r--', LineWidth=3)
xlb = xlabel('Traveling Distance (m)','interpreter','Latex');
ylb = ylabel('PDF','interpreter','Latex');
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set([xlb,ylb],'interpreter','Latex','fontsize',18);
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
saveas(gcf, 'PDF-Lapsing_distance.png')
