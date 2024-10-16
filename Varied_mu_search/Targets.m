% generate target distribution as a pattern of hierarchical Gaussian
% distribution

% Four parameters control the distribution
% num_patches: number of patches
% num_tar: number of targets within each patch
% std_dev: standard deviation of the normal distribution of patches;
% std_tar: standard deviation of the normal distribution of targets in each patch;

clear;clc;

mean = 5000;
std_dev = 3000;
num_patches = 10;

x_patch = mean + std_dev * randn(num_patches, 1);
y_patch = mean + std_dev * randn(num_patches, 1);

scatter(x_patch, y_patch)

std_tar = 1500;
num_tar = 50;
tar_x_rec = [];
tar_y_rec = []; % record target distribution
for i = 1:num_patches
    x_tar = x_patch(i) + std_tar * randn(num_tar, 1);
    y_tar = y_patch(i) + std_tar * randn(num_tar, 1);
    scatter(x_tar, y_tar)
    hold on
    tar_x_rec = [tar_x_rec; x_tar];
    tar_y_rec = [tar_y_rec; y_tar];
end

data = [tar_x_rec, tar_y_rec];
filename = 'target_3.csv';  % Name of the CSV file
csvwrite(filename, data); % save as .csv file
save('TarDist3.mat','tar_x_rec','tar_y_rec'); % save as .mat file
%% plot target distribution
load("TarDistLarge.mat")
scatter(tar_x_rec, tar_y_rec)
xlb = xlabel('$x$ (m)','interpreter','Latex');
ylb = ylabel('$y$ (m)','interpreter','Latex');
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set([xlb,ylb],'interpreter','Latex','fontsize',18);
saveas(gcf,'TargetDistribution.png')