% Optimized varied mu search?
% determine the mu based on known target distribution
load("TarDistLarge.mat")
R = 200; % R determines the resolution of grid
Dim = 10000;
[X,Y] = meshgrid(0:R:Dim,0:R:Dim);
% the agent goes from the center of the domain [5000,5000]
maxT = 50000;
xy = zeros(maxT,2);
xy(1,:) = [5000,5000];
mu_grid = zeros(size(X,1), size(X,2));
mu_grid(:,:) = 2;
% determine at which grid cell the targets located
for i = 1:numel(tar_x_rec)
    [row_x, col_x]= find(X <= tar_x_rec(i) & tar_x_rec(i) <= X + R); 
    [row_y, col_y] = find(Y <= tar_y_rec(i) & tar_y_rec(i) <= Y + R);
    if numel(row_y) == 0 || numel(col_x) ==0
        continue
    end
    mu_grid(row_y(1), col_x(1)) = 1;
end

% plot the distribution of mu according to the positions of targets
figure(1)
h = pcolor(X,Y,mu_grid);
colormap('gray')
set(h, 'EdgeColor', 'none');
xlb = xlabel('$x$ (m)','interpreter','Latex');
ylb = ylabel('$y$ (m)','interpreter','Latex');
set(gca,'TickLabelInterpreter','latex','fontsize',14);
set([xlb,ylb],'interpreter','Latex','fontsize',18);
saveas(gcf, 'mu-distribution.png')

% colormap(flipud(colormap));

%% search (Parallel)
NumLevy = 50; % number of cases with different trajectories
mu1 = 1; mu2 = 2; 
maxL = 500; maxT = 50000; R = 25;
num_det_tar = zeros(NumLevy,1);
Last_tar_index = zeros(NumLevy,1);
board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0]; %boundary

parfor k = 1:NumLevy 
    tic
    xy = zeros(maxT,2);
    xy(1,:) = [5000,5000];
    Stepfile = load(strcat('StepLength-',num2str(k),'.mat'))
    step_mu11 = Stepfile.step_mu1;
    step_mu22 = Stepfile.step_mu2;
    direc = Stepfile.direc;
    
    next_step = step_mu11(1);
    Target = [tar_x_rec, tar_y_rec];
    LastTar = 0;
    for j =1:maxT-1
        [row_x, col_x]= find(X <= xy(j,1) & xy(j,1) <= X + R);
        [row_y, col_y] = find(Y <= xy(j,2) & xy(j,2) <= Y + R);
        mu = mu_grid(row_y(1), col_x(1));
        u = cos(direc(j)); %component of the direction vector
        v = sin(direc(j));
        if mu == 1
            next_step = step_mu22(j);
        else
            next_step = step_mu11(j);
        end
        xy(j+1,:) = xy(j,:) + [u*next_step, v*next_step];
        locx = xy(j+1,1); locy = xy(j+1,2); xyloc = [locx, locy];
        if ~inpolygon(xyloc(1),xyloc(2),board(:,1),board(:,2))
            xyloc(xyloc < 0) = abs(xyloc(xyloc < 0));
            if xyloc(1) > Dim
                xyloc(1) = 2*Dim - xyloc(1);
            end
            if xyloc(2) > Dim
                xyloc(2) = 2*Dim - xyloc(2);
            end
        end
        xy(j+1,:) = xyloc;
        dist = pdist2(xy(j+1,:), Target) - R;
        % ====== varies mu with long memorization ======
        if min(dist) <= 0
            Target(find(dist<0),:) = [];
            LastTar = j+1;
        end
        
    end
    num_det_tar(k) = numel(tar_x_rec)- size(Target,1);
    Last_tar_index(k) = LastTar;

%     figure(2);
%     hold on
%     plot(tar_x_rec,tar_y_rec,'rp','markersize',10,'MarkerFaceColor','r');
%     plot(xy(1:maxT,1),xy(1:maxT,2),'b-','LineWidth',1.5);
%     plot(xy(1,1), xy(1,2),'cs','markersize',14,'MarkerFaceColor','c');
%     plot(xy(maxT,1),xy(maxT,2),'mo','markersize',10,'MarkerFaceColor','b');
%     xlb = xlabel('$x$ (m)','interpreter','Latex');
%     ylb = ylabel('$y$ (m)','interpreter','Latex');
%     
%     lgd = legend('Targets','Trajectory','Initial location','Last location',...
%      'location','northwest');
%     % legend boxoff
%     set(gca,'TickLabelInterpreter','latex','fontsize',14);
%     set([xlb,ylb],'interpreter','Latex','fontsize',18);
%     set(lgd,'interpreter','Latex','fontsize',12);
%     saveas(gcf, strcat('Number k = ', num2str(k),'-trajectories.png'))
%     close(gcf)
    toc
end
save('KnownTargetSearchEffi.mat','num_det_tar', 'Last_tar_index');