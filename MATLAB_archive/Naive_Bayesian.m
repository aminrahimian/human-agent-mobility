clear;
clc;

addpath('SubFunc');

% rng(2);

%Create a grid of evenly spaced points in two-dimensional space.
Dim = 2000;
x1 = 0:(Dim/500):Dim;
x2 = 0:(Dim/500):Dim;
[X1,X2] = meshgrid(x1,x2);
XX = [X1(:) X2(:)];

% Evaluate the pdf of the normal distribution at the grid points.
% Single-peak
mu = [Dim Dim]/2;
Sigma = [Dim 0; 0 Dim] * 20;
YY = mvnpdf(XX,[Dim*0.5, Dim*0.5],Sigma);
YY = YY/sum(YY);
YY = reshape(YY,length(x2),length(x1));

% Multi-peak
% Sigma = [Dim 0; 0 Dim] * 20;
% YY1 = mvnpdf(XX,[Dim*0.5, Dim*0.5],Sigma);
% YY1 = YY1/sum(YY1);
% YY1 = reshape(YY1,length(x2),length(x1));
% 
% YY2 = mvnpdf(XX,[Dim*0.2, Dim*0.8],Sigma);
% YY2 = YY2/sum(YY2);
% YY2 = reshape(YY2,length(x2),length(x1));
% 
% YY3 = mvnpdf(XX,[Dim*0.8, Dim*0.2],Sigma);
% YY3 = YY3/sum(YY3);
% YY3 = reshape(YY3,length(x2),length(x1));
% 
% YY4 = mvnpdf(XX,[Dim*0.8, Dim*0.8],Sigma);
% YY4 = YY4/sum(YY4);
% YY4 = reshape(YY4,length(x2),length(x1));
% 
% YY5 = mvnpdf(XX,[Dim*0.2, Dim*0.2],Sigma);
% YY5 = YY5/sum(YY5);
% YY5 = reshape(YY5,length(x2),length(x1));
% 
% YY = (YY1 + YY2 + YY3 + YY4 + YY5)/5;

% pre-determination
tau = 0.5;
dt = tau;
V_best = 2.8;
dT = 60;
R = 100;

% determine whether add environmental condition
ENV = 0; % if 1, heterogeneous environmental condition is used
if ENV == 1
    load('roi.mat');
    inroi = inROI(roi1,X1(:),X2(:)) | inROI(roi2,X1(:),X2(:));
    inroi = reshape(inroi,length(x1),length(x2));
else
    inroi = false(length(x1),length(x2));
end

scale_factor = ones(length(x1),length(x2));
scale_factor(inroi) = 0.3;


NN = 500; % number of realizations
TimeLength_record = zeros(NN,1); % record the leng of time needed

parfor nn = 1:NN % loop for each realization
    disp(nn);
    
    Real_Target = rand(1,2)*Dim;
    Belief_new = YY;
   
    % Define velocity and location vectors (each row a point)
    xy_pede = [0,0];

    v_desire = V_best * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
    vel_pede = V_best * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
    theta_pede = rand(1)*2*pi;
    u_pede = vel_pede.*cos(theta_pede); % u, v component of pedestrian
    v_pede = vel_pede.*sin(theta_pede);

    uv_pede = [u_pede, v_pede];
    xy_pede_save = xy_pede;
    t = 1;

%     fig = figure();

    while norm(Real_Target - xy_pede) > R * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1)
        disp(t);
        
        %  use the max belief as fake target location
        %     max_idx = find(Belief_new == max(Belief_new(:)),1,'first');
        %     max_loc = [X1(max_idx),X2(max_idx)];
        %     % Define the location of the target using random samping
        %     r = rand(1)* 50;
        %     theta = rand(1) * 2*pi;
        %     r_target = [max_loc(1) + r*cos(theta), max_loc(2) + r*sin(theta)];
        
        % randomly sample the fake target according to the belief map
        sample_loc = datasample(1:length(X1(:)),1,'weight',Belief_new(:));
        r_target_x = mean(X1(sample_loc));
        r_target_y = mean(X2(sample_loc));
        r_target = [r_target_x, r_target_y];
        
        % update the movement of the agent
        [F_Di]=calc_F_Di(xy_pede,uv_pede,r_target,v_desire,tau);
        uv_pede_old = uv_pede; % save the previous step velocity
        uv_pede = uv_pede + F_Di*dt; % update velocity
        xy_pede = xy_pede + uv_pede*dT;
        xy_pede_save(t+1,:) = xy_pede;
        v_desire = V_best * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
        
        % update belief map
        Belief_old = Belief_new;
        radius_factor = 1 * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1); % apply a radius factor to weaken the exponential decay
        Likely = 1 - exp(-1*sqrt(...
            (X1 - xy_pede(1)).^2 + (X2 - xy_pede(2)).^2)./(radius_factor * R));
        NormalizationTerm = sum(Belief_old.*Likely,'all');        
        Belief_new = Belief_old.*Likely/NormalizationTerm;

        % plot gif
%         imagesc(x1,x2,Belief_new);
%         colormap('summer');
%         colorbar;
%         set(gca,'Ydir','normal');
%         hold on
%         plot(Real_Target(1),Real_Target(2),'rp','markersize',10,'MarkerFaceColor','r');
%         plot(0,0,'gs','markersize',10,'MarkerFaceColor','g');
%         plot(xy_pede_save(:,1),xy_pede_save(:,2),'m-','LineWidth',1.5);
%         plot(xy_pede_save(end,1),xy_pede_save(end,2),'bo','markersize',10,'MarkerFaceColor','b');
%         if ENV
%             load('roi.mat');
%             x1 = roi1.Position;
%             x2 = roi2.Position;
%             fill(x1(:,1),x1(:,2),'b', 'FaceAlpha', 0.3,'edgecolor','none');
%             fill(x2(:,1),x2(:,2),'b', 'FaceAlpha', 0.3,'edgecolor','none');
%         end
% 
%         axis([0 Dim 0 Dim]);
%         % set(gca,'Ydir','reverse');
%         xlb = xlabel('m');
%         ylb = ylabel('m');
%         ttl = title(['Bayesian (',sprintf('%.2f',(t-1)*dT/3600),'hour)']);
% 
%         if ENV
%             lgd = legend('real target','agent initial location','agent path','agent current position','environmental condition',...
%                 'location','northeast');
%         else
%             lgd = legend('real target','agent initial location','agent path','agent current position',...
%                 'location','northeast');
%         end
%         % legend boxoff
%         set(gca,'TickLabelInterpreter','latex','fontsize',10);
%         set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);
% 
%         box on
%         set(gcf,'units','pixels','innerposition',[200,200,600,500]);
%         set(gca,'looseInset',[0 0 0 0]);
% 
%         drawnow;
%         frame = getframe(fig);
%         im = frame2im(frame);
%         [A,map] = rgb2ind(im,256);
%         if t == 1
%             imwrite(A,map,['test.gif'],'gif','LoopCount',Inf,'DelayTime',0.1);
%         else
%             imwrite(A,map,['test.gif'],'gif','WriteMode','append','DelayTime',0.1);
%         end
            
        t = t+1;
        
        % judge if pass the target during each time step
        if t > 2
            v1 = xy_pede_save(t,:); % the first vertex
            v2 = xy_pede_save(t-1,:); % the second vertex
            vec1 = v2 - v1;
            vec2 = Real_Target - v1;
            if dot(vec1,vec2)>=0 && dot(vec1,vec2)<=dot(vec1,vec1)
                d = abs( det([Real_Target-v1;v2-v1]) )/norm(v2-v1); % distanct from the real target to v1-v2
                if d <= R
                    break
                end
            end
        end

        % break over 10 hours
        if t*dT/3600> 72
            break
        end
    end
    
    TimeLength_record(nn) = (t-1)*dT/3600;
    disp(['using ',num2str((t-1)*dT/3600),' hours']);
end
%%
figure(2);
imagesc(x1,x2,Belief_new);
colormap('summer');
colorbar;
set(gca,'Ydir','normal');
hold on
plot(Real_Target(1),Real_Target(2),'rp','markersize',10,'MarkerFaceColor','r');
plot(0,0,'gs','markersize',10,'MarkerFaceColor','g');
plot(xy_pede_save(:,1),xy_pede_save(:,2),'m-','LineWidth',1.5);
plot(xy_pede_save(end,1),xy_pede_save(end,2),'bo','markersize',10,'MarkerFaceColor','b');
if ENV
    load('roi.mat');
    x1 = roi1.Position;
    x2 = roi2.Position;
    fill(x1(:,1),x1(:,2),'b', 'FaceAlpha', 0.3,'edgecolor','none');
    fill(x2(:,1),x2(:,2),'b', 'FaceAlpha', 0.3,'edgecolor','none');
end

axis([0 Dim 0 Dim]);
% set(gca,'Ydir','reverse');
xlb = xlabel('m');
ylb = ylabel('m');
ttl = title('Bayesian');

if ENV
    lgd = legend('real target','agent initial location','agent path','agent current position','environmental condition',...
        'location','northeast');
else
    lgd = legend('real target','agent initial location','agent path','agent current position',...
        'location','northeast');
end
% legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,600,500]);
set(gca,'looseInset',[0 0 0 0]);
%%
imagesc(YY);
%%
load("bayesian_1k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin);
hold on
load("bayesian_2k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin);

xlabel('T (hour)');
ylabel('pdf');
title('Bayesian');

legend('1k*1k','2k*2k');
%%
[numbin,edges] = histcounts(TimeLength_record,0:1:24,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));

plot(bincenter,numbin);
%%
save('bayesian_2k_60.mat','TimeLength_record');
%%

for i = 1:2
    eval(['load(''bayesian_',num2str(i),'k_60.mat'')']);
    T_mean(i) = mean(TimeLength_record);
end

save('bayesian_meantime.mat','T_mean');