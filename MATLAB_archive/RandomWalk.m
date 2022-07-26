clear;
clc;

addpath('..\SubFunc');

% rng(1);

%Create a grid of evenly spaced points in two-dimensional space.
Dim = 2000;
x1 = 0:(Dim/500):Dim;
x2 = 0:(Dim/500):Dim;
[X1,X2] = meshgrid(x1,x2);
XX = [X1(:) X2(:)];

board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0];

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


NN = 1; % number of realizations
TimeLength_record = zeros(NN,1); % record the leng of time needed

for nn = 1:NN % loop for each realization
    disp(nn);
    
    Real_Target = rand(1,2)*Dim;
   
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
    while norm(Real_Target - xy_pede) > R * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1)
%         disp(t);
        
        % randomly sample the fake target according to the belief map
        theta = rand(1)*2*pi;
        r_target_x = xy_pede(1) + 100*cos(theta);
        r_target_y = xy_pede(2) + 100*sin(theta);
        r_target = [r_target_x, r_target_y];
        
        % update the movement of the agent
        [F_Di]=calc_F_Di(xy_pede,uv_pede,r_target,v_desire,tau);
        uv_pede_old = uv_pede; % save the previous step velocity
        uv_pede = uv_pede + F_Di*dt; % update velocity
        xy_pede = xy_pede + uv_pede*dT;

        if ~inpolygon(xy_pede(1),xy_pede(2),board(:,1),board(:,2))
            xy_pede(xy_pede < 0) = abs(xy_pede(xy_pede < 0));
            xy_pede(xy_pede > Dim) = 2*Dim - xy_pede(xy_pede > Dim);
        end

        xy_pede_save(t+1,:) = xy_pede;
        v_desire = V_best * interp2(X1,X2,scale_factor,xy_pede(1),xy_pede(2),'nearest',1);
                
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

        % break over ** hours
        if t*dT/3600> 72
            break
        end
    end
    
    TimeLength_record(nn) = (t-1)*dT/3600;
    disp(['using ',num2str((t-1)*dT/3600),' hours']);
end
%% plot a single trajectory
figure(2);
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
ttl = title('Random Walk');
if ENV
    lgd = legend('real target','agent initial location','agent path','agent current position','environmental condition',...
        'location','northeast');
else
    lgd = legend('real target','agent initial location','agent path','agent current position',...
        'location','northwest');
end
% legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,500,500]);
set(gca,'looseInset',[0 0 0 0]);
%% save result
[numbin,edges] = histcounts(TimeLength_record,0:2:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));

plot(bincenter,numbin);
%%
save('randwalk_2k_60_Env.mat','TimeLength_record'); 
%%
kk = [1,2,4,6];
T_mean = zeros(1,length(kk));
for i = 1:length(kk)
    eval(['load(randwalk_',num2str(kk(i)),'k_60)']);
    T_mean(i) = mean(TimeLength_record);
end
