L = 1:10000;

R = 100;

v = 2.8;

T = (L + L.^2/2/R)/v;

plot(L,T/3600,'linewidth',1.5);

xlb = xlabel('L (m)');
ylb = ylabel('T (hour)');
ttl = title('Through Search');

set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,500,500]);
set(gca,'looseInset',[0 0 0 0]);
%%
% through search
clear;
L = 0:1000:5000;
R = 100;
v = 2.8;
T = (L + L.^2/2/R)/v/2;
plot(L,T/3600,'linewidth',1.5);
hold on

% random walk
load('../RandomWalk/randwalk_meantime.mat')
plot(0:1000:4000,[0,T_mean],'linewidth',1.5);

% bayesian
load('../Bayesian/bayesian_meantime.mat')
plot(0:1000:2000,[0,T_mean],'linewidth',1.5);

% minimum time
for kk = 1:5
    T_minn(kk) = mean(sqrt((rand(5000)*kk*1000).^2 + (rand(5000)*kk*1000).^2),'all')/v/3600;
end
plot(0:1000:5000,[0,T_minn],'linewidth',1.5);

fill([0:1000:5000,5000:-1000:0],[0,T_minn,flip(T/3600)],'c','FaceAlpha',0.4);

% formatting
xlb = xlabel('L (m)');
ylb = ylabel('T (hour)');
ttl = title('Baseline comparison');

lgd = legend('through search', 'random walk','Bayesian','minimum time','improvable regime',...
    'location','northwest');

legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,500,500]);
set(gca,'looseInset',[0 0 0 0]);