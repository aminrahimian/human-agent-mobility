%% plot pdf of time
load("randwalk_1k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin,'linewidth',1.5);
hold on

load("randwalk_2k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin,'linewidth',1.5);

load("randwalk_3k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin,'linewidth',1.5);

load("randwalk_4k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin,'linewidth',1.5);

axis([0,60,0,0.3]);

xlb = xlabel('T (hour)');
ylb = ylabel('pdf');
ttl = title('Random Walk');
lgd = legend('$1k \times 1k$','$2k \times 2k$','$3k \times 3k$','$4k \times 4k$',...
    'location','northeast');

legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,500,500]);
set(gca,'looseInset',[0 0 0 0]);
%%

for i = 1:4
    eval(['load(''randwalk_',num2str(i),'k_60.mat'')']);
    T_mean(i) = mean(TimeLength_record);
end

save('randwalk_meantime.mat','T_mean');

%%
load("randwalk_2k_60.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin,'linewidth',1.5);
hold on
load("randwalk_2k_60_Env.mat");
[numbin,edges] = histcounts(TimeLength_record,0:3:72,'normalization','pdf');
bincenter = 0.5*(edges(1:end-1) + edges(2:end));
plot(bincenter,numbin,'linewidth',1.5);

axis([0,60,0,0.15]);

xlb = xlabel('T (hour)');
ylb = ylabel('pdf');
ttl = title('Random Walk');
lgd = legend('$2k \times 2k$','$2k \times 2k$, with barrier',...
    'location','northeast');

legend boxoff
set(gca,'TickLabelInterpreter','latex','fontsize',10);
set([xlb,ylb,ttl,lgd],'interpreter','Latex','fontsize',12);

box on
set(gcf,'units','pixels','innerposition',[200,200,500,500]);
set(gca,'looseInset',[0 0 0 0]);