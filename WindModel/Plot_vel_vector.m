function[px,py,pvx,pvy] = PlotVel()
clear;

mainfolder = pwd;
addpath(mainfolder);
Framerange = 6:1:215;


fig = figure();

for jj = 1:length(Framerange)
    
    FrameNum = Framerange(jj);
    
    [pvx,pvy,px,py,~]=vp(fullfile(mainfolder,'OneDrive_1_8-7-2024'),FrameNum,[],0); % in pixel per frame
    
    quiver(px,py,pvx*5,pvy*5,'off','k-','LineWidth',0.25);

    axis([0 1600 0 1600]);

    set(gcf,'MenuBar','figure',...
        'Units','centimeters',...
        'Position',[15,1,25,25],...
        'Resize',0);
    ax = gca;
    set(gca,'FontName','Arial',...
        'YDir','reverse',...
        'Box','on',...
        'Units','centimeters',...
        'looseInset',[0,0,0,0]...
        );
    ax.PlotBoxAspectRatio = [1,1,1];

    pause(0.7);
end

% lenX = length(px)
% lenY = length(py)
% lenVX = length(pvx)
% lenVY = length(pvy)
% g = zeros(4,lenX)
% 
% target = [5;
%           0]
% k = 1
% % for x = 1:lenX
% %     g(1:4,x) = [px(x);
% %           py(x);
% %           pvx(x);
% %           pvy(x)]
% %     if abs(target(1) - g(1,x)) <= 3 & abs(target(2) - g(2,x)) <= 3
% %         j(k) = x
% %         k = k + 1
% %     else
% %         k = k
% %     end
% % end
% g = 100:1:200
% [~,idx] = unique(px);
% v = interp1(px(idx),pvx(idx),g)
% h = max(pvx)
end