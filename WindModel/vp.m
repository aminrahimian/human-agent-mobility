function [pvx,pvy,px,py,pt]=vp(filename,framerange,region,noisy)
% Usage: [pvx,pvy,px,py,pt]=vp(filename,[framerange],[region],[noisy])
% Working from the velocity tracks recorded in "filename", vp plots the
% velocity at each particle in frames "framerange" in the area "region".
% Velocities are returned in "pvx" and "pvy"; positions, "px" and "py";
% times, "pt". If noisy==0, no plot is produced. Specify "region" in
% pixels, as a vector of the form [left bottom width height]. The input can
% either be a .gdf file (sorted framewise with x, y, t, vx, and vy in
% columns 2-6, respectively) or a directory containing .mat files created
% by streamprojbN. Requires getmaxlen.m for plotting. For improved speed
% with .gdf files, first run index_gdf.m. See also vframe.m, aframe.m,
% vrms.m, divframe.m, vortframe.m.

% Written by Doug Kelley 28 July 2009. 
% Updated 31 Aug 2009 to handle underscores in filenames.
% Updated 14 October 2009 to work with streamprojN output and to respect
% "region" not only in plotting but also in data.
% Updated 18 January 2010 to exclude critical points.
% Added "bigfile" option 25 January 2010. 
% Added scalebar 29 January 2010.
% Changed "bigfile" to "readwhole" 24 February 2010. 
% Added output "pt" 10 March 2010.
% Corrected scale arrow placement when region is not default, 4 August
% 2010.
% Added compatibility with index_gdf (for speed!) and removed "readwhole"
% option 4 January 2011.
% Made default region fit all data, 22 June 2011. 
% Index file must be in same directory as .gdf, as of 5 December 2011. 
% Now pre-allocating for speed when reading a directory of .mat files, as
% of 17 July 2012.

% Note by Xinyu
% the folder that stores the filename should be added to the path

magicnum = 82991;
indexsuffix = '_ind'; % for naming files output by index_gdf.m 
framerangedefault = [0 inf]; % all frames
noisydefault = true;
scaleinset = .01; % fraction of plot width, for scalebar
scalesize = .02; % fraction of plot width, for scalebar

% Format of input data:
keepcols = [2 3 5 6];
tcolinit = 4;

% Data format after discarding unused cols:
xcol = 1;
ycol = 2;
vxcol = 3;
vycol = 4;

% Parse input arguments
if nargin < 1
    error(['Usage: [pvx,pvy,px,py,pt] = ' mfilename ...
        '(filename,[framerange],[region],[noisy])']);
end
if ~exist('framerange','var') || isempty(framerange)
    framerange=framerangedefault;
elseif numel(framerange)==1
    framerange=framerange*[1 1];
end
if ~exist('noisy','var') || isempty(noisy)
    noisy=noisydefault;
end

if ~exist(filename,'file')
    error(['Sorry, cannot find ' filename '.']);
elseif exist(filename,'file')==2 % a single file; expect .gdf
    
    [path,shortname]=fileparts(which(filename));
    indfile=fullfile(path,[shortname indexsuffix '.mat']); % specify full path to stay in same directory
    if exist(indfile,'file') % with an index file, can do this the fast way.
        frameind=load(indfile);
        tmin=max([framerange(1) frameind.n(1)]);
        tmax=min([framerange(2) frameind.n(end)]);
        if tmin>tmax
            error(['Sorry, ' filename ...
                ' does not contain frames between ' ...
                num2str(framerange(1)) ' and ' ...
                num2str(framerange(2)) '.']);
        end
        fidin=fopen(filename);
        header=fread(fidin,6,'int32');
        if header(1)~=magicnum
            error(['Sorry, ' filename ...
                ' does not appear to be a .gdf file.']);
        end
        ncol=header(3);
        startloc=frameind.nstarts(frameind.n==tmin);
        readsize=frameind.nstarts(frameind.n==tmax) + ...
            frameind.nsizes(frameind.n==tmax) - ...
            frameind.nstarts(frameind.n==tmin); % in bytes
        fseek(fidin,startloc,'bof');
        bt=fread(fidin,readsize/4,'float32'); % 4 bytes per float32
        bt=reshape(bt,ncol,numel(bt)/ncol)';
        fclose(fidin);
        pt=bt(:,tcolinit);
        bt=bt(:,keepcols);
        px=bt(:,xcol);
        py=bt(:,ycol);
        pvx=bt(:,vxcol);
        pvy=bt(:,vycol);

    else % Open the .gdf input file, read the header, get initial time
        fidin=fopen(filename);
        header=fread(fidin,6,'int32');
        if header(1)~=magicnum
            error(['Sorry, ' filename ...
                ' does not appear to be a .gdf file.']);
        end
        ncol=header(3);
        buffer=fread(fidin,ncol,'float32')';
        tmin=max(buffer(tcolinit),framerange(1));
        px=[];
        py=[];
        pvx=[];
        pvy=[];
        pt=[];
        % Loop over frames, keeping all particles in each
        while ~feof(fidin)

            % Read in the next frame
            t=buffer(tcolinit);
            fseek(fidin,-4*ncol,'cof'); % back up one sample
            bt=[];
            while buffer(tcolinit)==t
                buffer=fread(fidin,ncol,'float32')';
                bt=[bt;buffer];
                if feof(fidin)
                    break
                end
            end
            if ~feof(fidin)
                bt(end,:)=[]; % remove last sample, which has wrong time
            end    
            if t < framerange(1) % before firstframe
                continue
            end
            if t > framerange(2) % after lastframe
                break
            end
            bt=bt(:,keepcols); % drop track index, time
            px=[px;bt(:,xcol)];
            py=[py;bt(:,ycol)];
            pvx=[pvx;bt(:,vxcol)];
            pvy=[pvy;bt(:,vycol)];
            pt=[pt;t*ones(size(bt,1),1)];
        end % loop over frames
        tmax=t-1;
        fclose(fidin);
        if numel(px)==0
            error(['Sorry, ' filename ...
                ' does not contain frames between ' ...
                num2str(framerange(1)) ' and ' ...
                num2str(framerange(2)) '.']);
        end
    end % if exist(indfile,'file')
    
else % a directory; expect .mat files output from streamprojN
    
    list=dir([filename '/*.mat']);
    tlist=nan*ones(size(list));
    for ii=1:numel(tlist)
        tlist(ii)=str2double(strrep(list(ii).name,'.mat','')); % convert filenames to times
    end
    tlist(isnan(tlist)) = []; % remove NaN's due to other files
    tmin=max(framerange(1),tlist(1));
    tmax=min(framerange(2),tlist(end));
    if tmin>tmax
        error(['Sorry, ' filename ' does not contain frames between ' ...
            num2str(framerange(1)) ' and ' num2str(framerange(2)) '.']);
    end
    Nt=numel(tmin:tmax);
    for t=tmin:tmax % loop over times
        ind=tlist==t;
        varlist=open([filename '/' list(ind).name]);
        d=load([filename '/' list(ind).name],'p','pvx','pvy'); % grab data
        if any(strcmp(fieldnames(varlist),'hyp')) % contains triangulation, so load it
            hypell=load([filename '/' list(ind).name],'hyp','ell'); % grab data
            Ncrit=size(hypell.hyp,1)+size(hypell.ell,1); % count of crit points
            d.p(:,end-Ncrit+1:end)=[]; % exclude crit pts
            d.pvx(end-Ncrit+1:end)=[];
            d.pvy(end-Ncrit+1:end)=[];
        end
        Np=size(d.p,2);
        if t==tmin % first frame? pre-allocate for speed.
            px=NaN(Nt*Np,1);
            py=NaN(Nt*Np,1);
            pvx=NaN(Nt*Np,1);
            pvy=NaN(Nt*Np,1);
            pt=NaN(Nt*Np,1);
            memloc=1;
        end
        px(memloc:memloc+Np-1)=d.p(1,:)';
        py(memloc:memloc+Np-1)=d.p(2,:)';
        pvx(memloc:memloc+Np-1)=d.pvx;
        pvy(memloc:memloc+Np-1)=d.pvy;
        pt(memloc:memloc+Np-1)=t;
        memloc=memloc+Np;
    end % for t=tmin:tmax
    px(memloc:end)=[];
    py(memloc:end)=[];
    pvx(memloc:end)=[];
    pvy(memloc:end)=[];
    pt(memloc:end)=[];
end % if ~exist(filename,'file')

if exist('region','var') && ~isempty(region)
    keepers=( (px>=region(1)) & (px<=region(1)+region(3)) & ...
        (py>=region(2)) & (py<=region(2)+region(4)) ); % exclude points near edges
    px=px(keepers);
    py=py(keepers);
    pvx=pvx(keepers);
    pvy=pvy(keepers);
    pt=pt(keepers);
end

% If requested, plot the particles
if noisy
    figure;
    maxlen=getmaxlen(px,py,pvx,pvy);
    quiver(px,py,pvx/maxlen,pvy/maxlen,0);
    set(gca,'dataaspectratio',[1 1 1], ...
        'nextplot','add');
    axis tight
    myscale=round(scalesize*diff(xlim)*maxlen);
    myinset=scaleinset*diff(xlim);
    hq=quiver(min(xlim)+myinset,min(ylim)+myinset,myscale/maxlen, ...
        0,0,'k','linewidth',3);
    text(min(xlim)+myinset,min(ylim)+myinset,num2str(myscale),...
        'verticalalignment','bottom','backgroundcolor','w','fontsize',11)
    uistack(hq,'top');
    if framerange(1)==framerange(2)
        title([num2str(numel(px)) ' particles in ' ...
            strrep(filename,'_','\_') ': frame ' num2str(framerange(1))]);
    else
        title([num2str(numel(px)) ' particles in ' ...
            strrep(filename,'_','\_') ': frames ' ...
            num2str(tmin) ' to ' num2str(tmax) ]);
    end
end
