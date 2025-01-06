%%Construct Q-Table
[px,py,pvx,pvy]=Plot_vel_vector();
%[pxx,pyy,pvxx,pvyy] = PlotVel();
% Define grid size
MPgrid_size_x = 1600;
LPgrid_size_x = 0;
MPgrid_size_y = 1600;
LPgrid_size_y = 0;
MPgrid_sixe_xv = 70; %km/h
LPgrid_size_xv = -70; %km/h
MPgrid_size_yv = 70; %km/h
LPgrid_size_yv = -70;%km/hr
MPgrid_size_DW = 360; %degrees
LPgrid_size_DW = -360; %degrees
% Define number of buckets along each axis
num_buckets_px = 10;
num_buckets_py = 10;
num_buckets_pvx = 10;
num_buckets_pvy = 10;
num_buckets_DW = 10;
n_buckets = [num_buckets_px, num_buckets_py,num_buckets_pvx,num_buckets_pvy,num_buckets_DW,9];
% Calculate bucket size
bucket_size_px = MPgrid_size_x / num_buckets_px;
bucket_size_py = MPgrid_size_y / num_buckets_py;
bucket_size_pvx = MPgrid_sixe_xv - LPgrid_size_xv / num_buckets_pvx;
bucket_size_pvy = MPgrid_size_yv - LPgrid_size_yv / num_buckets_pvy;
bucket_size_DW = MPgrid_size_DW - LPgrid_size_DW / num_buckets_DW;
% % Example: Mapping a grid point (px, py) to a bucket
% px = 12345; % Example x-coordinate
% py = 5678;  % Example y-coordinate
% 
% % Find bucket indices
% bucket_x = floor(px / bucket_size_x) + 1; % +1 for 1-based indexing
% bucket_y = floor(py / bucket_size_y) + 1;
% 
% % Ensure the indices are within the bounds
% bucket_x = min(max(bucket_x, 1), num_buckets_x);
% bucket_y = min(max(bucket_y, 1), num_buckets_y);
gu = rand(2,1);
Q = rand(10,10,10,10,10,9)*2 -1; %Dposx,Dposy,Dvelx,Dvely,DirW,Action
%Q(10,10,10,10,10,:) = round(9*rand(9,1))
%%Q-Learning Parameters
lr = 0.5;
discount = 0.9;
episodes = 30000;
%%%Epsilon-Greedy Strategy
epsilon = 1;
eps_decay_begin = 1;
eps_decay_end = 10000;
eps_decay_val = epsilon/(eps_decay_end-eps_decay_begin);
%Target Location
Tpos = [1430,1200];
tot_reward = zeros([1,200]);
g = []
for x = 1:episodes
    count_step = 0;
    currREW = 0
    t = x
    
    %Variable for Target being reached
    
    %Initialize Position and Velocity of Drone
    Dpos = [400;
            100];
    DVel = [1;
            0];
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py);
    % g_endX = Dpos(1) + 10
    % g_endY = Dpos(2) + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    wvx = interp1(px(idx),pvx(idx),Dpos(1)); %interpolated wind-x velocities
    awvx = wvx;
    wvy = interp1(py(idy),pvy(idy),Dpos(2)); %interpolated wind y-velocities
    awvy = wvy;
    WDir = atan2(awvy,awvx); %Wind Direction
    WDir = rad2deg(WDir);
    AC = round(9*rand(1));
    state = [Dpos(1), Dpos(2), DVel(1), DVel(2), WDir, AC];
    bucketSZ = [bucket_size_px, bucket_size_py, bucket_size_pvx, bucket_size_pvy, bucket_size_DW, 8/9];
    low = [LPgrid_size_x, LPgrid_size_y, LPgrid_size_xv, LPgrid_size_yv, LPgrid_size_DW, 1];
    xx = []
    yy = []
    I = []
    T = []
    I(1) = Dpos(1)
    T(1) = Dpos(2)
    xx(1) = Dpos(1)
    yy(1) = Dpos(2)
    %Loop over each dimension
    %Could be made into a function
    for i = 1:6
        % Calculate the bucket index
        bucket_index(i) = floor((state(i) - low(i)) / bucketSZ(i)) + 1;
        % Ensure the index is within valid range
        if bucket_index(i) <= 0;
            bucket_index(i) = 1;
        elseif bucket_index(i) >= n_buckets(i);
            bucket_index(i) = n_buckets(i) - 1;
        end
    end
    SX = 0;
    time = 0;
    while (SX ~= 1 && count_step<200)
        count_step = count_step + 1
        WinD = [awvx,awvy]
        %Exploration Phase
        if rand()>epsilon
            [~,action] = max(Q(bucket_index(1),bucket_index(2),bucket_index(3),bucket_index(4), bucket_index(5),:));
        else
            action = randi([1,9],1);
        end
        time = time + 1
        ggg = time
       
        [xi,y,vx,vy,WinD,action,angle_deg,awvx,awvy] = Act(action,time,DVel(1),DVel(2),WinD,Dpos(1),Dpos(2),px,py,pvx,pvy)
        
        [rew,SX] = Reward(xi,y,Tpos,WinD,vx,vy,Dpos(1),Dpos(2))
        
        currREW = currREW + rew
        
        NState = [xi,y,vx,vy,WDir,action]
        
        nbucket_index = bucket(NState,bucketSZ,low,n_buckets)
        if SX ~= 1
            %Q Learning
            max_fut_q = max(Q(nbucket_index(1),nbucket_index(2),nbucket_index(3),nbucket_index(4),nbucket_index(5),:));
            curr_q = Q(bucket_index(1),bucket_index(2),bucket_index(3),bucket_index(4),bucket_index(5),bucket_index(6));
            new_q = (1-lr)*curr_q +lr*(rew+discount*max_fut_q);
            Q(bucket_index(1),bucket_index(2),bucket_index(3),bucket_index(4),bucket_index(5),bucket_index(6)) = new_q;
        else
            Q(bucket_index(1),bucket_index(2),bucket_index(3),bucket_index(4),bucket_index(5),bucket_index(6)) = 1
        end
        bucket_index = nbucket_index
        DVel = [vx;
                vy]
        Dpos = [xi;
                y]
       
        if mod(t,100) == 0
            xx(ggg + 1) = xi
            yy(ggg + 1) = y
        end
        
%         I(ggg + 1) = xi
%         T(ggg + 1) = y
    
    end
    tot_reward(t)=currREW;
    g(t) = SX
    % Decay exploration with 
    if (t<eps_decay_end && t>eps_decay_begin)
            epsilon = epsilon - eps_decay_val;
    end
if mod(t,1000) == 0
    movie(xx,yy,Tpos(1),Tpos(2),201)
end
% if tot_reward(t) >= -50
%      movies(I,T,Tpos(1),Tpos(2),201)
% end
end
function [rew,SX] = Reward(x,y,Tpos,WinD,vx,vy,oldx,oldy)
VDir = atan2(vy,vx)
VDir = rad2deg(VDir)

% TargetMag = sqrt(Tpos(1)^2 + Tpos(2)^2)
% TDir = atan2(y,x)
% TDir = rad2deg(TDir)
% DroneMag = sqrt(x^2 + y^2)
% OldDroneMag = sqrt(oldx^2 + oldy^2)
Tpos = transpose(Tpos)
v = [x;
    y]
DirecTar = (Tpos - v) / norm(Tpos - v)
WDir = WinD/norm(WinD)
Ang = dot(DirecTar,WDir)

Oerror = sqrt((Tpos(1) - oldx)^2 + (Tpos(2) - oldy)^2)
Nerror = sqrt((Tpos(1) - x)^2 + (Tpos(2) - y)^2)
if Nerror <= 5
    rew = 1
    SX = 1
elseif Nerror < Oerror
    rew = 0.5 * (1 - Nerror/1030.533842)
    SX = 0
    if Ang > 0
        rew = rew + 0.4
    elseif Ang < 0
        rew = rew - 0.4
    else
        rew = 0
    end
else Nerror > Oerror
    if Ang > 0
        rew = -1 + 0.5
        SX = 0
    else
        rew = -1
        SX = 0
    end
end
end

% if abs(WDir - VDir) <= 10 && abs(TDir - VDir) <= 10 && Nerror == 0
%     rew = 2
%     SX = 1
% elseif abs(WDir - VDir) <= 10 && abs(TDir - VDir) <= 10 && Nerror < Oerror && Nerror ~= 0
%     rew = 1
%     SX = 0
% elseif (abs(WDir - VDir) <= 10 && abs(TDir-VDir)) >= 10 && Nerror < Oerror|| (abs(WDir - VDir) >= 10 && abs(TDir-VDir) <= 10) &&  Nerror < Oerror  
%     rew = 0
%     SX = 0
% elseif (abs(WDir - VDir) >= 10 && abs(TDir-VDir) >= 10) || Nerror > Oerror
%     rew = -1
%     SX = 0
% elseif abs(WDir - VDir) <= 10 && abs(TDir - VDir) >= 10 && Nerror > Oerror
%     rew = -2
%     SX = 0
% else
%     rew = -1
%     SX = 0
% end

function[bucket_index] = bucket(state,bucketSZ,low,n_buckets)
%Loop over each dimension
    %Could be made into a function
    for i = 1:6
        % Calculate the bucket index
        bucket_index(i) = floor((state(i) - low(i)) / bucketSZ(i))+1;
        % Ensure the index is within valid range
        if bucket_index(i) <= 0
            bucket_index(i) = 1;
        elseif bucket_index(i) >= n_buckets(i)
            bucket_index(i) = n_buckets(i) - 1;
        end
    end
end
    function [x,y,vx,vy,WinD,action,angle_deg,awvx,awvy] = Act(action,time,IntVx,IntVy,WindV,PosX,PosY,px,py,pvx,pvy)
mass = 1
if action == 9 %Stay in same Direction
    % % Randomly select an angle in degrees within the given range
    % angle_deg = randi([170,190],1)
    % 
    % F_engy = 5sin(angle_deg) %Newtons
    % F_engx = 5cos(angle_deg) %Newtons
    % Accx = F_engx/mass
    % Accy = F_engy/mass
    
    %No need to accelerate just maintain velocity
    
    angle_deg = atan2(IntVy,IntVx)
    angle_deg = rad2deg(angle_deg)
    fvx = @(time)IntVx + WindV(1)
    vx = IntVx + WindV(1)
    fvy = @(time) IntVy + WindV(2)
    vy = IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    % Reflecting X-axis
    if x < 0
        x = abs(x);  % Reflect off the left boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the right boundary
    else x = x
    end
    
   
    y = integral(fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    % Reflecting Y-axis
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    else
        y=y
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    % Lwvx = len(wvx)
    % neg = []
    % posit = []
    % indW = 1
    % indWW = 1
    % for y = 1:Lwvx
    %     if wvx(y) < 0
    %         neg(indW) = wvx(y)
    %         indW = indW + 1
    %     else
    %         posit(indWW) = wvx(y)
    %         indWW = indWW + 1
    %     end
    % end
    
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 1 %Move North
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([80,100],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    
    FAccx = @(time) F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time) F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time, 'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time, 'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the left boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the right boundary
    end
    
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 2 % Due South
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([260,280],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time) F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time) F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time, 'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time, 'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
   
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 3 %Move East
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([-10,10],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time) F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time) F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time, 'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time, 'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time, 'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time, 'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 4 %Move West
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([170,190],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time) F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time) F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time, 'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time, 'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 5 %Move NE
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([11,79],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time)F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time)F_engy/mass
    Accy = F_engy/mass

    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time, 'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time, 'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 6 %Move NW
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([101,169],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time)F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time)F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time,'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time, 'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 7 % Move SW
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([191,259],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time) F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time) F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time,'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time,'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
elseif action == 8 %Move SE
    % Randomly select an angle in degrees within the given range
    angle_deg = randi([281,349],1)
    
    F_engy = 5*sind(angle_deg) %Newtons
    F_engx = 5*cosd(angle_deg) %Newtons
    FAccx = @(time)F_engx/mass
    Accx = F_engx/mass
    FAccy = @(time)F_engy/mass
    Accy = F_engy/mass
    
    %Integrate the acceleration to get velocity
    vx = integral(FAccx, time-1, time,'ArrayValued', true);
    vx = vx + IntVx + WindV(1)
    Fvx = @(time)vx + IntVx + WindV(1)
    vy = integral(FAccy, time-1, time,'ArrayValued', true);
    vy = vy + IntVy + WindV(2)
    Fvy = @(time)vy + IntVy + WindV(2)
    %Integrate velocity to get position
    x = integral(Fvx,time-1,time,'ArrayValued', true)
    x = x + PosX
    if x < 0
        x = abs(x);  % Reflect off the bottom boundary
    elseif x > 1600
        x = 2 * 1600 - x;  % Reflect off the top boundary
    end
    y = integral(Fvy,time-1,time,'ArrayValued', true)
    y = y + PosY
    if y < 0
        y = abs(y);  % Reflect off the bottom boundary
    elseif y > 1600
        y = 2 * 1600 - y;  % Reflect off the top boundary
    end
    %Determine local wind velocity & Direction
    [~,idx] = unique(px); %position of wind vectors
    [~,idy] = unique(py)
    % g_endX = x + 10
    % g_endY = y + 10
    % gx = (g_endX - 10):1:g_endX
    % gy = (g_endY - 10):1:g_endY
    
    
    wvx = interp1(px(idx),pvx(idx),PosX,'linear','extrap') %interpolated wind-x velocities
    awvx = wvx
    wvy = interp1(py(idy),pvy(idy),PosY,'linear','extrap') %interpolated wind y-velocities
    awvy = wvy
    WinD = [awvx;
            awvy]
    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
end
end

function [videoWriterObj] = movie(x,y,pos1,pos2,len)
% Define video parameters
outputVideoFile = strcat('Trajectories_', datestr(now, 'yyyymmdd_HHMMSS'), '.mp4');
%outputVideoFile = strcat('TrajectoriesSSS.mp4');  % Specify the output video file
frameRate = 30;  % Frames per second
% Create a VideoWriter object
videoWriterObj = VideoWriter(outputVideoFile, 'MPEG-4');
videoWriterObj.FrameRate = frameRate;
% Open the video file for writing
open(videoWriterObj);
% Loop through your data (in this example, we use random images)
numFrames = len; %length of position series
%load(strcat('data.mat'), 'xpos', 'zpos')
xpos = x;%x
ypos = y;%y

clf
PlotVel;
figure(2);

hold on

% Plot the target marker (will remain visible throughout)
plot(pos1, pos2, '*', 'MarkerSize', 20, 'Color', 'r');

% Set fixed axis limits (for consistent view)
axis([0 1600 0 1600]);  % Example for a 1600x1600 

% plot(pos1,pos2,'.');
%plot(100,100,'*','MarkerSize',20 )
% plot(200,150,'*','MarkerSize',20 )
for frame = 1:numFrames
    
    Ny_par = ypos(1:frame);
    Nx_par = xpos(1:frame);
    plot(Nx_par,Ny_par,'b-',LineWidth=1.5);
    
    
    xlb = xlabel('$y$ (m)','interpreter','Latex');
    ylb = ylabel('$x$ (m)','interpreter','Latex');
    set(gca,'TickLabelInterpreter','latex','fontsize',14);
    set([xlb,ylb],'interpreter','Latex','fontsize',18);
    %axis([-15 5 0 1])
    set(gcf, 'Position', [-1200, -1400, 1200, 1400]);
    title(strcat('Numframe=', num2str(frame)), 'Interpreter','latex','FontSize',18);
    % Write the frame to the video
    writeVideo(videoWriterObj, getframe(gcf));
    pause(0.01);
end

hold off
% Close the video file
close(videoWriterObj);
end

% function [videoWriterObj] = movies(x,y,pos1,pos2,len)
% % Define video parameters
% outputVideoFile = strcat('Trajectories_', datestr(now, 'yyyymmdd_HHMMSS'), '.mp4');
% 
% %outputVideoFile = strcat('TrajectoriesSSS.mp4');  % Specify the output video file
% frameRate = 30;  % Frames per second
% % Create a VideoWriter object
% videoWriterObj = VideoWriter(outputVideoFile, 'MPEG-4');
% videoWriterObj.FrameRate = frameRate;
% % Open the video file for writing
% open(videoWriterObj);
% % Loop through your data (in this example, we use random images)
% numFrames = len; %length of position series
% %load(strcat('data.mat'), 'xpos', 'zpos')
% xpos = x;%x
% ypos = y;%y
% 
%  clf
%  Plot_vel_vector();
%  figure(1);
% hold on
% %trajectoryHandle =
% plot(NaN, NaN, 'b-', 'LineWidth', 1.5);  % Initialize empty plot for trajectory
% 
% % Plot the target marker (will remain visible throughout)
% plot(pos1, pos2, '*', 'MarkerSize', 20, 'Color', 'r');
% 
% % Set fixed axis limits (for consistent view)
% axis([0 1600 0 1600]);  % Example for a 1600x1600 
% % plot(pos1,pos2,'.');
% %plot(100,100,'*','MarkerSize',20 )
% % plot(200,150,'*','MarkerSize',20 )
% for frame = 1:numFrames
%     
%     Ny_par = ypos(1:frame);
%     Nx_par = xpos(1:frame);
%     % Update the trajectory plot without affecting the wind field
%     %set(trajectoryHandle, 'XData', Nx_par, 'YData', Ny_par);
%     plot(Nx_par,Ny_par,'b-',LineWidth=1.5);
%     
%     
%     xlb = xlabel('$y$ (m)','interpreter','Latex');
%     ylb = ylabel('$x$ (m)','interpreter','Latex');
%     set(gca,'TickLabelInterpreter','latex','fontsize',14);
%     set([xlb,ylb],'interpreter','Latex','fontsize',18);
%     %axis([-15 5 0 1])
%     set(gcf, 'Position', [-1200, -1400, 1200, 1400]);
%     title(strcat('Numframe=', num2str(frame)), 'Interpreter','latex','FontSize',18)
%     % Write the frame to the video
%     writeVideo(videoWriterObj, getframe(gcf));
%     pause(0.01)
% end
% 
% hold off
% 
% % Close the video file
% close(videoWriterObj);
% end
%     curr_state
%     % State to be discretized
% 
% state = [s1, s2, s3, s4, s5, s6];
% 
% % Preallocate bucket index array
% bucket_index = zeros(1, 6);
% 
% 
% 
% 
%     %Variable for Target being reached
%     SX = 1
% 
%     while (~SX && count_step<200)
%         count_step = count_step + 1
% 
% 
% 
% 
% 
% 
% 
