clc; clear all; clf;

tau = 0.005

w1 = 0.3
w2 = 0.4
w3 = 0.3
alpha = 0.5
%%Construct Q-Table
[px,py,pvx,pvy]=Plot_vel_vector();

% state = [Dpos(1), Dpos(2), DVel(1), DVel(2), px, py, pvx, pvy]
stateDim = 6;

%action = [N,NE,NW,S,SE,SW,E,W,MaintainDirection]
actionDim = 9;

%number of agents (User Input)
numAgents = 4;

%Replay Buffer Size
bufferSize = 100000

%Batch Size
batchSize = 64
%Target Location
Tpos{1} = [1430,1200];
Tpos{2} = [1000,800];
Tpos{3} = [500,400];
Tpos{4} = [200,1500];

%Determine local wind velocity & Direction
[~,idx] = unique(px); %position of wind vectors
[~,idy] = unique(py)
% g_endX = Dpos(1) + 10
% g_endY = Dpos(2) + 10
% gx = (g_endX - 10):1:g_endX
% gy = (g_endY - 10):1:g_endY


%Initialize Agent State (Positions,Velocity,Wind)
for i = 1:numAgents
    Dpos{i} = randi([20,1500],1,2);
    PosPair = Dpos{i};
    XPos = PosPair(1);
    YPos = PosPair(2);
    wvx = interp1(px(idx),pvx(idx),XPos); %interpolated wind-x velocities
    awvx = wvx;
    wvy = interp1(py(idy),pvy(idy),YPos); %interpolated wind y-velocities
    awvy = wvy;
    W_x_y{i} = [awvx,awvy];
    DVel{i} = [1,0];
    actorNetwork{i} = createExMADDPG_Actor(stateDim, actionDim);
    targetActor{i} = actorNetwork{i}; %
end

criticNetwork = createExMADDPG_Critic(stateDim,actionDim,numAgents);
targetCritic = criticNetwork;

%Initialize Replay Buffer
replayBuffer = {}

maxEp = 2000

for x_Ep = 1:maxEp
    SX = 0
    count_step = 0
    time = 0
    curr_rew = 0
    while (SX ~= 1 && count_step<100)
        time = time + 1
        count_step = count_step + 1
        for x_nAgents = 1:numAgents
            state{x_nAgents}{1} = {[Dpos{x_nAgents},DVel{x_nAgents},W_x_y{x_nAgents}]}
            states{x_nAgents}{1} = {[Dpos{x_nAgents},DVel{x_nAgents},W_x_y{x_nAgents}]}
            s = cell2mat(state{x_nAgents}{1})
            s = dlarray(s)
            %ActorNetwork
            % actorNetwork = createExMADDPG_Actor(stateDim,actionDim)
            probs = predict(actorNetwork{i}, s); % Get action based on the state
            [~, action] = max(probs); % Choose action with highest probability
            state{x_nAgents}{2} = action
            states{x_nAgents}{2} = action
            EnvironmentState = state{x_nAgents}{1}
            EnvironmentState = cell2mat(EnvironmentState)
            WindV = [EnvironmentState(5),EnvironmentState(6)]
            [Newx,Newy,Newvx,Newvy,NeWDir,action,angle_deg,Newawvx,Newawvy] = Act(action,time,EnvironmentState(3),EnvironmentState(4),WindV,EnvironmentState(1),EnvironmentState(2),px,py,pvx,pvy)
            agentPositions(i,:) = [Newx,Newy]
            Dpos{x_nAgents} = [Newx,Newy]
            DVel{x_nAgents} = [Newvx,Newvy]
            W_x_y{x_nAgents} = [Newawvx,Newawvy]
            state{x_nAgents}{1} = {[Dpos{x_nAgents},DVel{x_nAgents},W_x_y{x_nAgents}]}
            NextState{x_nAgents}{1} = state{x_nAgents}{1}
        end
        for k = 1:4
            targetPositions(k,:) = Tpos{k}
        end
        totalRewards = computeTotalReward(agentPositions, targetPositions, w1, w2, w3, alpha)
        curr_rew = curr_rew + totalRewards
        rewards = curr_rew



        %Critic Network
        k1 = cell2mat(states{1}{1})
        k2 = cell2mat(states{2}{1})
        k3 = cell2mat(states{3}{1})
        k4 = cell2mat(states{4}{1})

        k11 = cell2mat(NextState{1}{1})
        k12 = cell2mat(NextState{2}{1})
        k13 = cell2mat(NextState{3}{1})
        k14 = cell2mat(NextState{4}{1})



        globalstates = [k1,k2,k3,k4]
        NextStates = [k11,k12,k13,k14]
        oneHotActions = zeros(numAgents, 9); % Initialize matrix (4x9)
        for i = 1:numAgents
            oneHotActions(i, state{i}{2}) = 1; % Set the selected action index to 1
        end
        globalAction = reshape(oneHotActions', 1, []); % Flatten to [1 x 36]
        %criticNetwork = createExMADDPG_Critic(globalstates,globalAction,numAgents);
        criticInput = [globalstates,globalAction,numAgents];

        % nummA = dlarray(numAgents)
        globalstates_dl = dlarray(globalstates)
        globalAction_dl = dlarray(globalAction)
        Q_pred = predict(criticNetwork, globalstates_dl,globalAction_dl);

        % === Store Experience in Replay Buffer ===
        if length(replayBuffer) >= bufferSize
            replayBuffer(1) = []; % Remove oldest transition if buffer full
        end
        replayBuffer{end+1} = {globalstates, globalAction, totalRewards, NextStates};

        % === Sample Minibatch for Training ===
        if length(replayBuffer) >= batchSize
            minibatch = datasample(replayBuffer, batchSize);
            
            % Extract states, actions, rewards, next states
            states = cellfun(@(x) x{1}, minibatch, 'UniformOutput', false);
            actions = cellfun(@(x) x{2}, minibatch, 'UniformOutput', false);
            rewards = cellfun(@(x) x{3}, minibatch, 'UniformOutput', false);
            nextStatesGlobal = cellfun(@(x) x{4}, minibatch, 'UniformOutput', false);
            % Split into individual agent states
            nextStates = cellfun(@(s) mat2cell(s, 1, repmat(stateDim, 1, numAgents)), nextStatesGlobal, 'UniformOutput', false);
            g1 = nextStates{1}{1}
            g2 = nextStates{2}{1}
            g3 = nextStates{3}{1}
            g4 = nextStates{4}{1}
            gg = [g1,g2,g3,g4]

            % === Compute Target Q-value (Bellman Equation) ===
            nextGlobalAction = [];
            for i = 1:numAgents
                nextStatess = nextStates{i}{1}
                nextStatess = dlarray(nextStatess)
                nextActionProbs = predict(targetActor{i}, nextStatess);
                [~, nextAction] = max(nextActionProbs);
                oneHotNextAction = zeros(1, actionDim);
                oneHotNextAction(nextAction) = 1;
                nextGlobalAction = [nextGlobalAction, oneHotNextAction]; % Store all next actions
            end
            nextGlobalAction = dlarray(nextGlobalAction)
            g = dlarray(gg)
            Q_next = predict(targetCritic, g, nextGlobalAction);
            Q_target = cellfun(@(r, q) r + gamma * q, rewards, num2cell(Q_next));
            
            % === Train Critic Network (MSE Loss) ===
            loss = mse(Q_pred, Q_target);
            gradients = dlgradient(loss, criticNetwork.Learnables);
            criticNetwork = adamupdate(criticNetwork, gradients);
            
            % === Train Actor Networks (Policy Gradient) ===
            for i = 1:nAgents
                state = cell2mat(state{i}{1})
                state = dlarray(state)
                actionPred = predict(actorNetwork{i}, state); % Actor1s chosen action
                actorInput = [state, actionPred]; % Critic needs (state, action)
                policyGradient = gradient(predict(criticNetwork, actorInput), actionPred);
                
                actorGradients = gradient(actionPred, actorNetwork{i}.Learnables);
                actorUpdate = policyGradient * actorGradients; % Chain rule
                
                actorNetwork{i} = adamupdate(actorNetwork{i}, actorUpdate);
            end
            
            % === Soft Update Target Networks ===
            targetCritic = tau * criticNetwork + (1 - tau) * targetCritic;
            for i = 1:nAgents
                targetActor{i} = tau * actorNetwork{i} + (1 - tau) * targetActor{i};
            end
        end
        
    end
end

       

        





            










criticNetwork = createExMADDPG_Critic(stateDim,actionDim,numAgents)
targetCritic = copy(criticNetwork)

actors = cell(1, numAgents);
targetActors = cell(1, numAgents);









%Critic Network tracks states and actions of all agents
function criticNetwork = createExMADDPG_Critic(stateDim, actionDim, numAgents)
    % Input for all agents' states and actions
    totalStateDim = stateDim * numAgents;
    totalActionDim = actionDim * numAgents;

    stateInput = featureInputLayer(totalStateDim, 'Name', 'stateInput');
    actionInput = featureInputLayer(totalActionDim, 'Name', 'actionInput');

    % Hidden layers for combined inputs
    commonPath = [
        concatenationLayer(1, 2, 'Name', 'concat')
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'qValue') % Single Q-value output
    ];

    % Combine inputs into layer graph
    criticNetwork = layerGraph();
    criticNetwork = addLayers(criticNetwork, stateInput);
    criticNetwork = addLayers(criticNetwork, actionInput);
    criticNetwork = addLayers(criticNetwork, commonPath);

    % Connect state and action inputs to common path
    criticNetwork = connectLayers(criticNetwork, 'stateInput', 'concat/in1');
    criticNetwork = connectLayers(criticNetwork, 'actionInput', 'concat/in2');

    criticNetwork = dlnetwork(criticNetwork)
end

%Actor Network (Maps the action to the state of each individual agent)
function actorNetwork = createExMADDPG_Actor(stateDim, actionDim)
    layers = [
        featureInputLayer(stateDim, 'Name', 'stateInput')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(actionDim, 'Name', 'action')
        softmaxLayer('Name', 'softmax') % Convert to probabilities
    ];

    layerGraphActor = layerGraph(layers);  % Create layer graph
    actorNetwork = dlnetwork(layerGraphActor); % Convert to dlnetwork
end

function [x,y,vx,vy,WDir,action,angle_deg,awvx,awvy] = Act(action,time,IntVx,IntVy,WindV,PosX,PosY,px,py,pvx,pvy)
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

    WDir = atan2(awvy,awvx) %Wind Direction
    WDir = rad2deg(WDir)
end
end





function totalRewards = computeTotalReward(agentPositions, targetPositions, w1, w2, w3, alpha)
    % Computes rewards for search efficiency without direct assignments.
    % Inputs:
    %   agentPositions  - Nx2 matrix of agent (UAV) positions
    %   targetPositions - Mx2 matrix of target positions
    %   w1, w2, w3 - Weights for reward components (coverage, efficiency, separation)
    %   alpha - Scaling factor for centralized reward
    % Output:
    %   totalRewards - Nx1 vector containing total rewards for each agent

    N = size(agentPositions, 1); % Number of agents

    %% 1. Compute Individual Rewards (R_i)
    % Reward based on distance reduction to any nearest target
    R_i = zeros(N, 1); % Initialize individual rewards
    for i = 1:N
        % Find closest target to the agent
        distances = vecnorm(targetPositions - agentPositions(i, :), 2, 2);
        minTargetDist = min(distances); 
        
        % Reward is negative distance to the closest target
        R_i(i) = -minTargetDist; % Encourage reaching unexplored areas
    end

    %% 2. Compute Centralized Reward (R_centralized)
    R_centralized = computeCentralizedReward(agentPositions, targetPositions, w1, w2, w3);

    %% 3. Compute Total Reward for Each Agent
    totalRewards = R_i + alpha * R_centralized;
end

function R_centralized = computeCentralizedReward(agentPositions, targetPositions, w1, w2, w3)
    % Computes the centralized reward based on search efficiency.
    % Inputs:
    %   agentPositions  - Nx2 matrix of agent positions
    %   targetPositions - Mx2 matrix of target positions
    %   w1, w2, w3 - Weights for coverage, efficiency, and separation rewards
    % Output:
    %   R_centralized - Centralized reward value
    
    N = size(agentPositions, 1); % Number of agents
    numTargets = size(targetPositions, 1); % Number of targets
    
    %% 1. Coverage Reward: Encourage agents to explore unique areas
    % We approximate coverage by minimizing the distance between agents
    minDistances = zeros(N, 1);
    for i = 1:N
        distances = vecnorm(agentPositions - agentPositions(i, :), 2, 2);
        distances(i) = inf; % Ignore self-distance
        minDistances(i) = min(distances); % Find closest other agent
    end
    R_coverage = sum(minDistances); % Encourage spreading out

    %% 2. Efficiency Reward: Encourage moving toward unexplored targets
    % Reward agents for moving closer to the nearest target
    R_efficiency = 0;
    for i = 1:N
        distances = vecnorm(targetPositions - agentPositions(i, :), 2, 2);
        minTargetDist = min(distances);
        R_efficiency = R_efficiency - minTargetDist; % Smaller is better
    end
    R_efficiency = R_efficiency / N; % Normalize by number of agents

    %% 3. Separation Reward: Avoid redundant searches
    % Penalize agents that move too close to each other
    agentDistances = squareform(pdist(agentPositions));
    agentDistances(agentDistances == 0) = inf; % Ignore self-distances
    minAgentDistances = min(agentDistances, [], 2);
    R_separation = -sum(1 ./ (minAgentDistances + 1)); % Encourage spacing out

    %% 4. Combine Rewards into Centralized Reward
    R_centralized = w1 * R_coverage + w2 * R_efficiency + w3 * R_separation;
end