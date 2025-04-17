clear; close all; clc;
%%%%
%%%
tau = 0.005

discount = 0.95

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


% %Initialize Agent State (Positions,Velocity,Wind)
for i = 1:numAgents
    actorNetwork{i} = createExMADDPG_Actor(stateDim, actionDim);
    targetActor{i} = actorNetwork{i}; %
end
% 
criticNetwork = createExMADDPG_Critic(stateDim,actionDim,numAgents);
targetCritic = criticNetwork;

%Initialize Replay Buffer
replayBuffer = {}

maxEp = 2000

for x_Ep = 1:maxEp
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
        % actorNetwork{i} = createExMADDPG_Actor(stateDim, actionDim);
        % targetActor{i} = actorNetwork{i}; %
    end
    % criticNetwork = createExMADDPG_Critic(stateDim,actionDim,numAgents);
    % targetCritic = criticNetwork;

    
    
    SX = 0
    count_step = 0
    time = 0
    curr_rew = 0
    totalRewards = 0
    while (SX ~= 1 && count_step<100)
        time = time + 1
        count_step = count_step + 1
        for x_nAgents = 1:numAgents
            state{x_nAgents}{1} = {[Dpos{x_nAgents},DVel{x_nAgents},W_x_y{x_nAgents}]}
            statEs{x_nAgents}{1} = {[Dpos{x_nAgents},DVel{x_nAgents},W_x_y{x_nAgents}]}
            s = cell2mat(state{x_nAgents}{1})
            s = dlarray(s)
            %ActorNetwork
            % actorNetwork = createExMADDPG_Actor(stateDim,actionDim)
            probs = predict(actorNetwork{x_nAgents}, s); % Get action based on the state
            [~, action] = max(probs); % Choose action with highest probability
            state{x_nAgents}{2} = action
            y(x_nAgents) = action
            statEs{x_nAgents}{2} = action
            EnvironmentState = state{x_nAgents}{1}
            EnvironmentState = cell2mat(EnvironmentState)
            WindV = [EnvironmentState(5),EnvironmentState(6)]
            [Newx,Newy,Newvx,Newvy,NeWDir,action,angle_deg,Newawvx,Newawvy] = Act(action,time,EnvironmentState(3),EnvironmentState(4),WindV,EnvironmentState(1),EnvironmentState(2),px,py,pvx,pvy)
            agentPositions(x_nAgents,:) = [Newx,Newy]
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
        y = dlarray(y)



        %Critic Network
        k1 = cell2mat(statEs{1}{1})
        k2 = cell2mat(statEs{2}{1})
        k3 = cell2mat(statEs{3}{1})
        k4 = cell2mat(statEs{4}{1})

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
        % globalstates_dl = dlarray(globalstates)
        % globalAction_dl = dlarray(globalAction)
        % Q_pred = predict(criticNetwork, globalstates_dl,globalAction_dl);

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
      
            
            jointNextStates = zeros(batchSize, numAgents * stateDim);
            
            for j = 1:batchSize
                agentStates = nextStatesGlobal{j};  % This is a 1x4 cell
                flatState = [];
                flatState = [flatState, agentStates];  % concatenate 1x6 → 1x24
                jointNextStates(j, :) = flatState;  % Store as row j
            end

            jointStates = zeros(batchSize, numAgents * stateDim);
            
            for j = 1:batchSize
                agentStatess = states{j};  % This is a 1x4 cell
                flatStates = [];
                flatStates = [flatStates, agentStatess];  % concatenate 1x6 → 1x24
                jointStates(j, :) = flatStates;  % Store as row j
            end
            
            jointActions = zeros(batchSize, numAgents * actionDim);
            
            for j = 1:batchSize
                agentA = actions{j};  % This is a 1x4 cell
                flatA = [];
                flatA = [flatA, agentA];  % concatenate 1x6 → 1x24
                jointActions(j, :) = flatA;  % Store as row j
            end
            h = dlarray(jointActions)
            hh = jointActions'
            % % Split into individual agent states
            % nextStates = cellfun(@(s) mat2cell(s, 1, repmat(stateDim, 1, numAgents)), nextStatesGlobal, 'UniformOutput', false);
            % g1 = nextStates{1}{1}
            % g2 = nextStates{2}{1}
            % g3 = nextStates{3}{1}
            % g4 = nextStates{4}{1}
            % gg = [g1,g2,g3,g4]

            % === Compute Target Q-value (Bellman Equation) ===
            nextStates = cellfun(@(s) mat2cell(s, 1, repmat(stateDim, 1, numAgents)), nextStatesGlobal, 'UniformOutput', false);
            NACC = cellfun(@(s) mat2cell(s, 1, repmat(stateDim, 1, numAgents)), states, 'UniformOutput', false);
            for j = 1:batchSize
                nextGlobalAction = []
                for k = 1:numAgents
                    AC = nextStates{j}{k}
                    nextStatess = dlarray(AC);
                    nextActionProbs = predict(targetActor{i}, nextStatess);
                    [~, nextAction] = max(nextActionProbs);
            
                    oneHotNextAction = zeros(1, actionDim);           
                    oneHotNextAction(nextAction) = 1;
                    nextGlobalAction = [nextGlobalAction, oneHotNextAction];
                end
                g(j,:) = nextGlobalAction
            end






            
            %for i = 1:numAgents
                % nextStatess = nextStates{i}{1}
            nextStatess = dlarray(jointNextStates)
            % nextActionProbs = predict(targetActor{i}, nextStatess);
            % [~, nextAction] = max(nextActionProbs);
            % 
            % oneHotNextAction = zeros(64, (actionDim*numAgents));           
            % oneHotNextAction(nextAction) = 1;
            % nextGlobalAction = [nextGlobalAction, oneHotNextAction]; % Store all next actions
            % %end
            %nextGlobalAction = dlarray(nextGlobalAction)
            %g = dlarray(gg)
            
            % Compute target Q-values for training critic
            % Q_target = jointRewards + discount * Q_next;  % [64 × 1]

            globalstates_dl = dlarray(jointStates);
            globalAction_dl = dlarray(jointActions);
            Q_next = predict(targetCritic, jointNextStates, g);
            % Q_target = cellfun(@(r, q) r + discount * q, transpose(rewards), num2cell(Q_next), 'UniformOutput', false);
            %Q_target = jointRewards + discount * Q_next;  % [64 × 1]
            for tk = 1:batchSize
                r = []
                for tkk = 1:numAgents
                    r = rewards{tk}
                    rew(tk,1) = sum(r)
                end
            end
            Q_target = cellfun(@(r, q) r + discount * q, num2cell(rew), num2cell(Q_next), 'UniformOutput', false);
            Q_pred = predict(criticNetwork, globalstates_dl,globalAction_dl);

            % === Train Critic Network (MSE Loss) ===
            Q_pred = dlarray(Q_pred','CB')
            Q_target = dlarray(cell2mat(Q_target'), 'CB');
            
            jointStates = dlarray(jointStates','CB')
            jointActions = dlarray(jointActions','CB')


            [loss, gradients] = dlfeval(@criticLoss, criticNetwork, ...
                                        jointStates, jointActions, Q_target);

            learnRate = double(0.005);
            gradDecay = 0.9;
            sqGradDecay = 0.999;
            iteration = 64
            
            % Initialize optimizer state
            trailingAvg = [];
            trailingAvgSq = [];

            
            [criticParams, trailingAvg, trailingAvgSq] = adamupdate( ...
                criticNetwork.Learnables, gradients, ...
                trailingAvg, trailingAvgSq, ...
                iteration,learnRate, 0.9, 0.999);
            
            % Get the current learnable parameters of the critic network
            % criticParams = getLearnables(criticNetwork);
            
            % Update the critic network's learnables with the new parameters
            % criticNetwork = setLearnableParameters(criticNetwork, criticParams);
            % 
            % % Now, you can proceed with the rest of the optimization process
            % 
            % 
            % criticNetwork = setLearnables(criticNetwork, criticParams);

            % criticNetwork = adamupdate(criticNetwork, gradients);
            
            % Use dlupdate to apply the updated parameters
            criticNetwork = dlupdate(@(w, g) w - learnRate * g, criticNetwork, criticParams);
            targetCritic = softUpdate(targetCritic, criticNetwork, tau);





            % loss = mse(Q_pred, Q_target);
            % gradients = dlgradient(loss, criticNetwork.Learnables);
            % criticNetwork = adamupdate(criticNetwork, gradients);
            
            % === Train Actor Networks (Policy Gradient) ===

            % for ii = 1:64
            %     stateSS = []
            %     actionPreD = []
            %     for i = 1:numAgents
            %             state = NACC{ii}{i}
            %             stateS{ii,i} = dlarray(state)
            %             % SS = [stateSS,dlarray(state)]
            % 
            %             actionPred{ii,i} = predict(actorNetwork{i}, stateS{ii,i}); % Actor1s chosen action
            %     end
            %     % stateSS(ii,:) =
            % end


            % % Stack all states for agent i across batch
            % for i = 1:numAgents  % Loop over each agent
            %     agentIStates = zeros(batchSize, stateDim);
            % 
            %     for ii = 1:batchSize  % Loop over each sample in the batch
            %         agentIStates(ii, :) = NACC{ii}{i};  % Grab agent i's state from batch
            %     end
            % 
            %     agentIStates_dl = dlarray(agentIStates', 'CB');  % [6 x 64]
            % 
            %     % Forward pass through the actor network for agent i
            %     actionPred = predict(actorNetwork{i}, agentIStates_dl);  % [9 x 64] if softmax
            % end
            % 
            % 
            % 
            % 
            % % Get all previous actions from the minibatch (jointActions already available from replay buffer)
            % 
            % jointActionsCopy = dlarray(h);  % Size [64 x 36]
            % 
            % % Replace agent i’s action portion with predicted action
            % for j = 1:batchSize
            %     [~, predictedIdx] = max(extractdata(actionPred(:,j))); % Get predicted discrete action index
            %     oneHot = zeros(1, actionDim);
            %     oneHot(predictedIdx) = 1;
            %     startIdx = (i-1)*actionDim + 1;
            %     endIdx = i*actionDim;
            %     jointActionsCopy(j, startIdx:endIdx) = oneHot;
            % end
            % 
            % jointActionsPred = dlarray(jointActionsCopy', 'CB');  % [36 x 64]
            % 
            % qValues = predict(criticNetwork, jointStates, jointActionsPred);  % [1 x 64]
            % for kl = 1:numAgents
            % 
            %     % Wrap inside dlfeval to trace the graph for gradient
            %     [policyLoss, actorGradients] = dlfeval(@actorLoss, actorNetwork{i}, agentIStates_dl, ...
            %                                            criticNetwork, jointStates, hh, i, actionDim);
            % 
            %     learnRate = 0.004;
            %     actorParams = actorNetwork{i}.Learnables;
            %     updatedParams = dlupdate(@(w, g) w - learnRate * g, actorParams, actorGradients);
            %     actorNetwork{i} = setLearnablesValue(actorNetwork{i}, updatedParams);
            % 
            % end

            for i = 1:numAgents
                % === Extract state for agent i ===
                agentStates_i = zeros(stateDim, batchSize);
                for j = 1:batchSize
                    agentStates = NACC{j};  % cell of 1x4 agent states
                    s_i = agentStates{i};   % 1x6 state for agent i
                    agentStates_i(:, j) = s_i';
                end
                dlAgentStates = dlarray(agentStates_i, 'CB');
            
                % === Compute Actor Loss using Policy Gradient ===
                [lossActor, gradientsActor] = dlfeval(@actorLoss, actorNetwork{i}, criticNetwork, ...
                                                      dlAgentStates, i, jointStates, numAgents, stateDim, actionDim);
            
                % === Update Actor Parameters ===
                [actorParams, trailingAvgA, trailingAvgSqA] = adamupdate( ...
                    actorNetwork{i}.Learnables, gradientsActor, ...
                    [], [], iteration, learnRate, gradDecay, sqGradDecay);
            
                actorNetwork{i} = dlupdate(@(w, g) w - learnRate * g, actorNetwork{i}, actorParams);
            
                % === Soft Update Target Actor ===
                targetActor{i} = softUpdate(targetActor{i}, actorNetwork{i}, tau);
            end










            
            


            % === Train Actor Networks (Policy Gradient) ===

            


            
            
        end
        
    end
end

       

        




function [loss, gradients] = actorLoss(actorNet, criticNet, agentStates, agentIdx, jointStates, numAgents, stateDim, actionDim)
    % Predict current agent's action
    actionProbs = forward(actorNet, agentStates);  % shape: [9 x B]
    [~, actions] = max(extractdata(actionProbs), [], 1);  % greedy actions
    oneHotActions = zeros(actionDim, size(agentStates, 2));
    for k = 1:size(agentStates, 2)
        oneHotActions(actions(k), k) = 1;
    end

    % Reconstruct joint actions with updated agent's action
    jointActions = reshape(extractdata(jointStates), numAgents * stateDim, []);
    jointActions = repmat(jointActions, 1, 1); % ensure it's CB format
    flatActions = zeros(numAgents * actionDim, size(agentStates, 2));
    for b = 1:size(agentStates, 2)
        for i = 1:numAgents
            if i == agentIdx
                flatActions((i-1)*actionDim+1:i*actionDim, b) = oneHotActions(:, b);
            else
                % keep the same action as used in batch
                % Note: You could also re-evaluate it here if you wish
                % but simpler to use what was in replay buffer
            end
        end
    end

    dlJointActions = dlarray(flatActions, 'CB');

    % Predict Q-values from critic
    Q = forward(criticNet, jointStates, dlJointActions);  % [1 x B]

    % Actor loss = negative of expected Q
    loss = -mean(Q);
    gradients = dlgradient(loss, actorNet.Learnables);
end


            



function [loss, gradients] = criticLoss(criticNetwork, states, actions, Q_target)
    Q_pred = predict(criticNetwork, states, actions);          % forward pass
    loss = mse(Q_pred, Q_target);                          % MSE loss
    gradients = dlgradient(loss, criticNetwork.Learnables);    % get gradients
end




function targetNet = softUpdate(targetNet, mainNet, tau)
    for i = 1:height(mainNet.Learnables)
        targetNet.Learnables.Value{i} = ...
            (1 - tau) * targetNet.Learnables.Value{i} + tau * mainNet.Learnables.Value{i};
    end
end



% criticNetwork = createExMADDPG_Critic(stateDim,actionDim,numAgents)
% targetCritic = copy(criticNetwork)
% 
% actors = cell(1, numAgents);
% targetActors = cell(1, numAgents);









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





function totalRewards = computeTotalReward(agentPositions, targetPositions, visitedTargets, w1, w2, w3, alpha)
    % Computes rewards for search efficiency with visit tracking.
    % Inputs:
    %   agentPositions  - Nx2 matrix of agent (UAV) positions
    %   targetPositions - Mx2 matrix of target positions
    %   visitedTargets  - 1xM logical array indicating visited targets
    %   w1, w2, w3 - Weights for reward components (coverage, efficiency, separation)
    %   alpha - Scaling factor for centralized reward
    % Output:
    %   totalRewards - Nx1 vector containing total rewards for each agent

    N = size(agentPositions, 1); % Number of agents
    M = size(targetPositions, 1); % Number of targets
    proximityThreshold = 20; % Distance to consider target reached

    %% 1. Compute Individual Rewards (R_i)
    R_i = zeros(N, 1); % Initialize individual rewards
    for i = 1:N
        distances = vecnorm(targetPositions - agentPositions(i, :), 2, 2);
        [minDist, idx] = min(distances);

        if minDist < proximityThreshold && ~visitedTargets(idx)
            R_i(i) = 100; % Large one-time reward
            visitedTargets(idx) = true; % Mark as visited
        elseif minDist < proximityThreshold && visitedTargets(idx)
            R_i(i) = 5; % Small bonus for staying near visited target
        else
            R_i(i) = -minDist; % Encourage moving closer
        end
    end

    %% 2. Compute Centralized Reward (R_centralized)
    R_centralized = computeCentralizedReward(agentPositions, targetPositions, w1, w2, w3);

    %% 3. Compute Total Reward for Each Agent
    totalRewards = R_i + alpha * R_centralized;
end

function R_centralized = computeCentralizedReward(agentPositions, targetPositions, w1, w2, w3)
    N = size(agentPositions, 1); % Number of agents
    numTargets = size(targetPositions, 1); % Number of targets

    %% 1. Coverage Reward: Encourage agents to explore unique areas
    minDistances = zeros(N, 1);
    for i = 1:N
        distances = vecnorm(agentPositions - agentPositions(i, :), 2, 2);
        distances(i) = inf; % Ignore self-distance
        minDistances(i) = min(distances); % Find closest other agent
    end
    R_coverage = sum(minDistances); % Encourage spreading out

    %% 2. Efficiency Reward: Encourage moving toward unexplored targets
    R_efficiency = 0;
    for i = 1:N
        distances = vecnorm(targetPositions - agentPositions(i, :), 2, 2);
        minTargetDist = min(distances);
        R_efficiency = R_efficiency - minTargetDist; % Smaller is better
    end
    R_efficiency = R_efficiency / N; % Normalize

    %% 3. Separation Reward: Avoid clustering
    agentDistances = squareform(pdist(agentPositions));
    agentDistances(agentDistances == 0) = inf;
    minAgentDistances = min(agentDistances, [], 2);
    R_separation = -sum(1 ./ (minAgentDistances + 1));

    %% Combine
    R_centralized = w1 * R_coverage + w2 * R_efficiency + w3 * R_separation;
end
