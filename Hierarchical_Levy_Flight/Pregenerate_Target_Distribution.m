clear; clc;
%This code pre-generate target distribution and trajectories of the soldier

%Generate and save target distribution
Dim = 100000; % dimension of the domain
Num_target = 2500;
% board = [0,0;Dim,0;Dim,Dim;0,Dim;0,0];

% pre-determination
V_best = 1;
R = 50; % detection range
lambda = 1/(2*R*(Num_target/Dim^2));
TypeDist = 1000; Num_SingleLevy = 20;
%% generate 20 groups of distribution in a 3-dimensional matrix
TarDist = zeros(Num_target,2,TypeDist);
for i = 1:TypeDist
    Real_Target = rand(Num_target,2)*Dim;
    TarDist(:,:,i) = Real_Target; 
end
save('TargetDistribution.mat','TarDist');

%% Single Levy search
maxAgent = lambda;
mu_agent = 1;
TimeLength_record_Single = zeros(TypeDist, Num_SingleLevy);
SingleLevyMove = zeros(80000,2, Num_SingleLevy); %same for different distribution
for j = 1:Num_SingleLevy
    StepLength = RandTruncLevyMulti(mu_agent, maxAgent, 80000)+1;
    StepDirec = rand(80000,1)*2*pi; % select a random angle
    SingleLevyMove(:,1,j) = StepLength;
    SingleLevyMove(:,2,j) = StepDirec;
end
save('MovementPatterm.mat','SingleLevyMove');

%% single levy compensation
load('TargetDistribution.mat', 'TarDist')
load('SingleLevy.mat')
load('SingleLevyMovement.mat')
maxAgent = lambda;
mu_agent = 1;
% TimeLength_record_Single = zeros(TypeDist, Num_SingleLevy);
% SingleLevyMove = zeros(80000,2, Num_SingleLevy, TypeDist);
%%
for i = 5:5%TypeDist
    Real_Target = TarDist(:,:,i);
    SingleCount = 1;
    for j = 1:1 %remove the cases with search time over 100 hours and compensate
        StepLength = RandTruncLevyMulti(mu_agent, maxAgent, 80000)+1;
        StepDirec = rand(80000,1)*2*pi; % select a random angle
        SingleLevyMove(:,1,10,i) = StepLength;
        SingleLevyMove(:,2,10,i) = StepDirec;

        xL = Dim/2; yL = Dim/2; % initial positions
        T_total_Single = 0;
        trajx = xL; trajy = yL;
        SingleIndex = 1; %using for sample value from the saved data
        
        while min(sqrt((xL- Real_Target(:,1)).^2 + (yL- Real_Target(:,2)).^2)) > R
            theta = StepDirec(SingleIndex);
            ux = cos(theta); uy = sin(theta);
            r = StepLength(SingleIndex);

%             theta = rand(1)*2*pi; % select a random angle
%             ux = cos(theta); uy = sin(theta); % make a random unit vector u(ux, uy)
%             r = RandTruncLevyAlpha(mu_agent, maxAgent); % Levy(0,c) truncated into [0,2]
            xL = xL +r*ux; yL = yL + r*uy;
            
            if xL < 0
                xL = abs(xL);
            elseif xL > Dim
                xL = 2*Dim - xL;
            elseif yL < 0
                yL = abs(yL);
            elseif yL > Dim
                yL = 2*Dim - yL;
            end

            trajx = [trajx; xL]; trajy = [trajy; yL]; % extend coordinates
            T_total_Single = T_total_Single + r;
            SingleIndex = SingleIndex + 1;

            % judge if pass the target during each time step
            if SingleIndex >=2
                for mm = 1:Num_target
                    v1 = [trajx(SingleIndex), trajy(SingleIndex)]; % the first vertex
                    v2 = [trajx(SingleIndex-1), trajy(SingleIndex-1)]; % the second vertex
                    vec1 = v2 - v1;
                    vec2 = Real_Target(mm,:) - v1;
                    if dot(vec1,vec2)>=0 && dot(vec1,vec2)<=dot(vec1,vec1)
                        d = abs( det([vec2;vec1]) )/norm(v2-v1); % distance from the real target to v1-v2
                        if d <= R
                            buff = norm(vec1) - sqrt(norm(vec2)^2-d^2);
                            T_total_Single = T_total_Single - buff; %select the precise time
                            break
                        end   
                    end
                end         
            end
             
            if T_total_Single/3600 > 100 % the set upper limit of search time
                break
            end
        end

        if T_total_Single/3600 >= 100
            continue
        else
            SingleCount = SingleCount + 1;
        end

        if SingleCount >= Num_SingleLevy + 2
           break
        end
        TimeLength_record_Single(i, 10) = T_total_Single/3600;
        disp(['using ',num2str(T_total_Single/3600),' hours']);
        %save time-series of step length and search time
    end
end
save('SingleLevy.mat','TimeLength_record_Single'); %rows for distribution and columns for Levy search
save('SingleLevyMovement.mat','SingleLevyMove');