# Human-agent-mobility 



## Target distribution search task

This section study provides python scripts to simulate search tasks where the targets follow a parametric distribution and the agent's objective is to maximize targets collected and minimize distance traveled. 

## getting started

### software 

+ Python 3.10

*Depending on the size of the dataset and parameters simulations, the user might require access to high-performance computing (HPC)*

## Collective search with social learning
### Basic Model
#### Social learning range
### Add negtive targets
#### Repulsive 'force' by detected negative targets

## RL models for deterministic environment:

We simulate search tasks where target locations follow a hierarchical distribution and their locations are fixed between search tasks (episodes) but unknown to the agent. 

The following methods were used previously to simulate target search. Although they can be used to learn optimal policies, they have some limitations which are presented here.

### Tabular SARSA & Q-learning methods:

Under tabular methods, each state is represented as an entry of the value function table. In our problem setting, we define states as the location of the agent in the search space; however, we have to discretize the search space such that we can assign an entry for each location of the agent. 

#### Limitations:

+ Large number of states: If the radius of detection is very small compared to the search space, the number of states is too large that updating all of them will take a long computational time.

  
### Planning & learning:

This method is similar to the previous one, the difference is that updating the values will made without extra information about the environment. 

#### Limitations:

+ Large number of states: If the radius of detection is tiny compared to the search space, the number of states is too large that updating all of them will take a long computational time.


### Actor-critic methods:

To reduce the number of states, we use approximation methods. The value function and policies are both parametrized and states are defined by a specific transformation of the search space. 
 
##### Limitations:

+ Rewards give information on the following quantities: distance, direction, and whether the agent is founded or not. However, we need independent modifications to each quantity during the learning process. Additionally, information on the targets collected can be used more efficiently.


 ### Actor-critic with posterior Approximation

 This is the most recent model used. Here, we plan to use all the previous experiences to improve the learning process. Namely, we plan to modify rewards from the environment to split them into two separate rewards and learn two independent value functions. 



