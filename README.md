# Human-agent-mobility 



## Target distribution search task

This section study provides python scripts to simulate search tasks where the targets follow a parametric distribution and the agent's objective is to maximize targets collected and minimize distance traveled. 

## getting started

### software 

+ Python 3.10

*Depending on the size of the dataset and parameters simulations, the user might require access to high-performance computing (HPC)*

## Collective search with social learning
In our collective search model, agents collaborate to search for targets, which increases efficiency. More details are provided below.
### Basic model
Social learning in collective search was first proposed by Bhattacharya and Vicsek[^1] in 2014 and later investigated by Garg et al.[^2][^3]. Based on their work, we modified the model to improve its clarity.
#### Target distribution
To generate a heterogeneous target distribution, we manipulated the initial spatial clustering of targets using a power-law distribution growth model. Additionally, we assume that targets do not regenerate after being collected by agents.
#### Social learning range
### Add negtive targets
#### Target distribution
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


## Reference
[^1]: Bhattacharya, K., & Vicsek, T. (2014). Collective foraging in heterogeneous landscapes. Journal of the Royal Society Interface, 11(100), 20140674.
[^2]: Garg, K., Kello, C. T., & Smaldino, P. E. (2022). Individual exploration and selective social learning: balancing exploration–exploitation trade-offs in collective foraging. Journal of the Royal Society Interface, 19(189), 20210915.
[^3]: Garg, K., Smaldino, P. E., & Kello, C. T. (2024). Evolution of explorative and exploitative search strategies in collective foraging. Collective Intelligence, 3(1), 26339137241228858.
