# Human-agent-mobility 



## Target distribution search task

This section study provides python scripts to simulate search tasks where the targets follow a parametric distribution and the agent's objective is to maximize targets collected and minimize distance traveled. 

## getting started

### software 

+ Python 3.10

*Depending on the size of the dataset and parameters simulations, user might require access to high performance computing (HPC)*

## RL models for deterministic environment:

We simulate search tasks where target locations follow a hierarchical distribution and their locations are fixed between search tasks (episodes) but unknown to the agent. 

### MC

User should transform network dataset information to run the simulations:

+ **generate_data.py:** It receives the adjency matrix of the network data set as csv file and generates N independent matrices X, where each matrix have m (number of influence samples obtained from an independent network cascades realization) rows and n columns (number of nodes of the network). User can modify the following parameters:

- m : number of influence samples.
- p_ic : probability  independent cascade model (ICM).
- N : number of ICM realizations.

**predetermined datasets:**

+ email-Eu-core dataset : [source.](https://snap.stanford.edu/data/email-Eu-core.html)
+ soc-hamsterster_v2 dataset : [source.](https://networkrepository.com/soc-hamsterster.php)
+ Erdos-Renyi: synthetic dataset using Erdos-Renyi model with parameters ($n=200, p=0.15$)

### execution:

+ **computing_mean_spread_size.py:** generates a dataset for each algorithm (setting to TRUE) given a list of parameters specified by the user (m ,k, epsilon):  

  + do_computation_exp_mech : simulation using exponential mechanism algorithm.
  + do_computation_randomized : simulation using randomized response algorithm.
  + do_computation_randomized_without_post_pr : simulation using randomized response algorithm skiping line 2  and using $\tilde{f}_0$ in line 3 of algorihm 5.
  + do_computation_greedy_algortihm: simulation using a non-private  algorithm. 
  + do_computation_greedy_algortihm_reference: simulation using a non-private  algorithm when $m \rightarrow \infty$.


+ **output structure**: 

| m  | k   | epsilon| Ixs_mu | Ixs_sd | Ixs_n |
| --- | ---- | --- | ---- | --- | ---- |
| 100 | 4 | 0.1| 120 | 20 | 10 |
| . | . | .| . | . | .|


-**computing_mean_spread_l2_regularization.py:** generates a dataset to evaluate  L2-regularization sweeping over different penalty values for a fixed m and epsilon. 

+ **output structure**: 

| k | penalty  | Ixs_mu | Ixs_sd | Ixs_n |
| --- | ---- | --- | ---- | --- |
| 4 | 0.1 | 120 | 20 | 10 |
| . | . | . | . | .|


+ **auxiliary script:**

  + **dp-models.py:** contains all dp algorithms and functions to generate all additional parameters needed in the simulations (e.g., matrix C used in algortihm 5)


### Visualization

+ **figure 1**: R script to generate a pdf file for figure 1 using outputs for computing_mean_spread_size.py setting *dataset_id = 'email-Eu-core'*. 

+ **figure 2**: R script to generate a pdf file for figure 2 using outputs for computing_mean_spread_size.py setting *dataset_id = 'erdos_renyi'*.



