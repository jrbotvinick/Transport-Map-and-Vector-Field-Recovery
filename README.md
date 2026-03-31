# Unique-Recovery-of-Transport-Maps-and-Vector-Fields

This repository contains Python code for the numerical experiments in the paper "On the Unique Recovery of Transport Maps and Vector Fields from Finite Measure-Valued Data" by Jonah Botvinick-Greenhouse (Cornell University) and Yunan Yang (Cornell University). 

Given a finite collection of probability measures $\rho_1,\dots,\rho_m \in \mathcal{P}(M)$, this code recovers the map $f$ from its pushforward action $\rho_j \mapsto f_{*}\rho_j$ and the vector field $v$ from its weighted divergence operation $\rho_j \mapsto \textup{div} (\rho_j v)$ on the collection of measures. In particular, the unknown map $f$ or vector field $v$ is parameterized as a neural network and objective functions comparing simulated pushforward / divergence operators with the observed datasets are minimized. Additional details about the objective functions, learning framework, and theoretical guarantees can be found in the paper. 

The files are organized as follows:

- `1D_transport_map_recovery.py`: Uniquely recovers a 1D pushforward map from its action on 5 densities.
- `1D_test_errors.py`: Computes the error statistics over ten trials run in `1D_transport_map_recovery.py`.
- `Lorenz_test.py`: Recovers the Lorenz system from the iterated pushforward action of its flow on a fixed Gaussian initial condition.
- `plot_lorenz.py`: Plots the Lorenz results and computes error statistics over 10 trials.
- `divergence_recovery.py`: Recovers a 2D vector field from its weighted divergence operator on a finite collection of densities.
- `plot_divergence_recovery.py`: Plots the results of `divergence_recovery.py` and examines reconstruction error as a function of the number of densities.

The following plots show a marginal of distributional snapshot data for learning the Lorenz-63 dynamics and the result of fitting a model to the data snapshots. They can be reproduced by running `Lorenz_test.py` and `plot_lorenz.py`. 

<img width="6492" height="1010" alt="lorenz_1a" src="https://github.com/user-attachments/assets/258d89e6-20bd-451a-8e0a-a2b18550baca" />
<img width="1500" height="1200" alt="lorenz_1b" src="https://github.com/user-attachments/assets/8a188b14-7b7f-4230-ac9f-d8aa7358e5ea" />
