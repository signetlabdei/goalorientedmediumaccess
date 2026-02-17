# goal_oriented_medium_access
Simulation code for the paper "A Theory of Goal-Oriented Medium Access: Protocol Design and Distributed Bandit Learning," by Federico Chiariotti and Andrea Zanella

## Abstract
> The Goal-oriented Communication (GoC) paradigm breaks the separation between communication and the content of the data, tailoring communication decisions to the specific needs of the receiver and targeting application performance. While recent studies show impressive encoding performance in point-to-point scenarios, the multi-node distributed scenario is still almost unexplored. Moreover, the few studies to investigate this consider a centralized collision-free approach, where a central scheduler decides the transmission order of the nodes. In this work, we address the Goal-oriented Multiple Access (GoMA) problem, in which multiple intelligent agents must coordinate to share a wireless channel and avoid mutual interference. We propose a theoretical framework for the analysis and optimization of distributed GoMA, serving as a first step towards its complete characterization. We prove that the problem is non-convex and may admit multiple Nash Equilibrium (NE) solutions. We provide a characterization of each node's best response to others' strategies and propose an optimization approach that provably reaches one such NE, outperforming centralized approaches by up to 100% while also reducing energy consumption. We also design a distributed learning algorithm that operates with limited feedback and no prior knowledge. 

The paper is submitted to [IEEE INFOCOM 2026](https://infocom2026.ieee-infocom.org/group/81).
A preprint version is [arXiv](https://arxiv.org/abs/2508.19141).

The main performance results are obtainable by running the scripts having ``sensors_`` at the beginning of the name, so, e.g.,
```
sensors_symm.m
sensors_asymm.m
```
will perform the simulation varying the available resources for the symmetric and asymmetric scenario.
