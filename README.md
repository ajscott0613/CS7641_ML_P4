# CS7641_ML_P4
CS7641 Machine Learning Project 4

This repo can be cloned by running "git clone https://github.com/ajscott0613/CS7641_ML_P4.git"

The requirements.txt file can be installed via the command line by running "pip install -r requirments" and contains the necessary packages and version to run the python files.

Each of the files noted in the instructions section can be ran via the command line and requirement specific command line inputs to run them.

## File Contents

agents.py: This file contains the Value, Policy, and Q-learning algorithms needed to solve the frozen lake MDP.

runExperiments.py:  This file contains several differnet experiments and generates pltos and tables for the Frozen Lake MDP problem.

forestAgents.py: This file contains the Value, Policy, and Q-learning algorithms needed to solve the Forest Management MDP.

runExperimentForest.py:  This file contains several differnet experiments and generates pltos and tables for the Forest Management MDP problem.

## Instructions

runExperiments.py can be ran from the command line by running <python3 runExperiments.py "args">  
The following arguments can be taken  
-exp vipi gamma, this generates plots for value and policy iteration vs discount factor gamma  
-exp vipi size, this generates plots for value and policy iteration vs MDP size  
-exp ql all, this generates plots for differnet learning rates/gamma value for Q-learning  
-exp ql optimal, this generates average score plot for the optimal Q-Learning aglorithm  

runExperiments.py can be ran from the command line by running <python3 runExperiments.py "args">  
The following arguments can be taken   
-exp vipi gamma, this generates plots for value and policy iteration vs discount factor gamma  
-exp vipi size, this generates plots for value and policy iteration vs MDP size  
-exp ql, this generates plots for differnet learning rates/gamma value for Q-learning  

