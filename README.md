# deceptive_path_learning
This project is developed and tested based on the [Pacman Projects](http://ai.berkeley.edu/project_overview.html) by the [University of California, Berkeley](http://berkeley.edu/).<br>

This repository includes two different approaches of reinforcement learning to generate deceptive paths in the 2-Dimention environment. Both the approaches tested on the same 15 randomly generated 8 times 5 maps, each with one dummy goal and one true goal.
#### Approach 1: Mean Q table


#### Approach 2: Reward-shaped Q-learning with an observer<br>
Checkout the branch with tag “Q-Learning_RewardShaping_Prob”
Execute the command as below:
```
python pacman.py -p PacmanQAgent -n 5005 -l TestGrid16 -a numTraining=5000 -x 5000
```
This command would initiate a Q-learning agent to learn deceptive path on the Test Gird with 5000 training episodes and 5 testing episodes.<br>
To test on other layouts (test grid 1-15), change the variable -l as below:
```
python pacman.py -p PacmanQAgent -n 5005 -l TestGrid[1-15] -a numTraining=5000 -x 5000
```
Note that to test in other grid needs to provide the position of real goal, please follow the instruction in the method "chooseTrueGoal"
