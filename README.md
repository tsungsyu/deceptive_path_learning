# deceptive_path_learning
This project was developed for the COMP90055 Computing Project subject, (Summer semester 2019) at the University of Melbourne.

Our work adapts the code developed by the [University of California, Berkeley](http://berkeley.edu/) for the [Pacman Projects](http://ai.berkeley.edu/project_overview.html).<br>

This repository includes two different approaches of reinforcement learning to generate deceptive paths in the 2-dimensional environment. Both the approaches are tested on the same 15 randomly generated 8-by-5 maps, each with one dummy goal and one true goal.

#### Approach 1: Mean Q table
```
python deceptiveGridWorld.py -g EvaluationGrid15 -q -o 800 -x 100
```
This runs the mean-Q-learner on EvaluationGrid15, quietly, with 800 phase-one training episodes, and 100 phase-two training episodes.

The agent was tested on EvaluationGrid1, EvaluationGrid2, ... EvaluationGrid15.

Line 529 of deceptiveGridWorld.py details additional options for parameters.


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
