# deceptive_path_learning
to test the Q-learning with an observer approach:
  use the command "python pacman.py -p PacmanQAgent -n 5005 -l TestGrid16 -a numTraining=5000 -x 5000"
  This command would initiate a Q-learning agent to learn deceptive path on the Test Gird with 5000 training episodes and 5 testing episodes.
to test on the other test grid 1-15
  use the command "python pacman.py -p PacmanQAgent -n 5005 -l TestGrid[1-15] -a numTraining=5000 -x 5000"
  Note that to test in other grid needs to provide the position of real goal, please follow the instruction in the method "chooseTrueGoal"
