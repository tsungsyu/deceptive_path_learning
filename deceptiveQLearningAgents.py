# deceptiveQLearningAgents.py
# ------------
# This file is a modified version of the qLearningAgents.py file written for
# the Pacman AI projects developed at UC Berkeley, which were developed primarily
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info on these projects, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# This file was adapted by Alexander Gunner (agunner@student.unimelb.edu.au, student ID 357149),
# for the COMP90055 Research subject at the University of Melbourne, Jan-Feb 2019.

from game import *
from deceptiveLearningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):


    def __init__(self, **args):
        "You can initialize Q-values here..."
        print "**args: ", args
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()
        self.maxQValuePolicy = False
        # State that is judged to be closest to the true goal, out of all the states that are equidistant from the
        # true goal and dummy goal. The Q value stored is for the action that takes the agent towards the true goal.
        self.lastDeceptivePoint = None
        # Actions that tend to lead the agent to the same state are excluded
        self.forbiddenActions = []
        # Record of the previous action taken (to prevent immediate backtracking)
        self.previousAction = None
        # Number of steps taken
        self.stepCount = 0
        # Agent's aim for this episode
        self.goalMode = 'deceptive'

    def isInTraining(self):
        """
        Returns Boolean stating whether the agent is in its second training phase.
        """

        return self.episodesSoFar < (self.phaseOneEpisodes + self.phaseTwoEpisodes)

    def isBacktrackAction(self, action):
        """
        Determines whether the proposed action would constitute a 180-degree reversal in direction
        """

        if self.previousAction == 'north':
            return action == 'south'
        elif self.previousAction == 'south':
            return action == 'north'
        elif self.previousAction == 'west':
            return action == 'east'
        elif self.previousAction == 'east':
            return action == 'west'
        else:
            return False


    def isEquilibriumState(self, state):
        """
        Determines whether a given state has at least one positively valued action
        and at least one negatively valued action
        """

        negativeActions = 0
        positiveActions = 0

        for action in self.getLegalActions(state):
            if self.getQValue(state, action) > 0:
                positiveActions += 1
            elif self.getQValue(state, action) < 0:
                negativeActions += 1

        return negativeActions and positiveActions


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]


    def calcMeanQValue(self, state):
        """
        Calculates the average of the Q values over all actions that can be taken from a given state
        """

        sumQValues = 0.0
        possibleActions = self.getLegalActions(state)
        numActions = 0

        for action in possibleActions:
            if (state, action) not in self.forbiddenActions:
                sumQValues += self.getQValue(state, action)
                numActions += 1

        if numActions == 0:
            return None

        else:
            return sumQValues / numActions


    def getValue(self, state):
        """
          Returns the value of a given state
          (in this case the mean of that state's Q values)

          If there are no legal actions (e.g. in a terminal state),
          then this function should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # If there are no legal actions, return 0.0.
        possibleActions = self.getLegalActions(state)
        if len(possibleActions) == 0:
            return 0.0

        return self.calcMeanQValue(state)


    def getAction(self, state):
        """
          Compute the action to take in the current state.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        action = None
        legalActions = self.getLegalActions(state)

        # List of actions, excluding those that are known to simply keep the agent in the same state
        possibleActions = []

        # List of actions and their respective Q values
        possibleActionQValues = util.Counter()

        # List of actions and the absolute values of their respective Q values,
        # excluding actions that would cause the agent to backtrack to the previous state
        possibleActionsNoBacktrack = util.Counter()
        absPossibleActionsNoBacktrack = util.Counter()

        # Assemble lists of actions that are permitted depending on the circumstances
        for action in legalActions:

            if (state, action) not in self.forbiddenActions:
                possibleActionQValues[action] = self.getQValue(state, action)
                possibleActions.append(action)

                if not self.isBacktrackAction(action):
                    possibleActionsNoBacktrack[action] = self.getQValue(state, action)
                    absPossibleActionsNoBacktrack[action] = abs(self.getQValue(state, action))

        if len(possibleActionQValues) > 0:

            print "goalMode: ", self.goalMode
            print "epsilon 1: ", self.epsilon1
            print "epsilon 2: ", self.epsilon2
            print "meanQValue: ", self.getValue(state)
            print "possibleActions: ", possibleActions
            print "possibleActionQValues: ", possibleActionQValues

            # Training to populate Q table
            if self.goalMode == 'maxQMode':
                if util.flipCoin(0.5):
                    # action = possibleActionsNoBacktrack.argMax()
                    action = possibleActionQValues.argMax()
                    # action = random.choice(possibleActions)
                else:
                    action = random.choice(possibleActions)

            elif self.goalMode == 'minQMode':
                if util.flipCoin(self.epsilon1):
                    # action = possibleActionsNoBacktrack.argMin()
                    action = possibleActionQValues.argMin()
                    # action = random.choice(possibleActions)
                else:
                    action = random.choice(possibleActions)
                    print "Random: ", action

            # Training to find deceptive path
            else:

                largestQValue = possibleActionQValues.argMax()

                print "Equilibrium state: ", state, self.isEquilibriumState(state)

                # If agent has already found an equidistant state with the largest-Q-value action seen so far,
                # then continue to the true goal
                if self.maxQValuePolicy:
                    action = possibleActionsNoBacktrack.argMax()

                # Otherwise, keep searching for the equidistant state that has the largest-Q-value action.
                else:

                    # If the agent has arrived at (what was thought to be) the LDP, and found that this state
                    # no longer has at least one positively valued action and at least one negatively valued action,
                    # then forget about this state.
                    if self.lastDeceptivePoint is not None and state == self.lastDeceptivePoint[0] and not self.isEquilibriumState(state):
                        self.lastDeceptivePoint = None

                    # If an equidistant state has been found...
                    if self.isEquilibriumState(state):

                        # If the agent has arrived at an equidistant state that has the largest-Q-value action
                        # seen so far (or if the agent has arrived at what is currently thought to be the LDP),
                        # then update the details of the likeliest LDP candidate...
                        if self.lastDeceptivePoint is None\
                                or possibleActionQValues.get(largestQValue) > self.lastDeceptivePoint[1]\
                                or state == self.lastDeceptivePoint[0]:

                            self.lastDeceptivePoint = (state, possibleActionQValues.get(largestQValue))

                            # Now head directly to the true goal, with probability epsilon2...
                            if util.flipCoin(1-self.epsilon2):
                                self.maxQValuePolicy = True
                                action = possibleActionsNoBacktrack.argMax()

                            # Or continue searching for equidistant states that might have a larger Q value
                            else:
                                action = absPossibleActionsNoBacktrack.argMin()

                            if self.epsilon2 >= 1.0/float(self.phaseTwoEpisodes):
                                self.epsilon2 -= 1.0/float(self.phaseTwoEpisodes)

                        # If this equidistant state does NOT have the largest-Q-value action
                        # of all equidistant states seen so far, then keep searching for such an equidistant state
                        else:
                            action = absPossibleActionsNoBacktrack.argMin()

                    # Otherwise, keep searching for an equidistant state:
                    else:

                        if self.getValue(state) > 0:
                            action = possibleActionsNoBacktrack.argMin()
                        elif self.getValue(state) < 0:
                            action = possibleActionsNoBacktrack.argMax()
                        else:
                            action = absPossibleActionsNoBacktrack.argMin()

        print "self.lastDeceptivePoint: ", self.lastDeceptivePoint

        return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        if state != nextState and self.isInTraining():
            self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (
                    reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))

        # Don't repeat actions that merely keep the agent in the same state
        if state == nextState and (state, action) not in self.forbiddenActions:
            self.forbiddenActions.append((state, action))

        self.previousAction = action
        self.stepCount += 1

        print "EPISODE ", self.episodesSoFar


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon1=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon1=0.1

        alpha    - learning rate
        epsilon1  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon1'] = epsilon1
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    # DeceptivePlanerExtractor
    # IdentityExtractor
    def __init__(self, extractor='DeceptivePlannerExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qValue = 0.0
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            qValue += (self.weights[key] * features[key])
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            self.weights[key] += self.alpha * (
                        reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)) * features[
                                     key]
            print key, " weight: ", self.weights[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
