# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)

    # Q table
    self.qValues = util.Counter()
    self.qTable = util.Counter()

    print "ALPHA", self.alpha
    print "DISCOUNT", self.discount
    print "EXPLORATION", self.epsilon

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
    return self.qValues[(state, action)]


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    possibleStateQValues = util.Counter()
    for action in self.getLegalActions(state):
    	possibleStateQValues[action] = self.getQValue(state, action)
    return possibleStateQValues[possibleStateQValues.argMax()]

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    possibleStateQValues = util.Counter()
    possibleActions = self.getLegalActions(state)
    choosedAction = None
    if len(possibleActions) == 0:
    	return choosedAction
    summation = 0.0
    for action in possibleActions:
      qvalue = self.getQValue(state, action)
      summation += qvalue
      possibleStateQValues[action] = qvalue

    # print "current position (%s, %s)" % (state.getPacmanPosition()[0], state.getPacmanPosition()[1])
    # print possibleStateQValues
    # print summation / len(possibleStateQValues)

    if possibleStateQValues.totalCount() == 0:
      choosedAction = random.choice(possibleActions)
    else:
      choosedAction = possibleStateQValues.argMax()
    return choosedAction

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    if len(legalActions) > 0:
    	if util.flipCoin(self.epsilon):
    		action = random.choice(legalActions)
    	else:
			action = self.getPolicy(state)

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    # print "State: ", state, " , Action: ", action, " , NextState: ", nextState, " , Reward: ", reward
    # print "QVALUE", self.getQValue(state, action)
    # print "VALUE", self.getValue(nextState)
    qValue = self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
    self.qValues[(state, action)] = qValue
    self.qTable[(state.getPacmanPosition(), action)] = qValue

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
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
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    self.stateStack.append(state)
    if self.episodesSoFar >= 500:
      reward += self.rewardShaping(state, nextState)

    qValue = self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
    self.qValues[(state, action)] = qValue
    self.qTable[(state.getPacmanPosition(), action)] = qValue

  def rewardShaping(self, state, nextState):
    return self.discount * prob2Value(nextState, calculateProbs(nextState)) \
           - prob2Value(state, calculateProbs(state))

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
    # weight of agent
    self.weights = util.Counter()
    # weight of max
    self.weightsMean = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    qValue = 0.0
    features = self.featExtractor.getFeatures(state, action)
    for key in features.keys():
    	qValue += (self.weights[key] * features[key])
    return qValue

  def update(self, state, action, nextState, reward):
    """
       Should update weights based on transition
       the reward is passed in by interacting with the environment: learingAgents.py line 200
    """
    # update observer's weight
    # print "Episodes So Far: %d" % self.episodesSoFar
    if state not in self.stateStack:
      self.stateStack.append(state)
    features = self.featExtractor.getFeatures(state, action)
    # update reward of agent respect to the reward of observer
    for key in features.keys():
      self.weights[key] += self.alpha * (
                reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)) * features[key]

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      print "--TRAINING VARIABLES--"
      print state.data.__dict__
      pass

