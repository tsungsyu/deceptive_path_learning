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

    "*** YOUR CODE HERE ***"
    self.qValues = util.Counter()
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
    "*** YOUR CODE HERE ***"
    possibleStateQValues = util.Counter()
    for action in self.getLegalActions(state):
    	possibleStateQValues[action] = self.getQValue(state, action)

    return possibleStateQValues[possibleStateQValues.argMax()]

  def getObserverValue(self, state):
    possibleStateQValues = util.Counter()
    for action in state.getFood().asList():
      possibleStateQValues[action] = self.getObserverQValue(state, action)
    return possibleStateQValues[possibleStateQValues.argMax()]

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    possibleStateQValues = util.Counter()
    possibleActions = self.getLegalActions(state)
    if len(possibleActions) == 0:
    	return None

    for action in possibleActions:
      possibleStateQValues[action] = self.getQValue(state, action)

    if possibleStateQValues.totalCount() == 0:
    	return random.choice(possibleActions)
    else:
    	return possibleStateQValues.argMax()

  def getObserverPolicy(self, state):

    possibleStateQValues = util.Counter()
    possibleActions = state.getFood().asList()
    if len(possibleActions) == 0:
      return None

    for (x, y) in possibleActions:
      possibleStateQValues[(x, y)] = self.getObserverQValue(state, (x, y))
    print "current Q values:"
    print possibleStateQValues
    if possibleStateQValues.totalCount() == 0:
      return random.choice(possibleActions)
    else:
      return possibleStateQValues.argMax()

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
    "*** YOUR CODE HERE ***"
    if len(legalActions) > 0:
    	if util.flipCoin(self.epsilon):
    		action = random.choice(legalActions)
    	else:
			action = self.getPolicy(state)

    return action

  def getObserverAction(self, state):
    # Pick Action
    possibleGoals = state.getFood().asList()
    action = []
    "*** YOUR CODE HERE ***"
    if len(possibleGoals) > 0:
      if util.flipCoin(self.epsilon):
        action.append(random.choice(possibleGoals))
      else:
        action.append(self.getObserverPolicy(state))
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
    print "State: ", state, " , Action: ", action, " , NextState: ", nextState, " , Reward: ", reward
    print "QVALUE", self.getQValue(state, action)
    print "VALUE", self.getValue(nextState)
    self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))

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
    self.observerWeight = util.Counter()

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

  def getObserverQValue(self, state, action):
    observersFeatures = self.featExtractor.getObserverFeatures(state, action)
    qValue = (self.observerWeight[action] * observersFeatures[action])
    # for key in observersFeatures.keys():
    #   # print "feature[%s], weight: %f, f(s,a): %f" % (key, self.observerWeight[key], observersFeatures[key])
    #
    #   print "%s q = %f" % (key,thisQ)
    #   qValue += thisQ
    return qValue

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    "*** YOUR CODE HERE ***"
    # Todo: calculate observerFeatures
    print "===============current position================="
    print state.getPacmanPosition()
    observerAction = self.getObserverAction(state)[0]
    print "observer choose: (%s,%s)" % (observerAction[0], observerAction[1])
    observerReward = self.observerDoAction(state, observerAction)

    observerFeatures = self.featExtractor.getObserverFeatures(state, observerAction)

    for key in observerFeatures.keys():
      # print "update::"
      # print "feature[%s], reward: %f, V(s',a'): %f, Q(s,a): %f, f(s,a): %f" % (key, observerReward,self.getObserverValue(nextState),self.getObserverQValue(state, observerAction),observerFeatures[key])
      self.observerWeight[key] += self.alpha * (observerReward + self.discount * self.getObserverValue(nextState) - self.getObserverQValue(state, observerAction)) * observerFeatures[key]

    features = self.featExtractor.getFeatures(state, action)
    # update reward to agent respect to the reward of observer
    reward += (-0.001 * observerReward)
    for key in features.keys():
      self.weights[key] += self.alpha * (
                reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)) * features[key]


  def observerPredictGoal(self, state, action):
    '''
    obserserver predict goal based on the path completion of each potential goal
    choose the one owing max path completion
    '''
    possiblePathComp = util.Counter()
    walls = state.getWalls()

    # Compute the location of pacman after he takes the next action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    foodList = state.getFood().asList()
    for food in foodList:
      distFromCurrentPos = distanceToNearest((next_x, next_y), food, walls)
      distFromStartPos = distanceToNearest(state.data.agentStartPos, food, walls)
      possiblePathComp[food] = distFromStartPos - distFromCurrentPos

    return possiblePathComp.argMax()

  def observerDoAction(self, state, observerAction):
    '''
    if observer predict correctly, get positive reward,
    else get negative reward
    '''
    # self.state.data.predictCount = self.state.data.predictCount + 1
    # print "--------------%s == %s--------------" % (observerAction, state.getTrueGoal())
    if observerAction == state.getTrueGoal():
      reward = 10.0
    else:
      reward = -10.0
    return reward

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
