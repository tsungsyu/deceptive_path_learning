# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math
import random
class FeatureExtractor:
  def getFeatures(self, state, action):
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

def closestFood(pos, food, walls):
  """
  closestFood -- this is similar to the function that we have
  worked on in the search project; here its all in one place
  """
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

class SimpleExtractor(FeatureExtractor):
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """

  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0

    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height)
    # features.divideAll(10.0)
    return features


class DeceptivePlannerExtractor(FeatureExtractor):
  """
  Returns features for an agent that seeks one of several candidate goals while aiming to hide its intention:
  - Distance to the last deceptive point (LDP), while it has not been reached
  - Distance to the intended goal, once the LDP has been reached
  """

  def getFeatures(self, state, action):
    # Extract the grid of wall locations and initialise the counter of features
    walls = state.getWalls()
    features = util.Counter()

    # Compute the location of pacman after he takes the next action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    foods = state.getFood().asList()
    try:
      next_state = state.generateSuccessor(0, action)
      for goal in foods:
        dist = distanceRandomSimulation(next_state, goal)
        features[goal] = float(dist) / (walls.width * walls.height)
    except:
      features[state.getTrueGoal()] = 0
    features.divideAll(10.0)

    features["bias"] = 1.0
    return features

  def calculateHeatMap(self, state):

    pos = state.getPacmanPosition()
    stepsSoFar = state.getStepsSoFar()
    walls = state.getWalls()

    probability4Goals = dict()
    goals = state.data.food.asList()

    for goal in goals:
      distFromCurrentPos = distanceRandomSimulation(state, goal)
      distFromStartPos = stepsSoFar
      distToCurrentPos = distanceRandomSimulation(state, state.data.agentStartPos)
      costDiff = distFromCurrentPos + distToCurrentPos - distFromStartPos -1
      probability4Goals[goal] = math.exp(-1 * float(costDiff) / (walls.width + walls.height))
      state.data.statePossibility = probability4Goals

def prob2Value(state, probability4Goals):
  '''
  Calculate additional reward based on the probabilities by Gaussian distribution
  diffProb = maximum dummy goal probability - true goal probability
  if diffProb = 0: reward is maximum
  if diffProb > 0: reward > 0
  if diffProb < 0: reward < 0
  '''
  value = 0.0
  truGoal = state.getTrueGoal()

  probOfTrueGoal = probability4Goals[truGoal]
  probDiffOfDummyGoals = {key:abs(prob - probOfTrueGoal) for key, prob in probability4Goals.items()
                      if key != truGoal}

  # if all dummy goals are ate, return 0
  if len(probDiffOfDummyGoals) == 0:
    return value

  minProbDiffOfDummyGoal = min(probDiffOfDummyGoals.values())
  variance = 1
  miu = 0
  scaleup = 10
  value = scaleup * 1 / (variance * math.sqrt(math.pi * 2)) * math.exp(-1 * (minProbDiffOfDummyGoal-miu)**2 / 2 * variance)
  return value

def calculateProbs(state):
  pos = state.getPacmanPosition()
  stepsSoFar = state.getStepsSoFar()
  walls = state.getWalls()

  probability4Goals = dict()
  goals = state.data.food.asList()

  for goal in goals:
    distFromCurrentPos = distanceRandomSimulation(state, goal)
    distFromStartPos = stepsSoFar
    distToCurrentPos = distanceRandomSimulation(state, state.data.agentStartPos)
    costDiff = distFromCurrentPos + distToCurrentPos - distFromStartPos
    probability4Goals[goal] = math.exp(-1 * float(costDiff) / (walls.width + walls.height))
  return probability4Goals

def distanceRandomSimulation(state, target):
    """
    random simulation until reach the target
    :param state: from this state
    :param target: position of target (x, y)
    :return:
    """
    new_state = state.deepCopy()
    walls = new_state.getWalls()
    trueGoal = state.getTrueGoal()
    actionsStack = []
    distance = 0
    reached = False
    while not reached:
      # Get valid actions
      x, y = new_state.getPacmanPosition()
      # Get valid next state
      legalFutureStates = Actions.getLegalNeighbors((x, y), walls)
      # the true goal state counld not be generated
      if trueGoal != target and trueGoal in legalFutureStates:
        legalFutureStates.remove(trueGoal)
      # Get legal actions
      actions = new_state.getLegalActions(0)
      # The agent should not use the reverse direction during simulation
      current_direction = new_state.getPacmanState().configuration.direction
      reversed_direction = Directions.REVERSE[current_direction]
      if reversed_direction in actions and len(actions) > 1:
          actions.remove(reversed_direction)
      for a in actions:
        dx, dy = Actions.directionToVector(a)
        next_x, next_y = int(x + dx), int(y + dy)
        # next state must be a legal state
        if (next_x, next_y) not in legalFutureStates:
            actions.remove(a)
      # Randomly chooses a valid action
      action = random.choice(actions)
      # interact with environment and generate new state
      new_state = new_state.generateSuccessor(0, action)
      newPosition = new_state.getPacmanPosition()
      actionsStack.append(action)
      if newPosition == target:
        reached = True

    for action in actionsStack:
        dx, dy = Actions.directionToVector(action)
        distance = distance + dx + dy
    return distance

def calculateProbsOfNextState(state, action):
  x, y = state.getPacmanPosition()
  stepsSoFar = state.getStepsSoFar()
  probability4Goals = dict()
  walls = state.getWalls()
  goals = state.data.food.asList()

  try:
    next_state = state.generateSuccessor(0, action)
    for goal in goals:
      distFromCurrentPos = distanceRandomSimulation(next_state, goal)
      distFromStartPos = stepsSoFar
      distToCurrentPos = distanceRandomSimulation(next_state, state.data.agentStartPos)
      costDiff = distFromCurrentPos + distToCurrentPos - distFromStartPos
      probability4Goals[goal] = math.exp(-1 * float(costDiff) / (walls.width + walls.height))
  except:
    for goal in goals:
      if goal == state.getTrueGoal():
        probability4Goals[goal] = 1
      else:
        probability4Goals[goal] = 0
  return probability4Goals
