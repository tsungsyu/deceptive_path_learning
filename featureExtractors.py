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

def distanceToNearest(pos, targetType, walls):
  """
  Returns distance to the nearest item of the specified type (e.g. food, Power Capsule)

  # Open-list nodes consist of position x,y and distance (initialised at 0)
  fringe = [(pos[0], pos[1], 0)]

  # Closed list as a set (unordered list of unique elements)
  expanded = set()

  while fringe:
    # Pop latest node from open list, and add to closed list
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # Exit if the target item already exists at this location
    if pos_x == targetType[0] and pos_y == targetType[1]:
      return dist
    # Otherwise, investigate neighbouring nodes and add them to the open list
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))

  # If target item not found
  return None
  """
  if targetType is not None and pos is not None:
    return abs(targetType[0] - pos[0]) + abs(targetType[1] - pos[1])
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

    # First feature guides the agent to the last deceptive point (LDP)
    # if not state.reachedLdp():
    #   dist = distanceToNearest((next_x, next_y), state.getLdp(), walls)
    #   features["LDP-distance"] = float(dist) / (walls.width * walls.height)

    # Once the LDP has been reached, switch to the second feature, which guides agent to the goal
    # else:
    trueGoal = state.getTrueGoal()
    #
    dist = distanceToNearest((next_x, next_y), trueGoal, walls)
    features["true-goal-dist"] = dist/10

    # foods = state.getFood().asList()
    # for goal in foods:
    #   dist = distanceToNearest((next_x, next_y), goal, walls)
    #   features[goal] = float(dist) / (walls.width * walls.height)

    # features["x"] = x
    # features["y"] = y

    features["probDiff"] = prob2Value(state)

    # Divide values in order to prevent unstable divergence
    # features.divideAll(10.0)
    features["bias"] = 1.0
    return features

  def getObserverFeatures(self, state, agentAction):
    """
    extract features of observer
    mainly calculate the path completion from current node
    :param state:
    :param agentAction:
    :return:
    """
    # Extract the grid of wall locations and initialise the counter of features
    stepsSoFar = state.getStepsSoFar()
    walls = state.getWalls()
    observerFeatures = util.Counter()
    pos = state.getPacmanPosition()
    # foodList = state.data.food.asList()
    # for food in foodList:
    # TODO only when choosing the feature which are not observer's choice the Q value seems making sence
    # if food == agentAction:
    distFromCurrentPos = distanceToNearest(pos, agentAction, walls)

    # TODO to use probabilty as feature
    distFromStartPos = distanceToNearest(state.data.agentStartPos, agentAction, walls)
    costDiff = distFromCurrentPos + stepsSoFar - distFromStartPos

    observerFeatures[agentAction] = math.exp(-1 * float(costDiff) / (walls.width + walls.height))
    # Divide values in order to prevent unstable divergence

    return observerFeatures

  def calculateHeatMap(self, state):

    pos = state.getPacmanPosition()
    stepsSoFar = state.getStepsSoFar()
    walls = state.getWalls()

    probability4Goals = dict()
    goals = state.data.food.asList()

    for goal in goals:
      distFromCurrentPos = distanceToNearest(pos, goal, walls)
      distFromStartPos = distanceToNearest(state.data.agentStartPos, goal, walls)
      distToCurrentPos = distanceToNearest(state.data.agentStartPos, pos, walls)
      # costDiff = distFromCurrentPos + stepsSoFar - distFromStartPos -1
      costDiff = distFromCurrentPos + distToCurrentPos - distFromStartPos -1
      probability4Goals[goal] = math.exp(-1 * float(costDiff) / (walls.width + walls.height))
      state.data.statePossibility = probability4Goals

def prob2Value(state):
  '''
  Calculate additional reward based on the probabilities by Gaussian distribution
  diffProb = maximum dummy goal probability - true goal probability
  if diffProb = 0: reward is maximum
  if diffProb > 0: reward > 0
  if diffProb < 0: reward < 0
  '''
  value = 0.0
  probability4Goals = calculateProbs(state)
  probOfTrueGoal = probability4Goals[state.getTrueGoal()]
  probDiffOfDummyGoals = {key:prob - probOfTrueGoal for key, prob in probability4Goals.items()
                      if key != state.getTrueGoal()}

  # if all dummy goals are ate, return 0
  if len(probDiffOfDummyGoals) == 0:
    return value

  minProbDiffOfDummyGoal = min(probDiffOfDummyGoals.values())
  print "minProbDiffOfDummyGoal: ", minProbDiffOfDummyGoal
  variance = 1
  miu = 0
  scaleup = 10
  value = scaleup * 1 / (variance * math.sqrt(math.pi * 2)) * math.exp(-1 * (minProbDiffOfDummyGoal-miu)**2 / 2 * variance)
  # print "probOfTrueGoal: ", probOfTrueGoal
  # print "maxOfDummyGoal: ", maxOfDummyGoal
  if minProbDiffOfDummyGoal < miu:
    print "value: ", value
    return value * (-1)
  else:
    print "value: ", value
    return value

def calculateProbs(state):
  pos = state.getPacmanPosition()
  stepsSoFar = state.getStepsSoFar()
  walls = state.getWalls()

  probability4Goals = dict()
  goals = state.data.food.asList()

  for goal in goals:
    distFromCurrentPos = distanceToNearest(pos, goal, walls)
    distFromStartPos = distanceToNearest(state.data.agentStartPos, goal, walls)
    distToCurrentPos = distanceToNearest(state.data.agentStartPos, pos, walls)
    # costDiff = distFromCurrentPos + stepsSoFar - distFromStartPos - 1
    costDiff = distFromCurrentPos + distToCurrentPos - distFromStartPos - 1
    probability4Goals[goal] = math.exp(-1 * float(costDiff) / (walls.width + walls.height))

  print probability4Goals
  return probability4Goals