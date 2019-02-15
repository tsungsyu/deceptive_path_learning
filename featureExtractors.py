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

    foods = state.getFood().asList()
    for goal in foods:
      dist = distanceToNearest((next_x, next_y), goal, walls)
      features[goal] = float(dist) / (walls.width * walls.height)
    # features["x"] = next_x
    # features["y"] = next_y


    # Divide values in order to prevent unstable divergence
    features.divideAll(10.0)
    features["bias"] = 1.0
    return features

def prob2Value(state, probability4Goals):
  '''
  Calculate additional reward based on the probabilities by Gaussian distribution
  diffProb = maximum dummy goal probability - true goal probability
  if diffProb = miu: reward is maximum
  '''
  value = 0.0
  if len(probability4Goals) == 0:
    return value

  truGoal = state.getTrueGoal()
  # if real goal is ate, return 0
  if state.getPacmanPosition() == truGoal:
    return value

  probOfTrueGoal = probability4Goals[truGoal]
  probDiffOfDummyGoals = {key:prob for key, prob in probability4Goals.items() if key != truGoal}

  # if all dummy goals are ate, return 0
  if len(probDiffOfDummyGoals) == 0:
    return value

  dists = dict()
  sigma = 1
  mu = 0
  scaleup = state.getWalls().width * state.getWalls().height
  for goal, prob in probDiffOfDummyGoals.items():
    # dists[goal] = calByGaussianDist(sigma, mu, probOfTrueGoal, prob)
    # dists[goal] = calByComparison(probOfTrueGoal, prob, mu)
    dists[goal] = calByEntropy(probOfTrueGoal, prob, mu)
  value = scaleup * max(dists.values())
  # print value
  return value

def calByEntropy(goalProbs, dummyProb, mu):
  entropy = -1 * (goalProbs * math.log(goalProbs, 2) + dummyProb * math.log(dummyProb, 2))
  if dummyProb - goalProbs < mu:
    return -1
  else:
    return entropy

def calByGaussianDist(sigma, mu, goalProbs, dummyProb):
  return 1 / (sigma * math.sqrt(math.pi * 2)) * math.exp(-1 * (dummyProb - goalProbs - mu)**2 / 2 * sigma)

def calByComparison(goalProbs,dummyProb, mu):
  if 0 <= dummyProb - goalProbs <= mu:
    return 1
  else:
    return -1

def calculateProbs(state):
  walls = state.getWalls()
  curPos = state.getPacmanPosition()
  startPos = state.data.agentStartPos
  stepsSoFar = state.getStepsSoFar()

  probability4Goals = dict()
  goals = state.data.food.asList()
  for goal in goals:
    probability4Goals[goal] = calculateProbByCostDiff(startPos, curPos, goal, walls)
  return probability4Goals

def calculateProbByCostDiff(startPos, curPos, goal, walls):
  if curPos != goal:
    distFromCurrentPos = distanceToNearest(curPos, goal, walls)
  else:
    distFromCurrentPos = 0

  distFromStartPos = distanceToNearest(startPos, goal, walls)
  distToCurrentPos = distanceToNearest(startPos, curPos, walls)
  costDiff = distFromCurrentPos + distToCurrentPos - distFromStartPos
  prob = math.exp(-1 * float(costDiff) / (walls.width + walls.height))
  return prob