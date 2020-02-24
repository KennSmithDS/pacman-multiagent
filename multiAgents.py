# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util
import logging, os, sys
import datetime as dt
from game import Agent
import traceback
import math

currentDt = dt.datetime.now().strftime('%m-%d-%Y-%H%M%S')
logPath = os.path.join(os.getcwd(), 'logs')

if not os.path.exists(logPath):
    os.mkdir(logPath)

fileName = os.path.join(logPath, f'assignment2__{currentDt}.log')
logging.basicConfig(filename=fileName,level=logging.DEBUG)

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self):
        self.foodStart = 0
        self.previousSpace = (0,0)
        self.agentLogPrefix = 'REFLEX AGENT:'

    def getAction(self, gameState):

        logging.info('-'*80)
        logging.info(f'{self.agentLogPrefix} NEW ACTION STATE')
        logging.info('-'*80)

        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        logging.info(f'{self.agentLogPrefix} Legal moves for Pac-man: {legalMoves}')

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        logging.info('-'*80)
        chosenPath = str(legalMoves[bestIndices[0]])
        logging.info(f'{self.agentLogPrefix} Scores for each move Pac-man can take: {scores}')
        logging.info(f'{self.agentLogPrefix} Path chosen: {chosenPath}')
        logging.info('-'*80)

        self.previousSpace = gameState.getPacmanPosition()
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def calcGridDistance(self, ax, ay, bx, by):
        return abs(ax-bx) + abs(ay-by)

    def calcCapsuleCost(self, capsulePositions, pacX, pacY, navSpace, longDist):
        capsulesLeft = len(capsulePositions)

        if capsulesLeft > 0:
            capsuleManQueue = util.PriorityQueue()
            sumOfManDist = 0

            for capsule in capsulePositions:
                capX, capY = capsule
                capsuleManDist = self.calcGridDistance(pacX, pacY, capX, capY)
                capsuleManQueue.push((capsule, capsuleManDist), capsuleManDist)
                sumOfManDist += capsuleManDist

            averageCapsuleDist = (sumOfManDist / len(capsulePositions))
            logging.info(f'{self.agentLogPrefix} Average distance to a capsule using Manhattan distance: {averageCapsuleDist}')
            closestCapsule, distToCapsule = capsuleManQueue.pop()
            logging.info(f'{self.agentLogPrefix} Capsule at {closestCapsule} is the closest to Pac-man using Manhattan distance {distToCapsule}')

            capsuleCost = (longDist-distToCapsule) * 10

            return closestCapsule, distToCapsule, capsuleCost
        
        return (0,0), 0, 0

    def calcFoodCost(self, foodLocations, pacX, pacY, longDist):
        foodLeft = len(foodLocations)
        foodStart = self.foodStart
        foodEaten = foodStart - foodLeft
        logging.info(f'{self.agentLogPrefix} Food start: {foodStart} | Food eaten: {foodEaten} | Food left: {foodLeft}')

        if foodLeft > 0:
            foodQueue = util.PriorityQueue()
            foodDistList = []

            for food in foodLocations:
                foodX, foodY = food
                foodDist = self.calcGridDistance(pacX, pacY, foodX, foodY)
                foodDistList.append(foodDist)
                foodQueue.push((food, foodDist), foodDist)

            totalDistCost = sum(foodDistList)
            averageDist = totalDistCost / foodLeft
            logging.info(f'{self.agentLogPrefix} Average distance to a food using Manhattan distance: {averageDist}')
            food, foodDist = foodQueue.pop()
            logging.info(f'{self.agentLogPrefix} Food at {food} is the closest to Pac-man using Manhattan distance {foodDist}')
            foodCost = ((1 / (foodDist)) * 100) + ((foodEaten/foodStart) * 500)
            return food, foodDist, foodCost
        else:
            return (pacX, pacY), 0, 999999

    def calcGhostCost(self, ghostPositions, pacX, pacY, longDist):

        while len(ghostPositions) > 0:
            ghostManQueue = util.PriorityQueue()
            sumOfManDist = 0

            ghostAlert = False

            for ghost in ghostPositions:
                ghostX, ghostY = ghost
                ghostManDist = self.calcGridDistance(pacX, pacY, ghostX, ghostY)
                ghostManQueue.push((ghost, ghostManDist), ghostManDist)
                sumOfManDist += ghostManDist

            averageGhostDist = (sumOfManDist / len(ghostPositions))
            logging.info(f'{self.agentLogPrefix} Average distance to a ghost using Manhattan distance: {averageGhostDist}')
            closestGhost, distToGhost = ghostManQueue.pop()
            logging.info(f'{self.agentLogPrefix} Ghost at {closestGhost} is the closest to Pac-man using Manhattan distance {distToGhost}')

            if distToGhost < 3:
                ghostAlert = True

            # need to work on ghost cost output, getting really skewed values, like -/+ 600000000
            if not ghostAlert:
                # as long as a ghost is 3 or more grid distance away, return the average manhattan distance
                logging.info(f'{self.agentLogPrefix} Ghost alert silent')
                return closestGhost, distToGhost, (1/distToGhost) * 10
            else:
                # otherwise return the distance to closest ghost
                logging.info(f'{self.agentLogPrefix} Ghost alert triggered')
                return closestGhost, distToGhost, (10 - distToGhost) * 100
                
        return (pacX, pacY), 0, 0

    def detectDeadEnd(self, currentGameState, pacX, pacY):
        legalMoves = currentGameState.getLegalActions()
        if 'Stop' in legalMoves:
            legalMoves.remove('Stop')
        logging.info(f'{self.agentLogPrefix} Pac-man has {len(legalMoves)} successor states')
        if len(legalMoves) <= 1:
            return True
        else:
            return False

    def setInitialFood(self, numFood):
        if numFood > self.foodStart:
            self.foodStart = numFood
        else:
            return

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        logging.info out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        "*** YOUR CODE HERE ***"

        baseScore = currentGameState.getScore()

        logging.info('-'*80)
        logging.info(f'{self.agentLogPrefix} NEW EVALUATION STATE: {action}')
        logging.info(f'{self.agentLogPrefix} CURRENT SCORE: {baseScore}')
        logging.info('-'*80)

        foodLeft = currentGameState.getNumFood()
        self.setInitialFood(foodLeft)
        logging.info(f'{self.agentLogPrefix} Food left on board: {foodLeft}')

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        gridWidth, gridHeight = currentGameState.data.layout.width, currentGameState.data.layout.height
        gridArea = (gridWidth-2) * (gridHeight-2)
        longestManDist = gridHeight + gridWidth - 2

        gridWalls = currentGameState.getWalls().asList()
        externalWalls = gridWidth*2 + (gridHeight-2)*2
        internalWalls = len(gridWalls) - externalWalls

        newPos = successorGameState.getPacmanPosition()
        pacX, pacY = newPos
        logging.info(f'{self.agentLogPrefix} New position: {newPos}')

        capsulePositions = currentGameState.getCapsules()

        gridFood = successorGameState.getFood().asList()

        if currentGameState.hasFood(pacX, pacY):
            logging.info(f'{self.agentLogPrefix} Food reward of +50 applied')
            rawScore = baseScore + 50
        else:
            rawScore = baseScore

        if action == 'Stop':
            logging.info(f'{self.agentLogPrefix} Stop penalty of -80 applied')
            rawScore -= 50

        if self.detectDeadEnd(successorGameState, pacX, pacY):
            logging.info(f'{self.agentLogPrefix} Deadend penalty of -500 applied')
            rawScore -= 1000

        if self.previousSpace == newPos:
            logging.info(f'{self.agentLogPrefix} Revisted space penatly of -50 applied')
            rawScore -= 50

        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = successorGameState.getGhostPositions()
        logging.info(f'{self.agentLogPrefix} Ghost positions: {newGhostPositions}')

        if newPos in newGhostPositions:
            rawScore -= 999999

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        logging.info(f'{self.agentLogPrefix} Scared timer: {newScaredTimes}')

        # function that takes positions of ghosts and pacman; RETURNS penalty for ghosts nearby
        nearestGhost, ghostDist, ghostWeight = self.calcGhostCost(newGhostPositions, pacX, pacY, longestManDist)
        logging.info(f'{self.agentLogPrefix} Ghost distance cost: {ghostWeight}')

        # final score for successor will be weighted by distance to ghosts, capsules, adjacent food and distant food
        scaredGhostsNearby = [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer > 15 and ghostDist < 10]
        logging.info(f'{self.agentLogPrefix} Scared ghosts nearby: {scaredGhostsNearby}')

        if nearestGhost in scaredGhostsNearby:
            rawScore += ghostWeight
        else:
            rawScore += -ghostWeight

        #  function defined above that takes positions of capsules, pacman, and navigable space; RETURNS reward for going to capsules
        nearestCapsule, capsuleDist, capsuleWeight = self.calcCapsuleCost(capsulePositions, pacX, pacY, gridArea-internalWalls, longestManDist)
        logging.info(f'{self.agentLogPrefix} Capusle distance cost: {capsuleWeight}')
        rawScore += capsuleWeight

        # function defined above that takes positions of food and pacman; RETURNS reward for going towards food
        nearestFood, foodDist, nearestFoodWeight = self.calcFoodCost(gridFood, pacX, pacY, longestManDist)
        logging.info(f'{self.agentLogPrefix} Food distance cost: {nearestFoodWeight}')
        rawScore += nearestFoodWeight

        logging.info(f'{self.agentLogPrefix} Final score for {action}: {rawScore}')
        return rawScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.agentLogPrefix = {0: 'MINIMAX AGENT:', 1: 'ALPHABETA AGENT:'}
        self.depthAlpha = {}
        self.depthBeta = {}

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        logging.info('-'*80)
        logging.info(f'{self.agentLogPrefix[0]} NEW GAME STATE')
        logging.info('-'*80)

        logging.info(f'{self.agentLogPrefix[0]} Number of agents in game state: {gameState.getNumAgents()}')
        legalActions = gameState.getLegalActions(0)
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        logging.info(f'{self.agentLogPrefix[0]} Legal actions from current game state: {legalActions}')

        bestAction = None
        maxValue = -math.inf

        for action in legalActions:
            newState = gameState.generateSuccessor(0, action)
            minimaxScore = self.minimaxRecursion(newState, 0, False, 1)

            if minimaxScore >  maxValue:
                maxValue = minimaxScore
                bestAction = action

        logging.info(f'{self.agentLogPrefix[0]} Pacman decided going {bestAction} has the best score of {maxValue}')

        logging.info('-'*80)
        logging.info(f'{self.agentLogPrefix[0]} ACTION CHOSEN: {bestAction}') # swap random with bestAction
        logging.info('-'*80)

        return bestAction # swap random with bestAction

        util.raiseNotDefined()

    def minimaxRecursion(self, gameState, depth, maximizerTurn, minimizerTurn):
        try:
        # check if recursion depth is 0 or if game is win/lose condition | terminal state
            if (depth == self.depth) or (gameState.isWin()) or (gameState.isLose()) or (len(gameState.getLegalActions(0)) == 0):
                terminalScore = self.evaluationFunction(gameState) # gameState.getScore()
                logging.info(f'{self.agentLogPrefix[0]} Score is {terminalScore} at terminal state')
                return terminalScore

            else:
                if maximizerTurn: # maximizing agent's turn
                    logging.info(f'{self.agentLogPrefix[0]} Maximizing agents turn at depth {depth}')
                    agent = 0
                    value = -math.inf
                    
                    legalActions = gameState.getLegalActions(agent)
                    if 'Stop' in legalActions:
                        legalActions.remove('Stop')

                    for action in legalActions: # iterate over each possible successor state / action
                        nextState = gameState.generateSuccessor(agent, action)
                        value = max(value, self.minimaxRecursion(nextState, depth, False, minimizerTurn+1)) # recursion returning NoneType
                    return value

                else: # minimizing agent's turn
                    agent = minimizerTurn
                    logging.info(f'{self.agentLogPrefix[0]} It is now ghost #{minimizerTurn} turn')
                    value = math.inf

                    legalActions = gameState.getLegalActions(agent)

                    if minimizerTurn < (gameState.getNumAgents() - 1):
                        logging.info(f'{self.agentLogPrefix[0]} Minimizing agent {minimizerTurn} turn at depth {depth}')
                        for action in legalActions: # iterate over each possible successor state / action
                            nextState = gameState.generateSuccessor(agent, action)
                            value = min(value, self.minimaxRecursion(nextState, depth, False, minimizerTurn+1)) # recursion returning NoneType
                        return value

                    else:
                        logging.info(f'{self.agentLogPrefix[0]} Last minimizing agent action for this turn at depth {depth}')
                        for action in legalActions: # iterate over each possible successor state / action
                            nextState = gameState.generateSuccessor(agent, action)
                            value = min(value, self.minimaxRecursion(nextState, depth+1, True, 0)) # recursion returning NoneType
                        return value

        # error handling block
        except Exception as e:
            logging.exception('-'*80)
            logging.exception(f'{self.agentLogPrefix[0]} - {traceback.print_exc(file=sys.stdout)} - {e}')
            logging.exception('-'*80)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        logging.info('-'*80)
        logging.info(f'{self.agentLogPrefix[1]} NEW GAME STATE')
        logging.info('-'*80)

        logging.info(f'{self.agentLogPrefix[1]} Number of agents in game state: {gameState.getNumAgents()}')
        legalActions = gameState.getLegalActions(0)
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        logging.info(f'{self.agentLogPrefix[1]} Legal actions from current game state: {legalActions}')

        bestAction = None
        bestValue = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in legalActions:
            newState = gameState.generateSuccessor(0, action)
            alphaBetaScore = self.alphaBetaRecursion(newState, 0, False, 1, alpha, beta)
            alpha = max(alphaBetaScore, alpha)

            if alphaBetaScore >  bestValue:
                bestValue = alphaBetaScore
                bestAction = action

        logging.info(f'{self.agentLogPrefix[1]} Pacman decided going {bestAction} has the best score of {bestValue}')

        logging.info('-'*80)
        logging.info(f'{self.agentLogPrefix[1]} ACTION CHOSEN: {bestAction}') # swap random with bestAction
        logging.info('-'*80)

        return bestAction # swap random with bestAction

        util.raiseNotDefined()

    def alphaBetaRecursion(self, gameState, depth, maximizerTurn, minimizerTurn, alpha, beta):
        try:
        # check if recursion depth is 0 or if game is win/lose condition | terminal state
            if (depth == self.depth) or (gameState.isWin()) or (gameState.isLose()) or (len(gameState.getLegalActions(0)) == 0):
                terminalScore = self.evaluationFunction(gameState) # gameState.getScore()
                logging.info(f'{self.agentLogPrefix[1]} Score is {terminalScore} at terminal state')
                return terminalScore

            else:
                # if depth not in self.depthAlpha:
                #     self.depthAlpha[depth] = -math.inf
                # if depth not in self.depthBeta:
                #     self.depthBeta[depth] = math.inf

                if maximizerTurn: # maximizing agent's turn
                    logging.info(f'{self.agentLogPrefix[1]} Maximizing agents turn at depth {depth}')
                    agent = 0
                    maxValue = -math.inf
                    
                    legalActions = gameState.getLegalActions(agent)
                    if 'Stop' in legalActions:
                        legalActions.remove('Stop')

                    for action in legalActions: # iterate over each possible successor state / action
                        logging.info(f'{self.agentLogPrefix[1]} Maximizing agent can take action to go {action}')
                        nextState = gameState.generateSuccessor(agent, action)
                        value = self.alphaBetaRecursion(nextState, depth, False, minimizerTurn+1, alpha, beta) # recursion returning NoneType
                        maxValue = max(maxValue, value)
                        alpha = max(alpha, value)
                        logging.info(f'{self.agentLogPrefix[1]} Alpha is equal to {alpha}')
                        logging.info(f'{self.agentLogPrefix[1]} Comparing maxs {alpha} to mins {beta}')
                        if beta < maxValue:
                            logging.info(f'{self.agentLogPrefix[1]} Branch pruned at depth {depth} for max {action}')
                            return maxValue
                    return maxValue

                # failed attempt at using class attribute to store dictionary of alpha and beta by depth
                    # for action in legalActions: # iterate over each possible successor state / action
                    #     nextState = gameState.generateSuccessor(agent, action)
                    #     value = max(self.depthAlpha[depth], self.alphaBetaRecursion(nextState, depth, False, minimizerTurn+1, self.depthAlpha[depth], self.depthBeta[depth])) # recursion returning NoneType
                    #     logging.info(f'{self.agentLogPrefix[1]} Comparing maxs {value} to mins {self.depthBeta[depth]}')
                    #     if value >= self.depthBeta[depth]:
                    #         return value
                    #     self.depthAlpha[depth] = max(self.depthAlpha[depth], value)                      
                    #     logging.info(f'{self.agentLogPrefix[1]} New alpha is equal to {self.depthAlpha[depth]}')
                    # return value

                else: # minimizing agent's turn
                    agent = minimizerTurn
                    logging.info(f'{self.agentLogPrefix[1]} Minimizing agent {minimizerTurn} turn at depth {depth}')
                    minValue = math.inf

                    legalActions = gameState.getLegalActions(agent)
                    logging.info(f'{self.agentLogPrefix[1]} Minimizing agent {minimizerTurn} can take these actions: {legalActions}')

                    if minimizerTurn < (gameState.getNumAgents() - 1):
                        for action in legalActions: # iterate over each possible successor state / action
                            logging.info(f'{self.agentLogPrefix[1]} Minimizing agent is considering going {action}')
                            nextState = gameState.generateSuccessor(agent, action)
                            value = self.alphaBetaRecursion(nextState, depth, False, minimizerTurn+1, alpha,beta) # recursion returning NoneType
                            minValue = min(value, minValue)
                            beta = min(value, beta)
                            logging.info(f'{self.agentLogPrefix[1]} Beta is equal to {beta}')
                            logging.info(f'{self.agentLogPrefix[1]} Comparing maxs {alpha} to mins {beta}')
                            if minValue < alpha:
                                logging.info(f'{self.agentLogPrefix[1]} Branch pruned at depth {depth} for min {action}')
                                return minValue
                        return minValue

                    else:
                        logging.info(f'{self.agentLogPrefix[1]} Last minimizing agent action for this turn at depth {depth}')
                        for action in legalActions: # iterate over each possible successor state / action
                            logging.info(f'{self.agentLogPrefix[1]} Minimizing agent is considering going {action}')
                            nextState = gameState.generateSuccessor(agent, action)
                            value = self.alphaBetaRecursion(nextState, depth+1, True, 0, alpha, beta) # recursion returning NoneType
                            minValue = min(value, minValue)
                            beta = min(value, beta)
                            logging.info(f'{self.agentLogPrefix[1]} Beta is equal to {beta}')
                            logging.info(f'{self.agentLogPrefix[1]} Comparing maxs {alpha} to mins {beta}')
                            if minValue < alpha:
                                logging.info(f'{self.agentLogPrefix[1]} Branch pruned at depth {depth} for min {action}')
                                return minValue
                        return minValue

                # failed attempt at using class attribute to store dictionary of alpha and beta by depth
                    # if minimizerTurn < (gameState.getNumAgents() - 1):
                    #     logging.info(f'{self.agentLogPrefix[1]} Minimizing agent {minimizerTurn} turn at depth {self.depthBeta[depth]}')
                    #     for action in legalActions: # iterate over each possible successor state / action
                    #         nextState = gameState.generateSuccessor(agent, action)
                    #         value = min(self.depthBeta[depth], self.alphaBetaRecursion(nextState, depth, False, minimizerTurn+1, self.depthAlpha[depth], self.depthBeta[depth])) # recursion returning NoneType
                    #         logging.info(f'{self.agentLogPrefix[1]} Comparing maxs {self.depthAlpha[depth]} to mins {value}')
                    #         if value <= self.depthAlpha[depth]:
                    #             return value
                    #         self.depthBeta[depth] = min(value, self.depthBeta[depth])
                    #         logging.info(f'{self.agentLogPrefix[1]} New beta is equal to {self.depthBeta[depth]}')
                    #     return value

                    # else:
                    #     logging.info(f'{self.agentLogPrefix[1]} Last minimizing agent action for this turn at depth {self.depthBeta[depth]}')
                    #     for action in legalActions: # iterate over each possible successor state / action
                    #         nextState = gameState.generateSuccessor(agent, action)
                    #         value = min(self.depthBeta[depth], self.alphaBetaRecursion(nextState, depth+1, True, 0, self.depthAlpha[depth], self.depthBeta[depth])) # recursion returning NoneType
                    #         logging.info(f'{self.agentLogPrefix[1]} Comparing maxs {self.depthAlpha[depth]} to mins {value}')
                    #         if value <= self.depthAlpha[depth]:
                    #             return value
                    #         self.depthBeta[depth] = min(value, self.depthBeta[depth])
                    #         logging.info(f'{self.agentLogPrefix[1]} New beta is equal to {self.depthBeta[depth]}')
                    #     return value

        # error handling block
        except Exception as e:
            logging.exception('-'*80)
            logging.exception(f'{self.agentLogPrefix[1]} - {traceback.print_exc(file=sys.stdout)} - {e}')
            logging.exception('-'*80)

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
