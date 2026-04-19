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
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        foodList = newFood.asList()
        if foodList:
            minFoodlist = min(manhattanDistance(newPos, food)
                              for food in foodList)
            score += 10.0 / minFoodlist
        else:
            score += 500

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostpos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostpos)

            if scaredTime > 0:
                score += 200.0 / (dist + 1)
            else:
                if dist <= 1:
                    score -= 100
                elif dist <= 3:
                    score -= 100.0 / dist
        if action == Directions.STOP:
            score -= 50

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        # util.raiseNotDefined()
        numAgents = gameState.getNumAgents()

        def minimax(state, agentIndex, depth):
            # Terminaalconditie: gewonnen, verloren, of maximale diepte bereikt
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            successorScores = [
                minimax(state.generateSuccessor(
                    agentIndex, action), nextAgent, nextDepth)
                for action in legalActions
            ]

            # Pacman = max, geesten = min
            if agentIndex == 0:
                return max(successorScores)
            else:
                return min(successorScores)

        # Kies de actie met de hoogste minimax-waarde voor Pacman
        bestAction = max(
            gameState.getLegalActions(0),
            key=lambda action: minimax(
                gameState.generateSuccessor(0, action), 1, 0
            )
        )
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        
        numAgents = gameState.getNumAgents()

        def alphaBeta(state, agentIndex, depth, alpha, beta):
            # Terminaalconditie
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # Pacman = maximizer
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(successor, nextAgent, nextDepth, alpha, beta))
                    if value > beta:  # Pruning (strict, niet >=)
                        return value
                    alpha = max(alpha, value)
                return value

            else:  # Geest = minimizer
                value = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(successor, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:  # Pruning (strict, niet <=)
                        return value
                    beta = min(beta, value)
                return value

        # Root: kies beste actie voor Pacman met alpha-beta
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphaBeta(successor, 1, 0, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        numAgents = gameState.getNumAgents()

        def expectimax(state, agentIndex, depth):
            # Terminaalconditie
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            successorScores = [
                expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                for action in legalActions
            ]

            if agentIndex == 0:  # Pacman = maximizer
                return max(successorScores)
            else:  # Geest = kans-node: uniform gemiddelde
                return sum(successorScores) / len(successorScores)

        # Kies de actie met de hoogste expectimax-waarde voor Pacman
        bestAction = max(
            gameState.getLegalActions(0),
            key=lambda action: expectimax(
                gameState.generateSuccessor(0, action), 1, 0
            )
        )
        return bestAction
        
        


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # 1. Voedsel: reciproque van dichtstbijzijnde voedsel + straf voor resterende voedsel
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    if foodDistances:
        score += 10.0 / min(foodDistances)
    score -= 4 * len(foodList)

    # 2. Capsules: straf voor resterende capsules (aanmoedigen om op te eten)
    score -= 20 * len(capsules)
    if capsules:
        score += 5.0 / min(manhattanDistance(pacmanPos, cap) for cap in capsules)

    # 3. Geesten: gevaar of kans afhankelijk van scaredTimer
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(pacmanPos, ghostPos)
        scaredTime = ghostState.scaredTimer

        if scaredTime > 0:
            # Scared ghost: achtervolgen loont (bonuspunten)
            score += 200.0 / (dist + 1)
        else:
            # Actieve geest: harde straf als te dichtbij
            if dist <= 1:
                score -= 500
            elif dist <= 3:
                score -= 150.0 / dist

    return score


# Abbreviation
better = betterEvaluationFunction
