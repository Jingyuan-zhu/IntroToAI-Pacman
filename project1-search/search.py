# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

# define a class that 
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class Node:

    def __init__(self, state, direction, cost, parent):
        self.state = state
        self.direction = direction
        self.cost = cost 
        self.parent = parent 

    def __str__(self):
        return "[" + str(self.state) + "]"


    def root(self):
        if self.parent is None:
            return True
        else:
            return False

    def getParent(self):
        return self.parent

    def getState(self):
        return self.state

    def getDirection(self):
        return self.direction

    def getCost(self):
        return self.cost 

    def getTotalCost(self):
        totalCost = 0
        current = self
        while not current.root():
           totalCost += current.getCost()
           current = current.getParent()
        return totalCost

    def getPath(self):
        path = [] 
        current = self
        while not current.root():
            path.append(current.getDirection())
            current = current.getParent()
        path.reverse()
        return path


def generalSearch(problem, queuing):
# initialize the expanded nodes
    expanded = []
# push the startState Node into the queuing strategy defined by specific algorithm
    queuing.push(Node(state=problem.getStartState(), direction=None, cost=None, parent=None))
# while the frontier is not empty
    while True:
# first define current Node (use current state as a representation of the current Node)
        current_node = queuing.pop()

# if the state is the valid current state
        if problem.isGoalState(current_node.getState()):
            return current_node.getPath()

        if current_node.getState() not in expanded:
            
            for state, direction, cost in problem.getSuccessors(current_node.getState()):
                queuing.push(Node(state,direction,cost,current_node))

            expanded.append(current_node.getState())

# stop if the frontier is empty 
        if queuing.isEmpty():
            return None



def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    return generalSearch(problem, util.Stack())
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return generalSearch(problem, util.Queue())
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return generalSearch(problem, util.PriorityQueueWithFunction(lambda node: node.getTotalCost()))
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return generalSearch(problem, util.PriorityQueueWithFunction(lambda node: node.getTotalCost() + heuristic(node.getState(), problem)))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
