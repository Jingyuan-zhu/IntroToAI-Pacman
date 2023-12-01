# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
#import copy


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # V(s) = max_{a in actions} Q(s,a)
        # policy(s) = arg_max_{a in actions} Q(s,a)
        "*** YOUR CODE HERE ***"
        # this function intends to find the value of each state in iteration

        
        for i in range(self.iterations):
            tempValues = util.Counter()
            for state in self.mdp.getStates():
                q_sa = []

                for action in self.mdp.getPossibleActions(state):
                    q_sa.append((action, self.getQValue(state,action)))

                tempValues[state] = max(q_sa, key = lambda x: x[1])[1] if len(q_sa) else 0
                
            self.values = tempValues



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q = 0

        for nextState, transProb in self.mdp.getTransitionStatesAndProbs(state,action):
            q += transProb * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))

        return q
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        qa_pair = []
        for action in self.mdp.getPossibleActions(state):
            qa_pair.append((action, self.getQValue(state,action)))

        return max(qa_pair, key = lambda qa: qa[1])[0] if len(qa_pair) else None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        "*** YOUR CODE HERE ***"
        #-------------FROM Q1--------------------------------------------
        # for i in range(self.iterations):
        #     tempValues = util.Counter()
        #     for state in self.mdp.getStates():
        #         q_sa = []

        #         for action in self.mdp.getPossibleActions(state):
        #             q_sa.append((action, self.getQValue(state,action)))

        #         tempValues[state] = max(q_sa, key = lambda x: x[1])[1] if len(q_sa) else 0
                
        #     self.values = tempValues

        states = self.mdp.getStates()
        for i in range(self.iterations):
            # tempValues = util.Counter()
            this_state = states[i % len(states)]
            stateval = self.computeActionFromValues(this_state)
            #value compute action from value
            #check mdp is terminal
            if self.mdp.isTerminal(this_state):
                #print("end")
                continue
            #if not
            #value of state = self.computeQValueFromValues
            value = self.computeQValueFromValues(this_state, stateval)
            self.values[this_state] = value


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #When you compute predecessors of a state, make sure to store them 
        # in a set, not a list, to avoid duplicates.
        #Please use util.PriorityQueue in your implementation. 
        # The update method in this class will likely be useful;
        allstates = self.mdp.getStates()

        predecessors = {s: set() for s in allstates}
        for s in allstates:
            for action in self.mdp.getPossibleActions(s):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(s, action):
                    predecessors[nextState].add(s)

        #create empty pq
        pq = util.PriorityQueue()

        for state in allstates:
            if not self.mdp.isTerminal(state):
                qValues = [self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)]
                diff = abs(max(qValues) - self.values[state])
                pq.push(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, a) for a in self.mdp.getPossibleActions(s)])

            for p in predecessors[s]:
                if not self.mdp.isTerminal(p):
                    qValues = [self.getQValue(p, a) for a in self.mdp.getPossibleActions(p)]
                    diff = abs(max(qValues) - self.values[p])
                    if diff > self.theta:
                        pq.update(p, -diff)

