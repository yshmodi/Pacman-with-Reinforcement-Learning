# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalue = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalue[state,action]
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0.0
        maxqvalue = -999999
        for action in legalActions:
            if self.getQValue(state,action) > maxqvalue:
                maxqvalue = self.getQValue(state,action)
        return maxqvalue  
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = [None]
        legalActions = self.getLegalActions(state)
        maxqvalue = -999999
        for action in legalActions:
            if self.getQValue(state,action) > maxqvalue:
                maxqvalue = self.getQValue(state,action)
                bestAction = [action]
            elif self.getQValue(state,action) == maxqvalue:
                bestAction.append(action)

        return random.choice(bestAction)
        #util.raiseNotDefined()

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

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)
        
        #util.raiseNotDefined()

        #return action

    def update(self, state, action, nextState, reward):
        next_x, next_y = nextState.getPacmanPosition()
        if (int(next_x), int(next_y)) in state.data.layout.tunnels:
            reward += 5  # Encourage tunnel entry
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        key = (state, action)
        self.qvalue[key] = (1.0 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.2,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
        legalActions = self.getLegalActions(state)
        x, y = state.getPacmanPosition()

        # Check if Pac-Man is at a tunnel exit (just teleported)
        is_at_tunnel_exit = (x, y) not in state.data.layout.tunnels and any(
            (int(x + dx), int(y + dy)) in state.data.layout.tunnels
            for action in legalActions
            for dx, dy in [Actions.directionToVector(action)]
        )

        # Force tunnel entry only if not at an exit and tunnel is beneficial
        if not is_at_tunnel_exit:
            for action in legalActions:
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(x + dx), int(y + dy)
                if (next_x, next_y) in state.data.layout.tunnels:
                    # Evaluate if tunnel leads to a better state (e.g., food proximity)
                    successor = state.generatePacmanSuccessor(action)
                    if successor.getNumFood() < state.getNumFood() or len(successor.getCapsules()) < len(state.getCapsules()):
                        print("Debug: Forcing beneficial tunnel move:", action)
                        self.doAction(state, action)
                        return action

        # Fall back to Q-learning
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
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getAction(self, state):
        legalActions = state.getLegalActions()
        q_values = [(action, self.getQValue(state, action)) for action in legalActions]
        print(f"State: {state.getPacmanPosition()}, Q-Values: {q_values}")
        if random.random() < self.epsilon:
            action = random.choice(legalActions)  # Explore randomly
        else:
            action = max(q_values, key=lambda x: x[1])[0]  # Exploit best action
        self.doAction(state, action)
        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor
        features = f.getFeatures(state,action)
        qvalue = 0
        for feature in features.keys():
            qvalue += self.weights[feature] * features[feature]
        if "tunnelEntry" in features:
            print(f"State: {state.getPacmanPosition()}, Action: {action}, Q-Value: {qvalue}, Features: {features}")
        return qvalue
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        actionsFromNextState = self.getLegalActions(nextState)
        maxqnext = -999999
        for act in actionsFromNextState:
            if self.getQValue(nextState,act) > maxqnext:
                maxqnext = self.getQValue(nextState,act)
        if maxqnext == -999999:
            maxqnext = 0
        diff = (reward + (self.discount * maxqnext)) - self.getQValue(state,action)
        features = self.featExtractor.getFeatures(state,action)
        self.qvalue[(state,action)] += self.alpha * diff 
        for feature in features.keys():
            self.weights[feature] += self.alpha * diff * features[feature]
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("Learned Weights:", self.weights)
            pass