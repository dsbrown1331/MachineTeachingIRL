'''
Try using polynomial basis functions and episodic semi-grad SARSA (pg 276 S&B)
-linear polynomial
'''

#TODO try Sutton's tilecoding software instead of poly basis to see if it works

import gym
import numpy as np
import matplotlib.pyplot as plt





# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(position, velocity, valueFunction):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(position, velocity, action))
    return np.argmax(values)
    
# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @maxSize: the maximum # of indices
    def __init__(self, stepSize, numOfTilings=8, maxSize=2048):
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.stepSize = stepSize / numOfTilings

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        # position and velocity needs scaling to satisfy the tile software
        self.positionScale = self.numOfTilings / (POSITION_MAX - POSITION_MIN)
        self.velocityScale = self.numOfTilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def getActiveTiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        activeTiles = tiles(self.hashTable, self.numOfTilings,
                            [self.positionScale * position, self.velocityScale * velocity],
                            [action])
        return activeTiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        #if position == POSITION_MAX:
        #    return 0.0
        activeTiles = self.getActiveTiles(position, velocity, action)
        return np.sum(self.weights[activeTiles])

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        activeTiles = self.getActiveTiles(position, velocity, action)
        estimation = np.sum(self.weights[activeTiles])
        delta = self.stepSize * (target - estimation)
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    # get # of steps to reach the goal under current state value function
    def costToGo(self, position, velocity):
        costs = []
        for action in ACTIONS:
            costs.append(self.value(position, velocity, action))
        return -np.max(costs)



def qval_argmax(state):
    features = featurize(state)
    qvals = np.dot(W,features)
    return np.argmax(qvals)

def featurize(state):
    features = [1]
    features.extend(state)
    for i in range(len(state)):
        for j in range(i,len(state)):
            features.append(state[i] * state[j])
    return np.array(features)

def run_episode(env, learning_rate, eps, render=False):
    state = env.reset()
    totalreward = 0
    action = qval_argmax(state)
    #eps greedy
    if np.random.rand() < eps:
        action = np.random.randint(n_actions)
    for t in xrange(200):
        if render: 
            env.render()
        #take action
        observation, reward, done, info = env.step(action)
        state_features = featurize(state)
        next_state_features = featurize(observation)
        qsa_est = np.dot(W[action,:], state_features)
        qsa_grad = state_features
        
        if done: #terminal state update
            update_direction = (reward - qsa_est) * qsa_grad 
            W[action,:] += learning_rate * update_direction
            if render:
                print observation
            break
        ###normal SARSA semi-gradient update
        action_next = qval_argmax(state)
        #eps greedy
        if np.random.rand() < eps:
            action_next = np.random.randint(n_actions)
        if render:
            print observation
            print "action: ", action_next
        qsa_next_est = np.dot(W[action_next,:], next_state_features)
        update_direction = (reward + gamma * qsa_next_est - qsa_est) * qsa_grad
        W[action,:] += learning_rate * update_direction
        #W[:] -= 0.0001 * W
        #update state and action and reward
        state = observation
        action = action_next
        totalreward += reward
    return totalreward

#from gym import wrappers
env = gym.make('CartPole-v0')
n_actions = 2
n_features = 15
gamma = 1
#init w
W = 200*np.ones((n_actions, n_features)) 



rewards = []
for i in xrange(300):
    print i
    learning_rate = 0.02#1.0/np.sqrt(1000.0+i)
    print learning_rate
    eps = 0#0.1#1.0/(1.0 + i)
    r = run_episode(env, learning_rate, eps)
    rewards.append(r)

#play back learned controller
r = run_episode(env, 0, 0, True)
print "reward", r
plt.plot(rewards)
plt.show()
print W
