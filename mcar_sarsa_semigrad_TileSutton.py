'''
Try using polynomial basis functions and episodic semi-grad SARSA (pg 276 S&B)
-linear polynomial
'''

#Got it working
#TODO: figure out why it is optimistic, is it just weights set to zero initially?
#TODO: figure out how to use inside of BIRL with a function approximation for reward, that is hopefully low dimensional (see what Littman and McLaughin did for their IRL stuff with MCar)

from TileCoding import *
from mpl_toolkits.mplot3d import Axes3D
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

##Trying to merge with suttons code

#from gym import wrappers
env = gym.make('MountainCar-v1')

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# all possible actions
ACTION_REVERSE = 0
ACTION_ZERO = 1
ACTION_FORWARD = 2
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(position, velocity, valueFunction, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(position, velocity, action))
    return np.argmax(values)
    
def getOptimalAction(position, velocity, valueFunction):
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


def run_episode(env, valueFunction, n, render=False, epsilon = 0):
    state = env.reset()
    print("init state", state)
    totalreward = 0
   
    # get initial action
    currentPosition = state[0]
    currentVelocity = state[1]
    currentAction = getAction(currentPosition, currentVelocity, valueFunction, epsilon)

    # track previous position, velocity, action and reward
    positions = [currentPosition]
    velocities = [currentVelocity]
    actions = [currentAction]
    rewards = [0.0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1
        if render: 
            env.render()

        if time < T:
            # take current action and go to the new state
            observation, reward, done, info = env.step(currentAction)
            newPosition = observation[0]
            newVelocity = observation[1]
            # choose new action
            newAction = getAction(newPosition, newVelocity, valueFunction, epsilon)

            # track new state and action
            positions.append(newPosition)
            velocities.append(newVelocity)
            actions.append(newAction)
            rewards.append(reward)

            if newPosition >= POSITION_MAX:
                T = time

        # get the time of the state to update
        updateTime = time - n
        if updateTime >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(updateTime + 1, min(T, updateTime + n) + 1):
                returns += rewards[t]
            # add estimated state action value to the return
            if updateTime + n <= T:
                returns += valueFunction.value(positions[updateTime + n],
                                               velocities[updateTime + n],
                                               actions[updateTime + n])
            # update the state value function
            if positions[updateTime] != POSITION_MAX:
                valueFunction.learn(positions[updateTime], velocities[updateTime], actions[updateTime], returns)
        if updateTime == T - 1:
            break
        currentPosition = newPosition
        currentVelocity = newVelocity
        currentAction = newAction

    
    return time
    
def run_rollout(env, start_state, init_action, valueFunction, render=False):
    env.reset()
    state = env.set_start_state(start_state)
    epsilon = 0.0
    print("init state", state)
   
    # get initial action
    currentPosition = state[0]
    currentVelocity = state[1]
    currentAction = init_action

    states_visited = []
    while True:
        if render: 
            env.render()


        # take current action and go to the new state
        observation, reward, done, info = env.step(currentAction)
        #save observations for post-processing
        states_visited.append(observation)
        newPosition = observation[0]
        newVelocity = observation[1]

        if done:
            return states_visited #can be used for extended demos and can post process features of interest


        # choose new action
        newAction = getAction(newPosition, newVelocity, valueFunction, epsilon)

        #update position and action for next step
        currentPosition = newPosition
        currentVelocity = newVelocity
        currentAction = newAction

    
if __name__ == "__main__":

    #rewards = []
    runs = 1
    episodes = 3000
    numOfTilings = 8
    alpha = 0.5

    # use optimistic initial value, so it's ok to set epsilon to 0
    EPSILON = 0

    valueFunction = ValueFunction(alpha, numOfTilings)
    for episode in range(0, episodes):
        step = run_episode(env, valueFunction,1, False, EPSILON)
        print('episode:', episode, "steps", step)
    #pickle the controller (value function)
    with open('mcar_policy.pickle', 'wb') as f:
        pickle.dump(valueFunction, f, pickle.HIGHEST_PROTOCOL)
        
    with open('mcar_policy.pickle', 'rb') as f:
        vFunc = pickle.load(f)

    #play back learned controller
    for i in range(3):
        steps = run_episode(env, vFunc, 1, True, EPSILON)
        print("time", steps)
