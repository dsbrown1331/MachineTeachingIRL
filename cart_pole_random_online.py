'''
Online solution from kvfrans.com
Try out a bunch of random linear controllers and see which one works best (highest cumulative reward)
'''



import gym
import numpy as np

#from gym import wrappers
env = gym.make('Pendulum-v0')

def run_episode(env, parameters, render=False):
    observation = env.reset()
    totalreward = 0
    
    for t in xrange(200):
        if render: 
            env.render()
        if np.dot(parameters, observation) < 0:
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

bestparams = None
bestreward = 0
n_reps = 5
for i in xrange(1000):
    print(i)
    parameters = np.random.rand(4)*2 - 1
    cum_reward = 0
    for rep in xrange(n_reps):
        r = run_episode(env, parameters)
        cum_reward += r
    reward = cum_reward / float(n_reps)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        #check if solved
        if reward == 200:
            break
#play back best controller
print(run_episode(env, bestparams, True))
