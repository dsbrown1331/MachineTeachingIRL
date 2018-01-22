'''
Try out a bunch of random linear controllers and see which one works best (highest cumulative reward)
'''



import gym
import numpy as np

#from gym import wrappers
env = gym.make('CartPole-v0')
n_obs = len(env.observation_space.sample())
n_actions = env.action_space.n
#env = wrappers.Monitor(env, '/tmp/carpole-exp-1', force=True)

n_reps = 10
render = False
n_pop = 1000
best_score = 0

for i_pop in range(n_pop):
    cum_score = 0
    #generate random linear controller
    weights = 2 * np.random.rand(n_obs) - 1
    observation = env.reset()
    for t in range(200):
        #print t
        if render:
            env.render()
        #print(observation)
        controller_input = np.sign(np.dot(weights,observation))
        if controller_input < 0:
            action = 0
        else:
            action = 1
        observation, reward, done, info = env.step(action)
        #print observation, reward, done, info
        cum_score += reward
        
        if done:
            #print("episode finished after {} timesteps".format(t+1))
            #print "breaking"
            break

    
    print "average score:", cum_score
    if cum_score > best_score:
        best_score = cum_score
        best_weights = weights
print "best score", best_score
print "best weights", best_weights
##run winning controller  
render = True      
observation = env.reset()
cum_score = 0
for t in range(200):
    #print t
    if render:
        env.render()
    #print(observation)
    controller_input = np.sign(np.dot(best_weights,observation))
    if controller_input < 0:
        action = 0
    else:
        action = 1
    observation, reward, done, info = env.step(action)
    #print observation, reward, done, info
    cum_score += reward
    
    if done:
        #print("episode finished after {} timesteps".format(t+1))
        #print "breaking"
        break

print "score", cum_score
