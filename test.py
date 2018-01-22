import gym
#from gym import wrappers
env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, '/tmp/carpole-exp-1', force=True)
for i_episode in range(1):
    observation = env.reset()
    for t in range(200):
        env.render()
        print(observation)
        action = 0#env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print observation, reward, done, info
        
        if done:
            print("episode finished after {} timesteps".format(t+1))
            break
        else:
            print "stepping"
            env.step(action)
