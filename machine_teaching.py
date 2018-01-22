#from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getAction
import pickle
import mcar_sarsa_semigrad_TileSutton
from mcar_sarsa_semigrad_TileSutton import ValueFunction, run_episode, getOptimalAction, run_rollout
import gym
import time
import numpy as np

def constant_feature_map(state):
    return np.array([1.0]);
    
def compute_feature_counts(feature_map, states, discount):
    fcounts = np.zeros(len(feature_map(states[0])))
    for i in range(len(states)):
        fcounts += discount ** i * feature_map(states[i])
    return fcounts

def get_feature_half_planes(state_feature_map, env, start_state, valueFunction, horizon, discount = 1.0):
    #run rollouts counting up the features for each possible action
    for init_action in range(env.action_space.n):
        print("init action", init_action)
        print("opt", getOptimalAction(start_state[0], start_state[1], valueFunction))
        states_visited = run_rollout(env, start_state, init_action, valueFunction, render=True)
        #print(states_visited)
        #compute feature counts
        fcounts = compute_feature_counts(state_feature_map, states_visited, discount)
        print(fcounts)
    
    

#rewards = []
runs = 1
episodes = 2000
numOfTilings = 8
alpha = 0.5
EPSILON = 0
num_samples = 10

horizon = 1
env = gym.make('MountainCar-v1')
# use optimistic initial value, so it's ok to set epsilon to 0
EPSILON = 0
#get optimal policy that has been previously learned
with open('mcar_policy.pickle', 'rb') as f:
    valueFunction = pickle.load(f)
    


#figure out the feasible region

#sample starting states
all_planes = []
#TODO could parallelize this!
for i in range(num_samples):
    start_state = env.observation_space.sample()
    half_planes = get_feature_half_planes(constant_feature_map, env, start_state, valueFunction, horizon) #return dictionary s:constraints
    #all_planes.extend(half_planes) #does this work for dictionaries?
    #constraint_maps = remove_redundancies(all_planes) #return dictionary s:non-redundant normed constraint?
    #solve_set_cover(constraints)
    
    #normed_planes = normalize(all_planes)
    #reduced_planes = remove_duplicates(normed_planes)
    
    

##play back learned controller
#for i in range(10):
#    #use random seeds to sample starting states and to enable repeating same state with different actions.

#    env.seed(123)
#    env.start_episode((-1.2,0))
#    env.render()
#    time.sleep(5)
#    steps = run_episode(env, valueFunction, 1, True)
#    print("time", steps)
