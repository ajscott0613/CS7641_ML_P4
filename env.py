import numpy as np
import gym
import agents
from gym.envs.toy_text.frozen_lake import generate_random_map

random_map = generate_random_map(size=35, p=0.8)
env = gym.make("FrozenLake-v1", desc=random_map)
# env = gym.make("FrozenLake-v1", map_name='8x8')
env.reset()                    
# env.render()

# print("Action space: ", env.action_space)
# print("Observation space: ", env.observation_space)

# print(env.action_space.n)
# print("-----------------------")

# new_state, reward, done, info = env.step(2)
# print("new_state: ", new_state)
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)
# print(env.env.P[new_state][0])
# env.render()

# new_state, reward, done, info = env.step(2)
# print("new_state: ", new_state)
# print("reward: ", reward)
# print("done: ", done)
# print("info: ", info)
# env.render()


print("Value Iteration &&&&&&&&&&&&&&&&&&&&&&&")
# print(env.env.P[5])

fl_VI = agents.valueItr(env, gamma=0.95)
fl_VI.value_iteration(verbose=True)
fl_VI.getPolicy(verbose=True)
policy = fl_VI.policy


e=0
for episode in range(100):
	state = env.reset()
	for t in range(10000):
		state, reward, done, info = env.step(policy[state])
		if done:
			if reward == 1:
				e += 1
			break
print("Agent reached goal ", e, " out of 100 episodes")
env.close()



print("Policy Iteration &&&&&&&&&&&&&&&&&&&&&&&")
# print(env.env.P[5])

fl_PI = agents.policyItr(env, gamma=0.95)
fl_PI.policy_iteration(verbose=True)
policy = fl_PI.policy


e=0
for episode in range(100):
	state = env.reset()
	for t in range(10000):
		state, reward, done, info = env.step(policy[state])
		if done:
			if reward == 1:
				e += 1
			break
print("Agent reached goal ", e, " out of 100 episodes")
env.close()



print(" Q-Learning &&&&&&&&&&&&&&&&&&&&&&&")
QLearner = agents.Q_Learning(env, gamma=0.95)
QLearner.train_agent()
QTable = QLearner.Q
Q_actions = []
for i in range(len(QTable)):
	Q_actions.append(np.argmax(QTable[i, :]))
Q_actions = np.array(Q_actions).reshape(8, 8)
print("### Q-Table ###")
print(QTable)
print(Q_actions)


e=0
for episode in range(100):
	state = env.reset()
	for t in range(10000):
		action = np.argmax(QTable[state, :])
		state, reward, done, info = env.step(action)
		if done:
			if reward == 1:
				e += 1
			break
print("Agent reached goal ", e, " out of 100 episodes")
env.close()


