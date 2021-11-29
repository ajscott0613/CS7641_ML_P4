import numpy as np
import gym
import agents
from hiive.mdptoolbox.example import forest
import time

# P, R = forest(S=2000, r1=100, r2= 15, p=0.01)
# # P, R = forest(S=5, r1=10, r2= 1000, p=0.0)

# print(P)
# print("----------------")
# print(R)
# print(P[0])
# print(P[1])
# print(P[0][1])
# print(R[1])
# env  = (P, R)


class valueItr():

	def __init__(self, env, gamma=0.95, threshold=0.0001):

		self.threshold = threshold
		observation_space = np.shape(env[1])[0]
		action_space = np.shape(env[1])[1]
		self.env = env
		self.P = env[0]
		P = self.P
		self.R = env[1]
		self.allStates = np.array(list(range(observation_space)))
		self.allActions = np.array(list(range(action_space)))
		# print(self.allActions)
		self.gamma = gamma
		self.values = np.zeros(np.shape(self.allStates))
		self.policy = np.random.randint(action_space, size=observation_space)
		self.trans  = dict()
		for s in range(observation_space):
			self.trans[s] = dict()
			for a in range(action_space):
				trans_list = []
				for t in range(len(P[a][s])):
					if P[a][s][t] > 0.0:
						trans_list.append((P[a][s][t], t))
				self.trans[s][a] = trans_list
		# print(self.trans)
		self.itr = 0
		self.iterArray = []
		self.maxdiffArray = []
		self.runtime = 0.0

	def value_iteration(self, verbose=False, debug=False):

		t0 = time.time()
		self.itr = 0
		max_iters = 1000000

		while self.itr < max_iters:
			prev_values = self.values.copy()
			# reward = 0.0
			for state in self.allStates:
				# if state == len(self.allStates) - 1:
				# 	reward = 1.0
				actionVals = np.zeros(np.shape(self.allActions))
				for action in self.allActions:
					# print("state = ", state, ", action = ", action)
					trans = self.trans[state][action]
					# print(trans)
					for t in trans:
						if state < self.allStates[-1]:
							actionVals[action] += self.R[state][action] + self.gamma*t[0]*self.values[t[1]]
						else:
							actionVals[action] += self.R[state][action]
						# print(t)
						# print(t[0]*self.values[t[1]])

				# print("actionVals: ", actionVals)
				# input()

				self.values[state] = np.max(actionVals)

			if np.max(np.abs(self.values - prev_values)) < self.threshold:
				print(self.values)
				break
			self.itr += 1

			self.iterArray.append(self.itr)
			self.maxdiffArray.append(np.max(np.abs(self.values - prev_values)))
		t1 = time.time()
		self.runtime = np.round(t1 - t0, 4)




	def getPolicy(self, verbose=False):

		reward = 0.0
		for state in self.allStates:
			if state == len(self.allStates) - 1:
				reward = 1.0
			actionVals = np.zeros(np.shape(self.allActions))
			for action in self.allActions:
				trans = self.trans[state][action]
				for t in trans:
					actionVals[action] += t[0]*self.values[t[1]]
				# print("state = ", state, ", action = ", action)
				# print("actionVals: ", actionVals)
			self.policy[state] = int(np.argmax(actionVals))



	def getTime(self):
		return self.runtime

	def getIters(self):
		return self.itr

	def convergenceData(self):
		return self.iterArray, self.maxdiffArray


class policyItr():

	def __init__(self, env, gamma=0.95, threshold=0.0001):

		self.threshold = threshold
		observation_space = np.shape(env[1])[0]
		action_space = np.shape(env[1])[1]
		self.env = env
		self.P = env[0]
		P = self.P
		self.R = env[1]
		self.allStates = np.array(list(range(observation_space)))
		self.allActions = np.array(list(range(action_space)))
		# print(self.allActions)
		self.gamma = gamma
		self.values = np.zeros(np.shape(self.allStates))
		# self.policy = np.random.randint(action_space, size=observation_space)
		self.policy = np.zeros(np.shape(self.allStates), dtype=int)

		self.trans  = dict()
		for s in range(observation_space):
			self.trans[s] = dict()
			for a in range(action_space):
				trans_list = []
				for t in range(len(P[a][s])):
					if P[a][s][t] > 0.0:
						trans_list.append((P[a][s][t], t))
				self.trans[s][a] = trans_list
		# print(self.trans)

		self.itr = 0
		self.itrFine = 0
		self.iterArray_Fine = []
		self.maxdiffArray_Fine = []
		self.iterArray = []
		self.maxdiffArray = []
		self.runTime = 0.0
		self.chgStates = []


	def policy_iteration(self, verbose=False):

		t0 = time.time()
		stable = False
		self.itr = 0
		while not stable:

			# policy iteration
			max_iters = 10000
			flag = False
			while self.itr < max_iters:
				prev_values = self.values.copy()
				for state in self.allStates:
					action = self.policy[state]
					trans = self.trans[state][action]
					actionVal = 0.0
					for t in trans:
						if state < self.allStates[-1]:
							actionVal += self.R[state][action] + self.gamma*t[0]*self.values[t[1]]
						else:
							actionVal += self.R[state][action]
					self.values[state] = actionVal
				if np.max(np.abs(self.values - prev_values)) < self.threshold:
					break
				# itr += 1
				flag = True
				self.iterArray_Fine.append(self.itrFine)
				self.maxdiffArray_Fine.append(np.max(np.abs(self.values - prev_values)))

			## polic improvement
			policy_prev = self.policy.copy()
			for state in self.allStates:
				actionVals = np.zeros(np.shape(self.allActions))
				for action in self.allActions:
					trans = self.trans[state][action]
					for t in trans:
						actionVals[action] += t[0]*self.values[t[1]]
				self.policy[state] = np.argmax(actionVals)

			if np.array_equal(policy_prev, self.policy):
				stable = True

			self.itr += 1

			self.iterArray.append(self.itr)
			self.maxdiffArray.append(np.max(np.abs(self.values - prev_values)))
			self.chgStates.append((policy_prev != self.policy).sum())
		t1 = time.time()
		self.runtime = np.round(t1 - t0, 4)

	def getStateCnvg(self):
		return self.iterArray, self.chgStates

	def getTime(self):
		return self.runtime

	def convergenceData(self):
		# print(self.iterArray)
		return self.iterArray_Fine, self.maxdiffArray_Fine

	def getIters(self):
		return self.itr

	def getItersAll(self):
		return self.itrFine


class Q_Learning():

	def __init__(self, env, gamma=0.99, alpha=0.02, epsilon=0.9995, eps_decay=0.9999):

		self.env = env
		self.P, self.R = env
		P = self.P
		self.observation_space = np.shape(env[1])[0]
		self.action_space = np.shape(env[1])[1]
		self.Q = np.zeros((self.observation_space, self.action_space))
		self.runTime = 0.0
		self.epochs = []
		self.maxdiffs = []
		self.AvgWinRate = []
		self.winEpochs = []

		# hyper-parameters
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_decay = eps_decay
		self.alpha = alpha
		self.itr = 0

		self.trans  = dict()
		for s in range(self.observation_space):
			self.trans[s] = dict()
			for a in range(self.action_space):
				trans_list = []
				for t in range(len(P[a][s])):
					if P[a][s][t] > 0.0:
						trans_list.append((P[a][s][t], t))
				self.trans[s][a] = trans_list
		# print(self.trans)
		# print(self.trans[0][0])
		# print(self.trans[0][1][0][1])
		# print(len(self.trans[0][0]))
		# print(len(self.trans[0][1]))
		self.reward_array = []


	def train_agent(self):

		t0 = time.time()
		self.itr = 0
		max_iters = 60000
		max_iters = 120000
		max_iters = 75000
		burn_in = 20000
		converged = False
		counter = 0
		wins = 0
		zeroCheck = self.Q.copy()
		avgItr = 0
		winArray = []

		winRate = 0.0
		winRatePrev = 100.0
		winRateCnt = 0
		winRateFlg = False

		# while not converged and self.itr < max_iters:
		while self.itr < max_iters:
		# while not winRateFlg and self.itr < 150000:

			done = False
			# state = 0
			state = 0
			Qprev = self.Q.copy()
			env_steps = 0
			reward_total = 0

			while not done and env_steps < self.observation_space*2:
			# while True:


				# Qprev = self.Q.copy()
				# self.env.render()
				# input()
				# if state == self.observation_space - 1:
				# 	done = True

				if np.random.random() < self.epsilon:
					action = np.random.randint(self.action_space)
				else:
					action = np.argmax(self.Q[state, :])

				if state == self.observation_space - 1:
					done = True

				# print("current state: ", state)
				# print("current action: ", action)

				# print("action: ", action)
				# print("state = ", state)
				if len(self.trans[state][action]) > 1:
					# print(self.trans[state][action][0][0])
					if np.random.random() > self.trans[state][action][0][0]:
						new_state = self.trans[state][action][1][1]
						# print("grow")
					else:
						new_state = self.trans[state][action][0][1]
						# print("burned")
				else:
					new_state = self.trans[state][action][0][1]
				reward = self.R[state][action]
				# print("new_state = ", new_state)
				# input()
				reward_total += reward



				self.Q[state, action] = self.Q[state, action] + \
					self.alpha*(reward + self.gamma*np.max(self.Q[new_state, :]) - self.Q[state, action]) 

				# self.env.render()
				# input()

				# state = new_state

				# if done:
				# 	if reward > 0.0:
				# 		wins += 1
				# 		break
					# else:
					# 	self.Q[state, action] = 0.0

				state = new_state


				# if done:
				# 	if reward > 0.0:
				# 		# print("Iter #", itr, " WIN!")
				# 		wins += 1
				# 		winArray.append(1)
				# 	else:
				# 		winArray.append(0)
				env_steps += 1
				# print(state)
				# print(done)

			# if self.itr > burn_in:
			# 	if np.abs(winRate - winRatePrev) < 1.0:
			# 		winRateCnt += 1
			# 		if winRateCnt >= 500:
			# 			winRateFlg = True
			# 	else:
			# 		winRateCnt = 0
			# 		winRateFlg = False
			# winRatePrev = winRate
			# print("finished episode")
			# if self.itr > burn_in:
			# 	if (self.AvgWinRate[-1] - self.AvgWinRate[self.itr-10000]) < 0.005:
			# 		winRateFlg = True
			# 	winrateaverage = self.AvgWinRate[-1] - self.AvgWinRate[self.itr-10000]
			# else:
			# 	winrateaverage = 0.0

			not_zero = not np.array_equal(self.Q, zeroCheck)
			if np.max(np.abs(self.Q - Qprev)) < 0.001 and (self.itr > burn_in and not_zero):
			# if (self.itr > burn_in and not_zero) and winRateFlg:
				counter += 1
			else:
				counter = 0

			if counter > 20:
				converged = True
				# print("Iter #", self.itr, "Converged!")

			if self.itr % 1000 == 0:
				print("iter num = ", self.itr, ", epsilon = ", self.epsilon, ", wins = ", wins, "max diff = ", np.max(np.abs(self.Q - Qprev)),
					"env_steps = ", env_steps, ", final state = ", state, ", zerocheck = ", not_zero, "reward _total = ", reward_total)

			# if wins > 0:
			self.epsilon *= self.eps_decay
			self.epsilon = max(self.epsilon, 0.01)
			self.itr += 1
			self.epochs.append(self.itr)
			self.maxdiffs.append(np.max(np.abs(self.Q - Qprev)))
			# if self.itr > 98:
			# 	self.winEpochs.append(self.itr)
			# 	total_wins = np.array(winArray[avgItr:]).sum()
			# 	self.AvgWinRate.append(total_wins / len(winArray[avgItr:]))
			# 	winRate = total_wins / len(winArray[avgItr:])
			self.reward_array.append(reward_total)


		t1 = time.time()
		self.runTime = np.round(t1 - t0, 4)
		print("total iterations:	", self.itr)


	def getTime(self):
		return self.runTime

	def convergenceData(self):
		# print(self.iterArray)
		return self.epochs, self.maxdiffs

	def performanceData(self):
		return self.itr, self.reward_array

	def getIters(self):
		return self.itr

	def getPolicy(self):
		Q_actions = []
		Qlen = np.shape(self.Q)[0]
		for i in range(Qlen):
			Q_actions.append(np.argmax(self.Q[i, :]))
		return np.array(Q_actions)


# P, R = forest(S=2000, r1=100, r2= 15, p=0.01)
# P, R = forest(S=5, r1=100, r2= 15, p=0.001)
# env = (P, R)

# fl_VI = valueItr(env, gamma=0.99, threshold=0.1)
# fl_VI.value_iteration(verbose=True)
# fl_VI.getPolicy(verbose=True)
# print(fl_VI.getIters())
# policy = fl_VI.policy
# print(policy)


# fl_PI = policyItr(env, gamma=0.99, threshold=0.1)
# fl_PI.policy_iteration(verbose=True)
# print(fl_PI.getIters())
# policy = fl_PI.policy
# print(policy)


# QLearner = Q_Learning(env, gamma=0.99, alpha=0.02)
# QLearner.train_agent()
# policy = QLearner.getPolicy()
# print(policy)
# iters, diff = QLearner.convergenceData()
# print("Policy Iteration Run Time: ", QLearner.getTime())
# print("Num Epochs: ", QLearner.getIters())
