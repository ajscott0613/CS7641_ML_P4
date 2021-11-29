import numpy as np
import time


class valueItr():

	def __init__(self, env, gamma=0.95):

		self.env = env
		self.allStates = np.array(list(range(env.observation_space.n)))
		self.allActions = np.array(list(range(env.action_space.n)))
		self.gamma = gamma
		self.values = np.zeros(np.shape(self.allStates))
		self.policy = np.random.randint(env.action_space.n, size=env.observation_space.n)
		self.dims = int(np.sqrt(env.observation_space.n))
		self.itr = 0
		self.iterArray = []
		self.maxdiffArray = []
		self.runtime = 0.0

	def value_iteration(self, verbose=False, debug=False):

		t0 = time.time()
		self.itr = 0
		max_iters = 100000
		while self.itr < max_iters:
			prev_values = self.values.copy()
			reward = 0.0
			for state in self.allStates:
				if state == len(self.allStates) - 1:
					reward = 1.0
				actionVals = np.zeros(np.shape(self.allActions))
				for action in self.allActions:
					trans = self.env.P[state][action]
					for t in trans:
						actionVals[action] += t[0]*self.values[t[1]]
				self.values[state] = reward + self.gamma*np.max(actionVals)

			# check for convergence
			if np.max(np.abs(self.values - prev_values)) < 0.0001:
				break
			self.itr += 1

			self.iterArray.append(self.itr)
			self.maxdiffArray.append(np.max(np.abs(self.values - prev_values)))
		t1 = time.time()
		self.runtime = np.round(t1 - t0, 4)

	def getTime(self):
		return self.runtime

	def getIters(self):
		return self.itr

	def convergenceData(self):
		return self.iterArray, self.maxdiffArray


	def getPolicy(self, verbose=False):

		reward = 0.0
		for state in self.allStates:
			if state == len(self.allStates) - 1:
				reward = 1.0
			actionVals = np.zeros(np.shape(self.allActions))
			for action in self.allActions:
				trans = self.env.P[state][action]
				for t in trans:
					actionVals[action] += t[0]*self.values[t[1]]
			self.policy[state] = int(np.argmax(actionVals))


class policyItr():

	def __init__(self, env, gamma=0.95):

		self.env = env
		self.allStates = np.array(list(range(env.observation_space.n)))
		self.allActions = np.array(list(range(env.action_space.n)))
		self.gamma = gamma
		self.values = np.zeros(np.shape(self.allStates))
		self.policy = np.random.randint(env.action_space.n, size=env.observation_space.n)
		self.dims = int(np.sqrt(env.observation_space.n))
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

			## policy iteration
			max_iters = 10000
			flag = False
			while self.itr < max_iters:
				prev_values = self.values.copy()
				reward = 0.0
				for state in self.allStates:
					if state == len(self.allStates) - 1:
						reward = 1.0
					action = self.policy[state]
					trans = self.env.P[state][action]
					actionVal = 0.0
					for t in trans:
						actionVal += t[0]*self.values[t[1]]
					self.values[state] = reward + self.gamma*actionVal
				if np.max(np.abs(self.values - prev_values)) < 0.0001:
					break
				self.itrFine += 1
				flag = True
				self.iterArray_Fine.append(self.itrFine)
				self.maxdiffArray_Fine.append(np.max(np.abs(self.values - prev_values)))


			## polic improvement
			policy_prev = self.policy.copy()
			for state in self.allStates:
				actionVals = np.zeros(np.shape(self.allActions))
				for action in self.allActions:
					trans = self.env.P[state][action]
					for t in trans:
						actionVals[action] += t[0]*self.values[t[1]]
				self.policy[state] = np.argmax(actionVals)

			if np.array_equal(policy_prev, self.policy):
				stable = True
			self.itr += 1
			# print("itr = ", self.itr)

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

	def __init__(self, env, gamma=0.9, alpha=0.02, epsilon=0.9995, eps_decay=0.9999):

		self.env = env
		self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
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


	def train_agent(self):

		t0 = time.time()
		self.itr = 0
		max_iters = 60000
		max_iters = 120000
		max_iters = 250000
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

		while not converged and self.itr < max_iters:
		# while self.itr < max_iters:
		# while not winRateFlg and self.itr < 150000:

			done = False
			# state = 0
			state = self.env.reset()
			Qprev = self.Q.copy()
			env_steps = 0

			while not done:
			# while True:


				# Qprev = self.Q.copy()
				# self.env.render()
				# input()

				if np.random.random() < self.epsilon:
					action = np.random.randint(self.env.action_space.n)
				else:
					action = np.argmax(self.Q[state, :])

				# print("current state: ", state)
				# print("current action: ", action)

				new_state, reward, done, info = self.env.step(action)

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

				if done:
					if reward > 0.0:
						# print("Iter #", itr, " WIN!")
						wins += 1
						winArray.append(1)
					else:
						winArray.append(0)
				env_steps += 1

			# if self.itr > burn_in:
			# 	if np.abs(winRate - winRatePrev) < 1.0:
			# 		winRateCnt += 1
			# 		if winRateCnt >= 500:
			# 			winRateFlg = True
			# 	else:
			# 		winRateCnt = 0
			# 		winRateFlg = False
			# winRatePrev = winRate
			if self.itr > burn_in:
				if (self.AvgWinRate[-1] - self.AvgWinRate[self.itr-10000]) < 0.005:
					winRateFlg = True
				winrateaverage = self.AvgWinRate[-1] - self.AvgWinRate[self.itr-10000]
			else:
				winrateaverage = 0.0

			not_zero = not np.array_equal(self.Q, zeroCheck)
			# if np.max(np.abs(self.Q - Qprev)) < 0.001 and (self.itr > burn_in and not_zero) and winRateFlg:
			if (self.itr > burn_in and not_zero) and winRateFlg:
				counter += 1
			else:
				counter = 0

			if counter > 20:
				converged = True
				# print("Iter #", self.itr, "Converged!")

			if self.itr % 1000 == 0:
				print("iter num = ", self.itr, ", epsilon = ", self.epsilon, ", wins = ", wins, "max diff = ", np.max(np.abs(self.Q - Qprev)),
					"env_steps = ", env_steps, ", final state = ", state, ", zerocheck = ", not_zero, "Win rate avg = ",
					winrateaverage)

			# if wins > 0:
			self.epsilon *= self.eps_decay
			self.epsilon = max(self.epsilon, 0.01)
			self.itr += 1
			self.epochs.append(self.itr)
			self.maxdiffs.append(np.max(np.abs(self.Q - Qprev)))
			if self.itr > 98:
				self.winEpochs.append(self.itr)
				total_wins = np.array(winArray[avgItr:]).sum()
				self.AvgWinRate.append(total_wins / len(winArray[avgItr:]))
				winRate = total_wins / len(winArray[avgItr:])


		t1 = time.time()
		self.runTime = np.round(t1 - t0, 4)
		print("total iterations:	", self.itr)


	def getTime(self):
		return self.runTime

	def convergenceData(self):
		# print(self.iterArray)
		return self.epochs, self.maxdiffs

	def performanceData(self):
		return self.winEpochs, self.AvgWinRate

	def getIters(self):
		return self.itr

	def getPolicy(self):
		Q_actions = []
		Qlen = np.shape(self.Q)[0]
		for i in range(Qlen):
			Q_actions.append(np.argmax(self.Q[i, :]))
		return np.array(Q_actions)


