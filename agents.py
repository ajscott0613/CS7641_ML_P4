import numpy as np


class valueItr():

	def __init__(self, env, gamma=0.95):

		self.env = env
		self.allStates = np.array(list(range(env.observation_space.n)))
		self.allActions = np.array(list(range(env.action_space.n)))
		self.gamma = gamma
		self.values = np.zeros(np.shape(self.allStates))
		self.policy = np.random.randint(env.action_space.n, size=env.observation_space.n)
		self.dims = int(np.sqrt(env.observation_space.n))

	def value_iteration(self, verbose=False, debug=False):

		itr = 0
		max_iters = 10000
		if debug:
			print(self.values.reshape((self.dims, self.dims)))
		while itr < max_iters:
			prev_values = self.values.copy()
			reward = 0.0
			for state in self.allStates:
				if state == len(self.allStates) - 1:
					reward = 1.0
				actionVals = np.zeros(np.shape(self.allActions))
				for action in self.allActions:
					trans = self.env.env.P[state][action]
					for t in trans:
						actionVals[action] += t[0]*self.values[t[1]]
						if debug:
							print("---------------------------------")
							print("current state: ", state)
							print("current action: ", action)
							print("t[0]: ", t[0])
							print("t[1]: ", t[1])
							print("self.values[t[1]]: ", self.values[t[1]])
							print("actionVals[action]: ", actionVals[action])
							input()
				self.values[state] = reward + self.gamma*np.max(actionVals)
				if debug:
					print("self.values[state]: ", self.values[state])
			if debug:
				print("** VALUES **")
				print(self.values.reshape((self.dims, self.dims)))
			if np.max(np.abs(self.values - prev_values)) < 0.001:
				break
			itr += 1

		if verbose:
			print("iters: ", itr)
			self.env.render()
			print("### STATE VALUES ###")
			print(self.values.reshape(self.dims, self.dims))


	def getPolicy(self, verbose=False):

		reward = 0.0
		for state in self.allStates:
			if state == len(self.allStates) - 1:
				reward = 1.0
			actionVals = np.zeros(np.shape(self.allActions))
			for action in self.allActions:
				trans = self.env.env.P[state][action]
				for t in trans:
					actionVals[action] += t[0]*self.values[t[1]]
			self.policy[state] = int(np.argmax(actionVals))

		if verbose:
			print("### POLCIY ###")
			print(self.policy.reshape(self.dims, self.dims))

class policyItr():

	def __init__(self, env, gamma=0.95):

		self.env = env
		self.allStates = np.array(list(range(env.observation_space.n)))
		self.allActions = np.array(list(range(env.action_space.n)))
		self.gamma = gamma
		self.values = np.zeros(np.shape(self.allStates))
		self.policy = np.random.randint(env.action_space.n, size=env.observation_space.n)
		self.dims = int(np.sqrt(env.observation_space.n))


	def policy_iteration(self, verbose=False):


		stable = False
		itr = 0
		while not stable:

			# policy iteration
			max_iters = 10000
			flag = False
			while itr < max_iters:
				prev_values = self.values.copy()
				reward = 0.0
				for state in self.allStates:
					if state == len(self.allStates) - 1:
						reward = 1.0
					action = self.policy[state]
					trans = self.env.env.P[state][action]
					actionVal = 0.0
					for t in trans:
						actionVal += t[0]*self.values[t[1]]
					self.values[state] = reward + self.gamma*actionVal
				if np.max(np.abs(self.values - prev_values)) < 0.001:
					break
				itr += 1
				flag = True

			## polic improvement
			policy_prev = self.policy.copy()
			for state in self.allStates:
				actionVals = np.zeros(np.shape(self.allActions))
				for action in self.allActions:
					trans = self.env.env.P[state][action]
					for t in trans:
						actionVals[action] += t[0]*self.values[t[1]]
				self.policy[state] = np.argmax(actionVals)

			if np.array_equal(policy_prev, self.policy):
				stable = True

		if verbose:
			print("iters: ", itr)
			# self.env.render()
			print("### STATE POLICY ###")
			print(self.policy.reshape(self.dims, self.dims))


class Q_Learning():

	def __init__(self, env, gamma=0.9):

		self.env = env
		self.gamma = gamma
		self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
		# self.Q = np.random.rand(self.env.observation_space.n, self.env.action_space.n)
		self.epsilon = 0.995
		self.eps_decay = 0.9999
		self.alpha = 0.01


	def train_agent(self):

		itr = 0
		max_iters = 100000
		converged = False
		counter = 0
		wins = 0
		while not converged and itr < max_iters:

			done = False
			# state = 0
			state = self.env.reset()
			Qprev = self.Q.copy()

			while not done:


				# Qprev = self.Q.copy()

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

				state = new_state

				if done:
					if reward > 0.0:
						# print("Iter #", itr, " WIN!")
						wins += 1


			if np.max(np.abs(self.Q - Qprev)) < 0.00001 and itr > 10000:
				counter += 1
			else:
				counter = 0

			if counter > 20:
				converged = True
				print("Iter #", itr, "Converged!")

			if itr % 1000 == 0:
				print("iter num = ", itr, ", epsilon = ", self.epsilon, ", wins = ", wins, "max diff = ", np.max(np.abs(self.Q - Qprev)))


			self.epsilon *= self.eps_decay
			self.epsilon = max(self.epsilon, 0.01)
			itr += 1

		print("total iterations:	", itr)


