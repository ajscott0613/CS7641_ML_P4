import numpy as np
import matplotlib.pyplot as plt
import gym
import agents
from gym.envs.toy_text.frozen_lake import generate_random_map
import frozenlake
import sys, getopt


def runFrozenLake(env, policy, itrs=1, verbose=False):
	avgScore =[]
	for i in range(itrs):
		e=0
		for episode in range(100):
			state = env.reset()
			for t in range(10000):
				state, reward, done, info = env.step(policy[state])
				if done:
					# if reward == 1:
					if state == env.observation_space.n - 1:
						e += 1
					break
		avgScore.append(e)
		if verbose:
			print("Run #", i, "Agent reached goal ", e, " out of 100 episodes")
	env.close()
	return np.round(np.mean(avgScore), 0)

def compare_policies(p1, p2):
	dim = np.shape(p1)

	diff = []
	count = 0
	for i in range(dim[0]):
		for j in range(dim[1]):
			if p1[i][j] != p2[i][j]:
				diff.append(i*dim[0] + j)
				count += 1

	return count, diff


if __name__ == "__main__":

	EXPS = [3]
	DIMS = [4, 6, 8, 10, 12, 14, 16, 18]

	OPTIMAL_POLICY_8 = np.array([[3, 2, 2, 2, 2, 2, 2, 2],
	 [3, 3, 3, 3, 3, 2, 2, 1],
	 [3, 3, 0, 0, 2, 3, 2, 1],
	 [3, 3, 3, 1, 0, 0, 2, 2],
	 [0, 3, 0, 0, 2, 1, 3, 2],
	 [0, 0, 0, 1, 3, 0, 0, 2],
	 [0, 0, 1, 0, 0, 0, 0, 2],
	 [0, 1, 0, 0, 1, 2, 1, 0]])


	for i in range(len(sys.argv)):
			if sys.argv[i] == '-exp':
				if sys.argv[i+1] == 'vipi':
					if sys.argv[i+2] == 'gamma':
						EXPS = [1]
					if sys.argv[i+2] == 'size':
						EXPS = [2]
				if sys.argv[i+1] == 'ql':
					if sys.argv[i+2] == 'all':
						EXPS = [4]
					if sys.argv[i+2] == 'optimal':
						EXPS = [5] 

	for exp in EXPS:

		## value and policy iteration vs gamma
		if exp == 1:

			data = np.zeros((12, 4))
			dataItr = 0
			gammas = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
			DIM = DIMS[2]
			random_map = generate_random_map(size=DIM)
			env = gym.make("FrozenLake-v1", map_name='8x8')
			env.reset()  
			env.render()

			# run value iteration
			Alliters = []
			Alldiffs = []
			for gamma in gammas:
				fl_VI = agents.valueItr(env, gamma=gamma)
				fl_VI.value_iteration(verbose=True)
				fl_VI.getPolicy(verbose=True)
				policy = fl_VI.policy
				iters, diff = fl_VI.convergenceData()
				Alliters.append(iters)
				Alldiffs.append(diff)
				# print(fl_VI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_VI.policy)
				print("Value Iteration Run Time: ", fl_VI.getTime())
				print("Num Iters: ", fl_VI.getIters())
				avgscore = runFrozenLake(env, fl_VI.policy, itrs=100)
				print("Average Score: ", avgscore)
				data[dataItr, 0] = gamma
				data[dataItr, 1] = fl_VI.getTime()
				data[dataItr, 2] = fl_VI.getIters()
				data[dataItr, 3] = avgscore
				dataItr += 1
				print(compare_policies(OPTIMAL_POLICY_8, policy.reshape(DIM, DIM)))

			# plot data
			legends = []
			for i in range(len(gammas)):
				plt.plot(Alliters[i], Alldiffs[i])
				legends.append("gamma = " + str(gammas[i]))
			plt.legend(legends, loc="upper right")
			plt.xlabel("Iterations")
			plt.ylabel("Max Difference (Utility)")
			plt.title("Value Iteration Convergence")
			#plt.show()
			plt.savefig('Value_Iteration_Gamma.png')
			plt.close()


			# run policy iteration
			Alliters = []
			Alldiffs = []
			for gamma in gammas:
				fl_PI = agents.policyItr(env, gamma=gamma)
				fl_PI.policy_iteration(verbose=True)
				policy = fl_PI.policy
				# iters, diff = fl_PI.convergenceData()
				# iters = fl_PI.getIters()
				iters, diff = fl_PI.getStateCnvg()
				Alliters.append(iters)
				Alldiffs.append(diff)
				# print(fl_PI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_PI.policy)
				avgscore = runFrozenLake(env, fl_PI.policy, itrs=100)
				print("Policy Iteration Run Time: ", fl_PI.getTime())
				print("Num Iters (main): ", fl_PI.getIters())
				print("Num Iters (all): ", fl_PI.getItersAll())
				print("Average Score: ", avgscore)
				data[dataItr, 0] = gamma
				data[dataItr, 1] = fl_PI.getTime()
				data[dataItr, 2] = fl_PI.getIters()
				data[dataItr, 3] = avgscore
				dataItr += 1
				print(compare_policies(OPTIMAL_POLICY_8, policy.reshape(DIM, DIM)))

			# plot data
			legends = []
			for i in range(len(gammas)):
				plt.plot(Alliters[i], Alldiffs[i])
				legends.append("gamma = " + str(gammas[i]))
			plt.legend(legends, loc="upper right")
			plt.xlabel("Iterations")
			plt.ylabel("Max Difference (Utility)")
			plt.title("Policy Iteration Convergence")
			#plt.show()
			plt.savefig('Policy_Iteration_Gamma.png')
			plt.close()


			columns = ('gamma', 'Runtime', 'Iterations', 'Avg Score')
			rows = (' Value Itr ', ' Value Itr ', ' Value Itr ', 
				' Value Itr ', ' Value Itr ', ' Value Itr ', ' Policy Itr ',
				 ' Policy Itr ', ' Policy Itr ', ' Policy Itr ', 
				 ' Policy Itr ', ' Policy Itr ',)
			cell_text = []
			n_rows = len(data)


			# Initialize the vertical-offset for the stacked bar chart.
			y_offset = np.zeros(len(columns))
			for row in range(n_rows):
			    y_offset = y_offset + data[row]
			    cell_text.append([x for x in data[row]])
			# Reverse colors and text labels to display the last value at the top.
			# cell_text.reverse()
			# Add a table at the bottom of the axes
			the_table = plt.table(cellText=cell_text,
	                      rowLabels=rows,
	                      colLabels=columns,
	                      loc='center')

			ax = plt.gca()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.box(on=None)

			fig = plt.gcf()
			plt.savefig('Policy_Value_Iteration.png',
	            bbox_inches='tight',
	            dpi=150
	            )


		if exp == 2:

			data = np.zeros((20, 20))
			dataItr = 0
			DIM = DIMS[2]
			# env.render()

			# run value iteration
			Alliters = []
			Alldiffs = []
			for dim in DIMS:
				random_map = generate_random_map(size=dim)
				env = gym.make("FrozenLake-v1", desc=random_map)
				env.reset()  
				fl_VI = agents.valueItr(env, gamma=0.99)
				fl_VI.value_iteration(verbose=True)
				fl_VI.getPolicy(verbose=True)
				policy = fl_VI.policy
				iters, diff = fl_VI.convergenceData()
				Alliters.append(iters)
				Alldiffs.append(diff)
				# print(fl_VI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_VI.policy)
				print("Value Iteration Run Time: ", fl_VI.getTime())
				print("Num Iters: ", fl_VI.getIters())
				avgscore = runFrozenLake(env, fl_VI.policy, itrs=100)
				print("Average Score: ", avgscore)
				data[dataItr, 0] = dim
				data[dataItr, 1] = fl_VI.getTime()
				data[dataItr, 2] = fl_VI.getIters()
				data[dataItr, 3] = avgscore
				dataItr += 1

			# plot data
			legends = []
			for i in range(len(DIMS)):
				plt.plot(Alliters[i], Alldiffs[i])
				legends.append("dim = " + str(DIMS[i]) + 'x' + str(DIMS[i]))
			plt.legend(legends, loc="upper right")
			plt.xlabel("Iterations")
			plt.ylabel("Max Difference (Utility)")
			plt.title("Value Iteration Convergence")
			#plt.show()
			plt.savefig('Value_Iteration_Size.png')
			plt.close()


			# run policy iteration
			Alliters = []
			Alldiffs = []
			for dim in DIMS:
				random_map = generate_random_map(size=dim)
				env = gym.make("FrozenLake-v1", desc=random_map)
				env.reset()  
				fl_PI = agents.policyItr(env, gamma=0.99)
				fl_PI.policy_iteration(verbose=True)
				policy = fl_PI.policy
				# iters, diff = fl_PI.convergenceData()
				# iters = fl_PI.getIters()
				iters, diff = fl_PI.getStateCnvg()
				Alliters.append(iters)
				Alldiffs.append(diff)
				# print(fl_PI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_PI.policy)
				avgscore = runFrozenLake(env, fl_PI.policy, itrs=100)
				print("Policy Iteration Run Time: ", fl_PI.getTime())
				print("Num Iters (main): ", fl_PI.getIters())
				print("Num Iters (all): ", fl_PI.getItersAll())
				print("Average Score: ", avgscore)
				data[dataItr, 0] = dim
				data[dataItr, 1] = fl_PI.getTime()
				data[dataItr, 2] = fl_PI.getIters()
				data[dataItr, 3] = avgscore
				dataItr += 1

			# plot data
			legends = []
			for i in range(len(DIMS)):
				plt.plot(Alliters[i], Alldiffs[i])
				legends.append("dim = " + str(DIMS[i]) + 'x' + str(DIMS[i]))
			plt.legend(legends, loc="upper right")
			plt.xlabel("Iterations")
			plt.ylabel("States Changed)")
			plt.title("Policy Iteration Convergence")
			#plt.show()
			plt.savefig('Policy_Iteration_Size.png')
			plt.close()

			




		if exp == 3:

			data = np.zeros((12, 4))
			dataItr = 0
			# gammas = [0.9, 0.99]
			gammas = [0.999]
			DIM = DIMS[2]
			random_map = generate_random_map(size=DIM)
			env = gym.make("FrozenLake-v1", map_name='8x8')
			env.reset()  
			env.render()


			Alliters = []
			Alldiffs = []
			AllEpcohs = []
			AllWinsAvg = []
			for gamma in gammas:
				QLearner = agents.Q_Learning(env, gamma=gamma)
				QLearner.train_agent()
				policy = QLearner.getPolicy()
				iters, diff = QLearner.convergenceData()
				Alliters.append(iters)
				Alldiffs.append(diff)
				epochs, winavg = QLearner.performanceData()
				AllEpcohs.append(epochs)
				AllWinsAvg.append(winavg)
				# print(fl_PI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_PI.policy)
				avgscore = runFrozenLake(env, policy, itrs=100)
				print("Policy Iteration Run Time: ", QLearner.getTime())
				print("Num Epochs: ", QLearner.getIters())
				print("Average Score: ", avgscore)
				data[dataItr, 0] = gamma
				data[dataItr, 1] = QLearner.getTime()
				data[dataItr, 2] = QLearner.getIters()
				data[dataItr, 3] = avgscore
				dataItr += 1
				print(compare_policies(OPTIMAL_POLICY_8, policy.reshape(DIM, DIM)))

			# plot data
			legends = []
			for i in range(len(gammas)):
				plt.plot(Alliters[i], Alldiffs[i])
				legends.append("gamma = " + str(gammas[i]))
			plt.legend(legends, loc="upper right")
			plt.xlabel("Epochs")
			plt.ylabel("Q-Learning (diff)")
			plt.title("Q-Learning Convergence")
			#plt.show()
			plt.savefig('Q_Learning_Gamma.png')
			plt.close()

			# plot data
			legends = []
			for i in range(len(gammas)):
				plt.plot(AllEpcohs[i], AllWinsAvg[i])
				legends.append("gamma = " + str(gammas[i]))
			plt.legend(legends, loc="upper left")
			plt.xlabel("Epochs")
			plt.ylabel("Win Rate Average")
			plt.title("Q-Learning Win Rate")
			#plt.show()
			plt.savefig('Q_Learning_Win_Rate_2.png')
			plt.close()



		if exp == 4:

			data = np.zeros((9, 5))
			dataItr = 0
			# gammas = [0.9, 0.99]
			gammas = [0.9, 0.99, 0.999]
			alphas = [0.01, 0.02, 0.05]
			DIM = DIMS[2]
			random_map = generate_random_map(size=DIM)
			env = gym.make("FrozenLake-v1", map_name='8x8')
			env.reset()  
			env.render()


			Alliters = []
			Alldiffs = []
			AllEpcohs = []
			AllWinsAvg = []
			for gamma in gammas:
				for alpha in alphas:
					QLearner = agents.Q_Learning(env, gamma=gamma, alpha=alpha)
					QLearner.train_agent()
					policy = QLearner.getPolicy()
					iters, diff = QLearner.convergenceData()
					Alliters.append(iters)
					Alldiffs.append(diff)
					epochs, winavg = QLearner.performanceData()
					AllEpcohs.append(epochs)
					AllWinsAvg.append(winavg)
					# print(fl_PI.policy.reshape(DIM, DIM))
					# runFrozenLake(env, fl_PI.policy)
					avgscore = runFrozenLake(env, policy, itrs=100)
					print("Policy Iteration Run Time: ", QLearner.getTime())
					print("Num Epochs: ", QLearner.getIters())
					print("Average Score: ", avgscore)
					data[dataItr, 0] = gamma
					data[dataItr, 1] = alpha
					data[dataItr, 2] = QLearner.getTime()
					data[dataItr, 3] = QLearner.getIters()
					data[dataItr, 4] = avgscore
					dataItr += 1
					# print(compare_policies(OPTIMAL_POLICY_8, policy.reshape(DIM, DIM)))


			# plot data
			gamma_keys = [0.9, 0.9, 0.9, 0.99, 0.99, 0.99, 0.999, 0.999, 0.999]
			alpha_keys = [0.01, 0.02, 0.05, 0.01, 0.02, 0.05, 0.01, 0.02, 0.05]
			legends = []
			for i in range(len(gamma_keys)):
				plt.plot(AllEpcohs[i], AllWinsAvg[i])
				legends.append("gamma = " + str(gamma_keys[i]) + ", alpha = " + str(alpha_keys[i]))
			plt.legend(legends, loc="upper left")
			plt.xlabel("Epochs")
			plt.ylabel("Win Rate Average")
			plt.title("Q-Learning Win Rate")
			#plt.show()
			plt.savefig('Q_Learning_Win_Rate_R2.png')
			plt.close()


			columns = ('gamma', 'alpha', 'Runtime', 'Iterations', 'Avg Score')
			rows = (' Q-Learner ', ' Q-Learner ', ' Q-Learner ', 
				' Q-Learner ', ' Q-Learner ', ' Q-Learner ', ' Q-Learner ',
				 ' Q-Learner ', ' Q-Learner ',)
			cell_text = []
			n_rows = len(data)


			# Initialize the vertical-offset for the stacked bar chart.
			y_offset = np.zeros(len(columns))
			for row in range(n_rows):
			    y_offset = y_offset + data[row]
			    cell_text.append([x for x in data[row]])
			# Reverse colors and text labels to display the last value at the top.
			# cell_text.reverse()
			# Add a table at the bottom of the axes
			the_table = plt.table(cellText=cell_text,
	                      rowLabels=rows,
	                      colLabels=columns,
	                      loc='center')

			ax = plt.gca()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			plt.box(on=None)

			fig = plt.gcf()
			plt.savefig('Q_Learning_Stats_Frozen_Lake_R2.png',
	            bbox_inches='tight',
	            dpi=150
	            )

		if exp == 5:

			DIM = DIMS[2]
			random_map = generate_random_map(size=DIM)
			env = gym.make("FrozenLake-v1", map_name='8x8')
			env.reset()  
			env.render()
			QLearner = agents.Q_Learning(env, gamma=0.99, alpha=0.02)
			QLearner.train_agent()
			policy = QLearner.getPolicy()
			epochs, winavg = QLearner.performanceData()
			avgscore = runFrozenLake(env, policy, itrs=100)
			print("Policy Iteration Run Time: ", QLearner.getTime())
			print("Num Epochs: ", QLearner.getIters())
			print("Average Score: ", avgscore)
			print("Policy: ")
			print(policy.reshape(DIM, DIM))


			plt.plot(epochs, winavg)
			plt.xlabel("Epochs")
			plt.ylabel("Win Rate Average")
			plt.title("Q-Learning Win Rate")
			#plt.show()
			plt.savefig('Q_Learning_Win_Rate_Optimal.png')
			plt.close()