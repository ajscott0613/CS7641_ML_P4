import numpy as np
import matplotlib.pyplot as plt
import gym
import forestAgents
from gym.envs.toy_text.frozen_lake import generate_random_map
import frozenlake
import sys, getopt
from hiive.mdptoolbox.example import forest

if __name__ == "__main__":


	for i in range(len(sys.argv)):
			if sys.argv[i] == '-exp':
				if sys.argv[i+1] == 'vipi':
					if sys.argv[i+2] == 'gamma':
						EXPS = [1]
					if sys.argv[i+2] == 'size':
						EXPS = [2]
				if sys.argv[i+1] == 'ql':
					EXPS = [4]

	for exp in EXPS:

		## value and policy iteration vs gamma
		if exp == 1:

			data = np.zeros((12, 3))
			dataItr = 0
			gammas = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
			P, R = forest(S=1000, r1=100, r2= 15, p=0.1)
			env = (P, R)

			# run value iteration
			Alliters = []
			Alldiffs = []
			for gamma in gammas:
				fl_VI = forestAgents.valueItr(env, gamma=gamma)
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
				# avgscore = runFrozenLake(env, fl_VI.policy, itrs=100)
				# print("Average Score: ", avgscore)
				data[dataItr, 0] = gamma
				data[dataItr, 1] = fl_VI.getTime()
				data[dataItr, 2] = fl_VI.getIters()
				# data[dataItr, 3] = avgscore
				dataItr += 1


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
			plt.savefig('Value_Iteration_Gamma_Forest.png')
			plt.close()


			# run policy iteration
			Alliters = []
			Alldiffs = []
			for gamma in gammas:
				fl_PI = forestAgents.policyItr(env, gamma=gamma)
				fl_PI.policy_iteration(verbose=True)
				policy = fl_PI.policy
				# iters, diff = fl_PI.convergenceData()
				# iters = fl_PI.getIters()
				iters, diff = fl_PI.getStateCnvg()
				Alliters.append(iters)
				Alldiffs.append(diff)
				# print(fl_PI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_PI.policy)
				# avgscore = runFrozenLake(env, fl_PI.policy, itrs=100)
				print("Policy Iteration Run Time: ", fl_PI.getTime())
				print("Num Iters (main): ", fl_PI.getIters())
				print("Num Iters (all): ", fl_PI.getItersAll())
				# print("Average Score: ", avgscore)
				data[dataItr, 0] = gamma
				data[dataItr, 1] = fl_PI.getTime()
				data[dataItr, 2] = fl_PI.getIters()
				# data[dataItr, 3] = avgscore
				dataItr += 1


			# plot data
			legends = []
			for i in range(len(gammas)):
				plt.plot(Alliters[i], Alldiffs[i])
				legends.append("gamma = " + str(gammas[i]))
			plt.legend(legends, loc="upper right")
			plt.xlabel("Iterations")
			plt.ylabel("Num States Changed")
			plt.title("Policy Iteration Convergence")
			#plt.show()
			plt.savefig('Policy_Iteration_Gamma_Forest.png')
			plt.close()


			columns = ('gamma', 'Runtime', 'Iterations')
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
			plt.savefig('Policy_Value_Iteration_Forest.png',
	            bbox_inches='tight',
	            dpi=150
	            )

		if exp == 2:

			data = np.zeros((12, 3))
			dataItr = 0
			states = [3, 5, 20, 100, 500, 1000]
			for n_states in states:
				P, R = forest(S=n_states, r1=100, r2= 15, p=0.1)
				env = (P, R)

				# run value iteration
				# Alliters = []
				# Alldiffs = []
				# Alltime = []
				fl_VI = forestAgents.valueItr(env, gamma=0.6)
				fl_VI.value_iteration(verbose=True)
				fl_VI.getPolicy(verbose=True)
				policy = fl_VI.policy
				# iters, diff = fl_VI.convergenceData()
				# Alliters.append(fl_VI.getIters())
				# Allstates.append(states)
				# Alltime.append(fl_VI.getTime())
				# # print(fl_VI.policy.reshape(DIM, DIM))
				# # runFrozenLake(env, fl_VI.policy)
				# print("Value Iteration Run Time: ", fl_VI.getTime())
				# print("Num Iters: ", fl_VI.getIters())
				# # avgscore = runFrozenLake(env, fl_VI.policy, itrs=100)
				# # print("Average Score: ", avgscore)
				data[dataItr, 0] = n_states
				data[dataItr, 1] = fl_VI.getTime()
				data[dataItr, 2] = fl_VI.getIters()
				# data[dataItr, 3] = avgscore
				dataItr += 1

			for n_states in states:
				P, R = forest(S=n_states, r1=100, r2= 15, p=0.1)
				env = (P, R)

				fl_PI = forestAgents.policyItr(env, gamma=0.6)
				fl_PI.policy_iteration(verbose=True)
				policy = fl_PI.policy
				# iters, diff = fl_PI.convergenceData()
				# iters = fl_PI.getIters()
				# iters, diff = fl_PI.getStateCnvg()
				# Alliters.append(iters)
				# Alldiffs.append(diff)
				# print(fl_PI.policy.reshape(DIM, DIM))
				# runFrozenLake(env, fl_PI.policy)
				# avgscore = runFrozenLake(env, fl_PI.policy, itrs=100)
				# print("Policy Iteration Run Time: ", fl_PI.getTime())
				# print("Num Iters (main): ", fl_PI.getIters())
				# print("Num Iters (all): ", fl_PI.getItersAll())
				# print("Average Score: ", avgscore)
				data[dataItr, 0] = n_states
				data[dataItr, 1] = fl_PI.getTime()
				data[dataItr, 2] = fl_PI.getIters()
				# data[dataItr, 3] = avgscore
				dataItr += 1

			columns = ('Num States', 'Runtime', 'Iterations')
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
			plt.savefig('Policy_Value_Iteration_Forest_Size.png',
	            bbox_inches='tight',
	            dpi=150
	            )



		if exp == 4:

			data = np.zeros((4, 5))
			dataItr = 0
			# gammas = [0.9, 0.99]
			gammas = [0.6, 0.7]
			alphas = [0.01, 0.1]
			P, R = forest(S=1000, r1=100, r2= 15, p=0.001)
			env = (P, R)


			Alliters = []
			Alldiffs = []
			AllEpcohs = []
			AllWinsAvg = []
			for gamma in gammas:
				for alpha in alphas:
					QLearner = forestAgents.Q_Learning(env, gamma=gamma, alpha=alpha)
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
					# avgscore = runFrozenLake(env, policy, itrs=100)
					print("Policy Iteration Run Time: ", QLearner.getTime())
					print("Num Epochs: ", QLearner.getIters())
					# print("Average Score: ", avgscore)
					data[dataItr, 0] = gamma
					data[dataItr, 1] = alpha
					data[dataItr, 2] = QLearner.getTime()
					data[dataItr, 3] = QLearner.getIters()
					data[dataItr, 4] = winavg[-1]
					dataItr += 1
					# print(compare_policies(OPTIMAL_POLICY_8, policy.reshape(DIM, DIM)))
					print(policy)


			# plot data
			# gamma_keys = [0.9, 0.9, 0.9, 0.99, 0.99, 0.99, 0.999, 0.999, 0.999]
			# alpha_keys = [0.01, 0.02, 0.05, 0.01, 0.02, 0.05, 0.01, 0.02, 0.05]
			gamma_keys = [0.6, 0.6, 0.7, 0.7]
			alpha_keys = [0.01, 0.1, 0.01, 0.1]
			legends = []
			for i in range(len(gamma_keys)):
				plt.plot(AllEpcohs[i], AllWinsAvg[i])
				legends.append("gamma = " + str(gamma_keys[i]) + ", alpha = " + str(alpha_keys[i]))
			plt.legend(legends, loc="upper left")
			plt.xlabel("Epochs")
			plt.ylabel("Total Rewards")
			plt.title("Q-Learning Rewards")
			#plt.show()
			plt.savefig('Q_Learning_Reward_Forest.png')
			plt.close()


			columns = ('gamma', 'alpha', 'Runtime', 'Iterations', 'Avg Score')
			rows = (' Q-Learner ', ' Q-Learner ', ' Q-Learner ', 
				' Q-Learner ',)
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
			plt.savefig('Q_Learning_Stats_Forest.png',
	            bbox_inches='tight',
	            dpi=150
	            )
