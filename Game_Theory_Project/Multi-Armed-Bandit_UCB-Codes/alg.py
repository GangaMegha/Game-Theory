import numpy as np
from constants import * 
from sampler import *

############################ Pay-off functions ###############################

# For playing against the field
def payoff(A, len_arr, population, player_type):
	# Pay-off for player i playing against the entire population when he's a hawk as well as a dove
	payoff_val_1 = np.sum([A[player_type][j] for j in population]) - A[player_type][player_type]
	payoff_val_2 = np.sum([A[1-player_type][j] for j in population]) - A[1-player_type][player_type]

	return np.ones(len_arr)*payoff_val_1/(len(population)-1), np.ones(len_arr)*payoff_val_1/(len(population)-1)

# For playing against a sampled population and pair wise contest
def payoff_random(A, arr, population, player_type, num=1, flag=0):
	N = len(population)
	payoff_val_1 = np.zeros(len(arr))
	payoff_val_2 = np.zeros(len(arr))

	for i in range(len(arr)):
		while(True) :
			if flag==1:
				# Choose the number of players to be sampled from the population
				num = np.random.random_integers(1,int(0.1*N))
			indx = np.random.random_integers(0,N-1,num)
			payoff_val_1[i] = np.mean([A[player_type][j] for j in population[indx]])
			payoff_val_2[i] = np.mean([A[1-player_type][j] for j in population[indx]])

			if(arr[i] not in indx) :
				break
	return payoff_val_1, payoff_val_2

# For group play
def payoff_group(A, population, player_type, num=1):

	# Compute the pay-off when he's a hawk and a dove
	payoff_val_1 = np.sum([A[player_type][j] for j in population])-A[player_type,player_type]
	payoff_val_2 = np.sum([A[1-player_type][j] for j in population])-A[1-player_type,player_type]
	
	return payoff_val_1/(len(population)-1), payoff_val_2/(len(population)-1)


############################ UCB ###############################

# For playing against the field
def ucb_alg_full(k, A, config):

	N_dove_avg = [] # Stores the avg number of hawk in each iteration
	N_mu_cap_avg = [] # stores the average mu_cap of the population in each iteration

	for i in range(repeat):

		# Holds the value of mu_cap or payoff for each player(row) and each arm (column)
		Players_mu_cap = np.zeros((config["n_players"], k))
		t = np.zeros((config["n_players"],1)) # Total number of arm pulls by each player
		T_k = np.zeros((config["n_players"],k)) # No. of plays of each arm for each player

		N_dove = [] # Stores the number of hawk in each iteration
		N_mu_cap = [] # Stores the mu_cap of the population in each iteration

		#---------------------------------------------------------
		'''
		Start with an initial random distribution of hawk and dove in the population
		'''
		# Randomly generated n_players samples of 0/1 
		Players_sample = np.random.random_integers(0,1,config["n_players"])

		# Calculate the average payoff when Player i is hawk and dove and the rest of the population remains the same
		index_0 = np.where(Players_sample==0)[0]
		index_1 = np.where(Players_sample==1)[0]

		# Each Hawk player pulls arm corresponding to Hawk as well as dove while rest of the population is the same
		Players_mu_cap[index_0,0], Players_mu_cap[index_0,1] = payoff(A, len(index_0), Players_sample, 0)
		T_k[:,0] += 1

		# Each Dove player pulls arm corresponding to Dove as well as hawk while rest of the population is the same
		Players_mu_cap[index_1,1], Players_mu_cap[index_1,0] = payoff(A, len(index_1), Players_sample, 1)
		T_k[:,1] += 1

		t[:,0] += k

		# The square root add on term in UCB algorithm
		add_on = np.sqrt(8*np.log(t)/T_k)

		# Stores the number of hawk in each iteration
		N_dove.append(len(np.where(Players_sample==1)[0]))
		# Stores the average population pay-off in each iteration
		N_mu_cap.append(np.mean(np.concatenate((Players_mu_cap[index_0, 0],Players_mu_cap[index_1, 1])) ))

		#---------------------------------------------------------

		for j in range(k,config["n_epochs"]):
			# Sample arm according to UCB algorithm (New population distribution)
			Players_sample = np.argmax(np.add(Players_mu_cap, add_on), axis=1)
			# Players_sample = [bern(T_k[p][1]/t[p], 1) for p in range(len(T_k))]
		
			# Stores the number of hawk in each iteration
			N_dove.append(len(np.where(Players_sample==1)[0]))

			# Find the payoff of each player corresponding to the new population when they are a hawk and dove
			index_0 = np.where(Players_sample==0)[0]
			index_1 = np.where(Players_sample==1)[0]

			# Since the population keeps changing, we need to update mu_cap_k for both arms 
			Players_mu_cap[index_0,0], Players_mu_cap[index_0,1] = payoff(A, len(index_0), Players_sample, 0)

			Players_mu_cap[index_1,1], Players_mu_cap[index_1,0] = payoff(A, len(index_1), Players_sample, 1)

			# Stores average population pay-off after each iteration
			N_mu_cap.append(np.mean(np.concatenate((Players_mu_cap[index_0, 0],Players_mu_cap[index_1, 1])) ))

			# Update the variables storing number of arm pulls
			T_k[index_0,0] += 1
			T_k[index_1,1] += 1
			t[:,0] +=1

			# The square root add on term in UCB algorithm
			add_on = np.sqrt(8*np.log(t)/T_k)

		N_dove_avg.append(N_dove)
		N_mu_cap_avg.append(N_mu_cap)

	# Averaging over multiple experiments
	N_dove_avg = np.array(N_dove_avg)
	N_dove_avg = np.mean(N_dove_avg, axis=0)

	N_mu_cap_avg = np.array(N_mu_cap_avg)
	N_mu_cap_avg = np.mean(N_mu_cap_avg, axis=0)

	return(N_dove_avg, N_mu_cap)

# For playing against a sampled population(flag=1) as well as for pair-wise contest (flag=0)
def ucb_alg_random(k, A, config, flag=0):

	N_dove_avg = [] # Stores the avg number of hawk in each iteration
	N_mu_cap_avg = [] # stores the average mu_cap of the population in each iteration

	for i in range(repeat):

		# Holds the value of mu_cap or payoff for each player(row) and each arm (column)
		Players_mu_cap = np.zeros((config["n_players"], k))
		t = np.zeros((config["n_players"],1)) # Total number of arm pulls by each player
		T_k = np.zeros((config["n_players"],k)) # No. of plays of each arm for each player

		N_dove = [] # Stores the number of hawk in each iteration
		N_mu_cap = [] # Stores the mu_cap of the population in each iteration

		alpha = 0.3

		#---------------------------------------------------------
		'''
		Start with an initial random distribution of hawk and dove in the population
		'''
		# Randomly generated n_players samples of 0/1 
		Players_sample = np.random.random_integers(0,1,config["n_players"])
		# Players_sample = np.zeros(config["n_players"]) # Initialise population to all Hawk
		# Players_sample = np.ones(config["n_players"]) # Initialise population to all Dove

		# Calculate the average payoff when Player i is hawk and dove and the rest of the population remains the same
		index_0 = np.where(Players_sample==0)[0]
		index_1 = np.where(Players_sample==1)[0]

		# Each Hawk player pulls arm corresponding to Hawk as well as dove while rest of the population is the same
		Players_mu_cap[index_0,0], Players_mu_cap[index_0,1] = payoff_random(A, index_0, Players_sample, 0, flag)
		T_k[:,0] += 1

		# Each Dove player pulls arm corresponding to Dove as well as hawk while rest of the population is the same
		Players_mu_cap[index_1,1], Players_mu_cap[index_1,0] = payoff_random(A, index_1, Players_sample, 1, flag)
		T_k[:,1] += 1

		t[:,0] += k

		# The square root add on term in UCB algorithm
		add_on = np.sqrt(8*np.log(t)/T_k)

		# Stores the number of hawk and the average population pay-off in each iteration
		N_dove.append(len(np.where(Players_sample==1)[0]))
		N_mu_cap.append(np.mean(np.concatenate((Players_mu_cap[index_0,0],Players_mu_cap[index_1,1])) ))

		#---------------------------------------------------------

		for j in range(k,config["n_epochs"]):
			# Sample arm according to UCB algorithm (New population distribution)
			Players_sample = np.argmax(np.add(Players_mu_cap, add_on), axis=1)

			# Stores the number of hawk in each iteration
			N_dove.append(len(np.where(Players_sample==1)[0]))

			# Find the payoff of each player corresponding to the new population when they are a hawk and dove
			index_0 = np.where(Players_sample==0)[0]
			index_1 = np.where(Players_sample==1)[0]

			# Since the population keeps changing, we need to update mu_cap_k for both arms 
			Players_mu_cap[index_0,0], Players_mu_cap[index_0,1] = payoff_random(A, index_0, Players_sample, 0, flag)

			Players_mu_cap[index_1,1], Players_mu_cap[index_1,0] = payoff_random(A, index_1, Players_sample, 1, flag)

			# Stores average population pay-off after each iteration
			N_mu_cap.append(np.mean(np.concatenate((Players_mu_cap[index_0,0],Players_mu_cap[index_1,1])) ))

			# Update the variables storing number of arm pulls
			T_k[index_0,0] += 1
			T_k[index_1,1] += 1
			t[:,0] +=1

			# The square root add on term in UCB algorithm
			add_on = np.sqrt(8*np.log(t)/T_k)

		N_dove_avg.append(N_dove)
		N_mu_cap_avg.append(N_mu_cap)

	# Compute the average over multiple experiments
	N_dove_avg = np.array(N_dove_avg)
	N_dove_avg = np.mean(N_dove_avg, axis=0)

	N_mu_cap_avg = np.array(N_mu_cap_avg)
	N_mu_cap_avg = np.mean(N_mu_cap_avg, axis=0)

	return(N_dove_avg, N_mu_cap)

# For group play
def ucb_alg_group(k, A, config):

	N_dove_avg = [] # Stores the avg number of hawk in each iteration
	N_mu_cap_avg = [] # stores the average mu_cap of the population in each iteration

	for i in range(repeat):

		# Holds the value of mu_cap or payoff for each player(row) and each arm (column)
		Players_mu_cap = np.zeros((config["n_players"], k))
		t = np.zeros((config["n_players"],1)) # Total number of arm pulls by each player
		T_k = np.zeros((config["n_players"],k)) # No. of plays of each arm for each player

		N_dove = [] # Stores the number of hawk in each iteration

		N_mu_cap = [] # Stores the mu_cap of the population in each iteration

		#---------------------------------------------------------
		'''
		Start with an initial random distribution of hawk and dove in the population
		'''
		# Randomly generated n_players samples of 0/1 
		Players_sample = np.random.random_integers(0,1,config["n_players"])
		# Players_sample = np.zeros(config["n_players"]) # Initialise population to all Hawk
		# Players_sample = np.ones(config["n_players"]) # Initialise population to all Dove

		# Calculate the average payoff when Player i is hawk and dove and the rest of the population remains the same
		index_0 = np.where(Players_sample==0)[0]
		index_1 = np.where(Players_sample==1)[0]

		# Each Hawk player pulls arm corresponding to Hawk as well as dove while rest of the population is the same
		Players_mu_cap[index_0,0], Players_mu_cap[index_0,1] = payoff_random(A, index_0, Players_sample, 0, num=5)
		T_k[:,0] += 1

		# Each Dove player pulls arm corresponding to Dove as well as while rest of the population is the same
		Players_mu_cap[index_1,1], Players_mu_cap[index_1,0] = payoff_random(A, index_1, Players_sample, 1, num=5)
		T_k[:,1] += 1

		t[:,0] += k

		# The square root add on term in UCB algorithm
		add_on = np.sqrt(8*np.log(t)/T_k)

		# Stores the number of hawk and the average population pay-off in each iteration
		N_dove.append(len(np.where(Players_sample==1)[0]))
		N_mu_cap.append(np.mean(np.concatenate((Players_mu_cap[index_0,0],Players_mu_cap[index_1,1])) ))

		#---------------------------------------------------------

		for j in range(k,config["n_epochs"]):
			# Sample arm according to UCB algorithm (New population distribution)
			Players_sample = np.argmax(np.add(Players_mu_cap, add_on), axis=1)
		
			# Stores the number of hawk in each iteration
			N_dove.append(len(np.where(Players_sample==1)[0]))

			n_group = np.random.random_integers(2,int(0.1*config['n_players'])) # Number of ppl who undergo interaction and modification per iteration

			# Sample players to undergo interaction in this round
			sample_index = np.random.random_integers(0, config["n_players"]-1, n_group)

			for idx in sample_index:
				# Since the population keeps changing, we need to update mu_cap_k for both arms 
				Players_mu_cap[idx,Players_sample[idx]], Players_mu_cap[idx,1-Players_sample[idx]] = payoff_group(A, Players_sample[sample_index], Players_sample[idx], num=5)

				# Update the variables storing number of arm pulls only for the current player type
				T_k[idx,Players_sample[idx]] += 1
				t[idx,0] +=1

			# The square root add on term in UCB algorithm
			add_on = np.sqrt(8*np.log(t)/T_k)
			
			# Find the new average population pay-off
			index_0 = np.where(Players_sample==0)[0]
			index_1 = np.where(Players_sample==1)[0]
			N_mu_cap.append(np.mean(np.concatenate((Players_mu_cap[index_0,0],Players_mu_cap[index_1,1])) ))

		N_dove_avg.append(N_dove)

	# Compute the average over multiple experiments
	N_dove_avg = np.array(N_dove_avg)
	N_dove_avg = np.mean(N_dove_avg, axis=0)

	N_mu_cap_avg = np.array(N_mu_cap_avg)
	N_mu_cap_avg = np.mean(N_mu_cap_avg, axis=0)

	return(N_dove_avg, N_mu_cap)

