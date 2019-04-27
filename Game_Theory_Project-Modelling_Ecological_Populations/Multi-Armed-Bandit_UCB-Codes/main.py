################# Main function #################

import argparse
from constants import *
from alg import *
from plot import *

# Function to parse hyperparameters from the commandline
def parse():
    parser = argparse.ArgumentParser()


    parser.add_argument('--M',
                        action="store", 
                        dest="M", 
                        default=4,
                        type=int,
                        help='Method :  1. Playing against field, \
                                        2. Playing against a sampled population, \
                                        3. Pair-wise contest, \
                                        4. Group play')

    parser.add_argument('--G',
                        action="store", 
                        dest="G", 
                        default=6,
                        type=int,
                        help='Gain')

    parser.add_argument('--C',
                        action="store", 
                        dest="C", 
                        default=10,
                        type=int,
                        help='Cost')

    parser.add_argument('--n_players',
                        action="store", 
                        dest="n_players", 
                        default=100,
                        type=int,
                        help='Number of players or Population size')

    parser.add_argument('--n_epochs',
                        action="store", 
                        dest="n_epochs", 
                        default=4000,
                        type=int,
                        help='Number of epochs or Number of arm pulls')

    arg_val = parser.parse_args()
    return(arg_val)  


# Calls the algorithms and the plotting functions
def main():

    arg_val = parse()

    config = vars(arg_val)
    k = 2 # number of arms : hawk/dove

    # Two player pay-off matrix
    A = [[(config["G"]-config["C"])/2     ,   config["G"]   ],
        [0                          ,   config["G"]/2 ]]

    A = np.array(A) 

    if config["M"]==1 : 
        # Playing against the field
        N_dove, N_mu = ucb_alg_full(k, A, config) 

    elif config["M"]==2 :
        # Playing against a sampled population
        N_dove, N_mu = ucb_alg_random(k, A, config, 1) 

    elif config["M"]==3 :
        # Pair-wise contest
        N_dove, N_mu = ucb_alg_random(k, A, config, 0)  

    elif config["M"]==4 :  
        # Group Play
        N_dove, N_mu = ucb_alg_group(k, A, config)   
    else :
        print("Unknown method specified!!!")
        exit()

    # Plot the evolution of population over time
    trajectory(config, N_dove, config["M"])

    # Plot the average population pay-off over time
    payoff_trajectory(config, N_mu, config["M"])

    if config["M"]==4:
        print("Average no. of doves in the population : {}".format(np.mean(N_dove[-int(0.8*config["n_players"]):])))
        print("Average population pay-off : {}".format(np.mean(N_mu[-int(0.8*config["n_players"]):])))

    print("\n\nAt MSNE :")
    print("Average no. of doves in the population : {}".format((1-1.0*config["G"]/config["C"])*config["n_players"]))
    print("Average population pay-off : 1.2")

if __name__== "__main__":
    main()