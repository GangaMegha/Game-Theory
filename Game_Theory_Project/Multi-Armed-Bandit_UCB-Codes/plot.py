############ Plotting functions #############

import matplotlib.pyplot as plt
import numpy as np
from constants import *
import os

# Evolution of population over time
def trajectory(hp, pops, M):
	plt.figure(dpi=300)
	plt.plot(pops,label='UCB behaviour', alpha=0.6)
	plt.hlines((1-1.0*hp["G"]/hp["C"])*hp["n_players"],0,hp["n_epochs"],color='r',linestyles='-',label='MSNE')
	plt.legend()
	plt.xlabel('Time')
	plt.ylabel('No. of Doves')
	plt.title('Evolution of population over time')
	if M==1:
		plt.savefig(os.path.join('MAB_plots','field_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	elif M==2:
		plt.savefig(os.path.join('MAB_plots','sampled_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	elif M==3:
		plt.savefig(os.path.join('MAB_plots','pair_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	else:
		plt.savefig(os.path.join('MAB_plots','group_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))

	plt.close()

# Average population pay-off over time
def payoff_trajectory(hp, pops, M):
	plt.figure(dpi=300)
	plt.plot(pops,label='UCB behaviour', alpha=0.6)
	plt.hlines(1.2,0,hp["n_epochs"],color='r',linestyles='-',label='MSNE')
	plt.legend()
	plt.xlabel('Time')
	plt.ylabel('Avg Population Pay-off')
	plt.title('Population Pay-off over time')
	if M==1:
		plt.savefig(os.path.join('MAB_plots','pay-off_field_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	elif M==2:
		plt.savefig(os.path.join('MAB_plots','pay-off_sampled_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	elif M==3:
		plt.savefig(os.path.join('MAB_plots','pay-off_pair_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	else:
		plt.savefig(os.path.join('MAB_plots','pay-off_group_{}_{}_epochs.png'.format(hp["n_players"], hp["n_epochs"])))
	plt.close()
