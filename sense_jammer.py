import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--protocol', action='store', default='control')
args = parser.parse_args()
PROTOCOL = args.protocol

print("You are running the sense jammer with protocol: %s"%PROTOCOL)

import imageio
import numpy as np, matplotlib

np.set_printoptions(3,suppress=True)
matplotlib.use('Agg')
font = {'family' : 'DejaVu Sans',
		'weight' : 'bold',
		'size'   : 7}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt


from jam_game import Jam_Game
from constants import *

class Sense_Jammer():
	def __init__(self,env):
		self.env = env

	def choose_action(self):
		state = self.env.get_state()
		state = np.power(state,2)
		mean_power = np.mean(np.mean(state))
		if np.max(state[:,1]) > mean_power:
			# jam on the highest channel
			return np.argmax(state[:,1]) + 1 
		else:
			return 0 # wait

if __name__ == "__main__":
	env = Jam_Game(n_channels, power_a_size, n_memory,protocol=PROTOCOL,dbg=False)
	sense_jammer = Sense_Jammer(env)
	episode_count = 0; episode_count_max = 5
	do_figs = False
	episode_rewards = []; episode_throughputs = []

	while True:
		# Viz variables
		viz_dict  = {
			"frames": [], # holds spectral images
			"jammer_frames": [],
			"rewards": [], # holds all rewards received, even when stuck
			"powers": [], # holds power decisions
			"descriptions": [], # holds descriptions of jammer action
			"times_of_application": [],
			"descriptions_applied": [],
			"rewards_applied": [] # holds all rewards applied
		}
		first_episode_state = np.zeros((sense_jammer.env.n_channels,sense_jammer.env.n_memory))
		viz_counter = -1
		#
		episode_reward = 0
		episode_throughput = 0
		episode_step_count = 0
		d = False

		sense_jammer.env.new_episode()
		s = sense_jammer.env.get_state()
		viz_dict["frames"].append(s)
		threshold = sense_jammer.env.tx_rx_pair.jam_threshold

		while not env.is_episode_finished():
			viz_counter += 1
			chan_action = sense_jammer.choose_action()
			pow_action = 0 # doesn't matter right now
			r,r_str = sense_jammer.env.make_action(chan_action, pow_action)
			
			if r is None:
				viz_dict["rewards"].append((sense_jammer.env.calculate_reward(chan_action,"throughput"))[0])
				viz_dict["descriptions"].append("stuck")
				# We are waiting for an action to complete
				# visualization
				s1 = sense_jammer.env.get_state()
				viz_dict["frames"].append(s1)
				if sense_jammer.env.last_action[0] != 0:
					first_episode_state[sense_jammer.env.last_action[0]-1,0] = sense_jammer.env.jam_props["power"];
				viz_dict["jammer_frames"].append(first_episode_state)
				viz_dict["powers"].append(sense_jammer.env.power_map[pow_action])
				first_episode_state = np.insert(first_episode_state[:,0:sense_jammer.env.n_memory-  1],0,np.zeros((1,sense_jammer.env.n_channels)),axis=1) 
				continue


			#variables that we only copy over if we aren't skipping frames	
			r_last = r  
			episode_reward += r
			episode_throughput += sense_jammer.env.tx_rx_pair.throughput
			episode_step_count += 1
			viz_dict["descriptions"].append(r_str)
			viz_dict["rewards"].append(r)
			## # Save for visualizations 
			if chan_action != 0:
				first_episode_state[chan_action-1,0] = sense_jammer.env.jam_props["power"];
			viz_dict["jammer_frames"].append(first_episode_state)
			viz_dict["powers"].append(sense_jammer.env.power_map[pow_action])
			first_episode_state = np.insert(first_episode_state[:,0:sense_jammer.env.n_memory-  1],0,np.zeros((1,sense_jammer.env.n_channels)),axis=1) 
			### 


			# check to see if the episode is finished (probably not applicable)
			d = sense_jammer.env.is_episode_finished() or episode_step_count == max_episode_length - 1
			if not d and viz_counter < episode_count_max:
				s1 = sense_jammer.env.get_state()
				viz_dict["frames"].append(s1)

			viz_dict["times_of_application"].append(viz_counter)
			viz_dict["descriptions_applied"].append(r_str)
			
			if viz_counter == 0:
				continue

			viz_dict["rewards_applied"].append({"t": viz_counter, "reward":r})
			if d or viz_counter > episode_count_max:
				break
		episode_rewards.append(episode_reward/episode_step_count)
		episode_throughputs.append(episode_throughput/episode_step_count)

		# Plotting periodically
		# Periodically save gifs of episodes, model parameters, and summary statistics.
		plot_max = 35
		if episode_count % 5 == 0 and do_figs:
			rew_app_arr = []; desc = ""; rew_app_recent = 0
			imageio_images = []
			for i,episode in enumerate(viz_dict["frames"]):
				fig,ax_fig = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
				plt.subplots_adjust(hspace=1.2, wspace=1.2)
				# Show state of tx-rx pair
				img = ax_fig[0][0].imshow(episode,vmin=0,vmax=plot_max); ax_fig[0][0].set_title("TX-RX Pair, SNR: %.1f dB"%sense_jammer.env.tx_rx_pair.snr)
				cb = fig.colorbar(img,ax=ax_fig[0][0],ticks=[0,threshold,plot_max])
				cb.ax.set_yticklabels(['0','thresh=%.1f'%threshold,'35+'])
				# show our jamming pattern
				img = ax_fig[0][1].imshow(viz_dict["jammer_frames"][i],vmin=0,vmax=plot_max); ax_fig[0][1].set_title("Jammer " + viz_dict["descriptions"][i])
				cb = fig.colorbar(img,ax=ax_fig[0][1],ticks=[0,threshold,plot_max])
				cb.ax.set_yticklabels(['0','thresh=%.1f'%threshold,'35+'])
				# show rewards received for these actions
				if i in viz_dict["times_of_application"]:
					# an experience was saved for the buffer at this time
					# show the reward that will be applied for this experience
					el_num = viz_dict["times_of_application"].index(i)
					try:
						reward_applied = viz_dict["rewards_applied"][el_num]
						desc = viz_dict["descriptions_applied"][el_num]
						rew_app_recent =  reward_applied["reward"]
						rew_app_arr.append([i, reward_applied["reward"]])
					except IndexError:
						break
				ax_fig[0][2].set_title("Reward Applications - %.2f, %s"%(rew_app_recent,desc))
				tmp_arr = np.array(rew_app_arr)
				if rew_app_arr != []:
					img = ax_fig[0][2].plot(tmp_arr[:,0], tmp_arr[:,1],c='b',marker='o'); ax_fig[0][2].set_xlim([0,len(viz_dict["frames"])]); ax_fig[0][2].set_ylim([0,1.5])
				else:
					img = ax_fig[0][2].plot(tmp_arr,c='b',marker='o'); ax_fig[0][2].set_xlim([0,len(viz_dict["frames"])]); ax_fig[0][2].set_ylim([0,1.5])

				# show power
				ax_fig[1][0].set_title("Jammer Power, Mean: %.1f"%np.mean(viz_dict["powers"][0:i]))
				ax_fig[1][0].plot(viz_dict["powers"][0:i]); ax_fig[1][0].set_xlim([0,len(viz_dict["frames"])]); ax_fig[1][0].set_ylim([0,plot_max])
				# show reward received
				ax_fig[1][1].set_title("Reward (Not Applied)")
				ax_fig[1][1].plot(viz_dict["rewards"][0:i]); ax_fig[1][1].set_xlim([0,len(viz_dict["frames"])]); ax_fig[1][1].set_ylim([0,1.5])
				plt.savefig("tmp.png")
				imageio_images.append(imageio.imread("tmp.png"))
				plt.clf()
				plt.close('all')
			imageio.mimsave('./sense_frames/image'+str(episode_count)+'.gif', imageio_images, duration=1)
		if episode_count % 250 == 0:
			print("Mean reward per action: %.2f"%np.mean(episode_rewards))
			print("Mean throughput per action: %.2f"%np.mean(episode_throughputs))

		episode_count += 1
