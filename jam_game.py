import numpy as np, json
from constants import *
class TX_RX_Pair(object):
	def __init__(self):
		self.is_jammed = False
		self.was_jammed = False
		self.status = "data" # random initial status
		with open(JAM_GAME_CONFIG, 'r') as f:
			self.config = json.load(f)
		self.alpha = self.config["power_discrepancy_factor"]
		# tracks how many timesteps left until it can make another action
		self.action_timer = 0

		# controls variables relating to the throughput at a certain time
		self.throughput_reports = np.zeros(1000)
		self.throughput = 0
		self.throughput_square_window_history = 5
		self.alpha_inc = .4 # moving average of throughput
		self.alpha_dec = .3
		self.t = 0 # tracks current timestep
		# how much "throughput" is achieved assuming 1 data packet per timestep
		self.throughput_unit_per_packet = 1

	def decide_jam(self, state):
		"""Uses a probabilistic model to determine whether or not the tx-rx pair is getting jammed."""

		# TODO - look at it from a BER - SNR standpoint

		# heuristic choice - sigmoid
		sensitivity = 1.0 # controls how sharply the sigmoid transitions
		p = 1.0 / (1.0 + np.exp( -sensitivity * (state[self.on_channel,0] - self.jam_threshold + self.leeway)))
		if np.random.random() < p:
			self.is_jammed = True
		else:
			self.is_jammed = False

	def decrement_throughput(self,dec_type = "basic"):
		"""Time is rolling by - decrease our throughput."""
		if dec_type == "moving_average":
			self.throughput = self.alpha_dec * self.throughput
		elif dec_type == "basic":
			self.throughput_reports[self.t] = 0
			self.throughput = np.mean(self.throughput_reports[self.t-self.throughput_square_window_history:self.t])

	def get_status(self):
		return self.status

	def increment_throughput(self, inc_type="basic"):
		"""A data packet was succesfully transmitted/received - update the throughput."""
		if inc_type == "moving_average":
			# Moving average
			self.throughput = self.alpha_inc * self.throughput + (1 - self.alpha_inc) * \
				self.throughput_unit_per_packet
		elif inc_type == "basic":
			self.throughput_reports[self.t] = self.throughput_unit_per_packet
			self.throughput = np.mean(self.throughput_reports[self.t-self.throughput_square_window_history:self.t])

	def new_episode(self):
		""" Reset properties that change based on the episode, that are unlikely to change through the course of it."""

		# snr shouldn't change too much in a simple case if we assume motionless users
		self.snr = np.random.choice(self.snr_choices)
		# favor low power
		#self.snr = np.random.uniform(low=self.snr_min,high=self.snr_max) 
		# sample evenly along the linear SNR scale
		#self.snr = 10 * np.log10(np.random.uniform(low=np.power(10,self.snr_min/10.0),high=np.power(10,self.snr_max/10.0)))

	def update_status(self):
		"""Updates the status of the tx-rx pair."""

		self.status = self.previous_action

		# First, check if we are busy
		if self.action_timer >= 0:
			return

		# 1. Hop if we want to hop
		# 2. Update current status to either hopping or transmitting
		if self.dbg:
			print("TX-RX - Updating status to " + self.previous_action)
		if self.previous_action == "hop":
			# hop to a new channel - its assumed the tx-rx pair follow some pre-agreed upon pr sequence
			on_channel = np.random.randint(0,self.n_channels)
			if on_channel == self.on_channel:
				# make sure its a new channel
				self.on_channel = np.mod(self.on_channel+1,self.n_channels)
			else:
				self.on_channel = on_channel
				self.action_timer = self.protocol["hop"]["length"]

class Control_Protocol_Pair(TX_RX_Pair):
	def __init__(self, n_channels,snr_min,snr_max,dbg=False):
		super(Control_Protocol_Pair, self).__init__()
		self.dbg = dbg
		self.n_channels = n_channels
		self.on_channel = np.random.randint(0,n_channels)
		self.possible_states = ["hop", "no_data", "data"]
		self.name = "control"
		# Rows -> statuses, Columns -> actions
		# List of statuses: Hopping + not being jammed, Transmitting Non-Data + not being jammed, Transmitting Data + not being jammed
		#					Hopping + being jammed, Transmitting Non-Data + being jammed, Transmitting Data + being jammed
		# List of actions: Hopping, Transmitting Non_data, Transmitting Data
		# kind of arbitrary probabilities for now
		self.transition_probabilities = [[.001, .001, 1-(.001 + .001)], [.1, .01, 1-(.1 + .01)], [.01, .4, 1-(.01 + .4)],
										[.99, .001, 1-(.99 + .001)],[.99, .001, 1-(.99+.001)],[.99,.001,1-(.99+.001)]]

		# possible SNRs in the scenario
		self.snr_min = snr_min; self.snr_max=snr_max
		# vary scale at which you favor low snr in scenario generation
		base = 5
		self.snr_choices = self.snr_min + (self.snr_max - self.snr_min) / (base-1) * (np.logspace(0,1,num=1000,base=base) - 1)

		self.new_episode()
		self.update_protocol()								

	def update(self, state):
		"""Checks to see if getting jammed and decides what to do."""
		# 1. are we getting jammed?
		# 2. update protocol
		# 3. decide to transmit/hop (don't hop, but you can transmit)

		if self.dbg:
			print("TX-RX - Action Timer: %d"%self.action_timer)

		# increment the timestep
		self.t += 1

		# determines whether or not the tx-rx pair is currently jammed based on interference
		self.decide_jam(state)

		# only select an action if we are not currently performing another action
		if self.action_timer <= 0:
			# protocol is state dependent
			self.update_protocol()

			# See if we were jammed while we were busy
			if self.dbg:
				print("TX-RX Is Jammed: %d, Was Jammed: %d"%(self.is_jammed, self.was_jammed))
			self.is_jammed = self.is_jammed or self.was_jammed
			self.was_jammed = False
			if self.dbg:
				print("TX-RX Is Jammed: %d, Was Jammed: %d"%(self.is_jammed, self.was_jammed))

			# select an action according to the protocol
			action = np.random.choice(range(len(self.protocol.keys())), p=[v["probability"] for k,v in self.protocol.items()])
			tmp = list(self.protocol.keys())
			action = tmp[action]

			if action == "no_data":
				# control packet
				state[self.on_channel,0] += self.protocol["no_data"]["power"]
				self.action_timer = self.protocol["no_data"]["length"]
				if not self.is_jammed:
					self.increment_throughput()

			elif action == "data":
				# data packet
				state[self.on_channel,0] += self.protocol["data"]["power"]
				self.action_timer = self.protocol["data"]["length"]
				if not self.is_jammed:
					self.increment_throughput()
			else:
				# Update the throughput since time is rolling by
				self.decrement_throughput()		

			if self.dbg:
				print("TX-RX - Taking action: " + action)

			self.previous_action = action
		else:
			# remember if we were ever jammed
			# TODO - update this based on some throughput threshold (# of bad packets, etc)
			self.was_jammed = self.is_jammed or self.was_jammed
			if self.dbg:
				print("TX-RX - Continuing to take action: " + self.previous_action)
			# Continue performing the last action
			if self.status == "no_data":
				# control packet
				state[self.on_channel,0] += self.protocol["no_data"]["power"]
				if not self.is_jammed:
					self.increment_throughput()
			elif self.status == "data":
				state[self.on_channel,0] += self.protocol["data"]["power"]
				if not self.is_jammed:
					self.increment_throughput()
			else:
				# Update the throughput since time is rolling by
				self.decrement_throughput()					

		self.action_timer += -1
		
		return state

	def update_protocol(self):
		"""Updates the probabilities/characteristics of the protocol according to the environment."""
		state_probabilities = int(self.possible_states.index(self.status)) + len(self.possible_states) * int(self.is_jammed)
		state_probabilities = self.transition_probabilities[state_probabilities][:]

		# get linear power, assuming N_o = 1, BPSK, etc..
		power = np.power(10,self.snr/10)

		self.protocol = {
			"hop": {
				"probability": state_probabilities[0],
				"length": 0 + 5* (int(self.is_jammed and self.status == "no_data")),
				"power": 0
			},
			"no_data": {
				"probability": state_probabilities[1], 
				"length": np.random.randint(0,2) + 1,
				"power": power # more important so higher power, they just need some sort of distinct feature
			},
			"data": {
				"probability": state_probabilities[2],
				"length": np.random.randint(1,4),
				"power": power / self.alpha
			}
		}
		# jam threshold is that it needs to be at least as loud as the tx-rx pair
		self.jam_threshold = np.max([v["power"] for k,v in self.protocol.items()]) 
		self.leeway = np.minimum(self.jam_threshold / 2, 3.0)	

class ACK_Protocol_Pair(TX_RX_Pair):
	def __init__(self,n_channels,snr_min,snr_max,dbg=False):
		super(ACK_Protocol_Pair, self).__init__()
		self.dbg = dbg
		self.n_channels = n_channels
		self.on_channel = np.random.randint(0,n_channels)
		self.name = "ack"
		self.possible_states = ["hop", "ack", "data","wait_ack","re_data"]
		# Rows -> statuses, Columns -> actions
		# List of statuses: Not being jammed and (Hopping, ACKING, Tx-Data, Waiting for ACK, Re-Tx-Data)
		#					Being jammed and (Hopping, ACKING, Tx-Data, Waiting for ACK, Re-Tx-Data)
		# List of actions: Hopping, ACKING, Tx-Data, Waiting for ACK, Re-Tx-Data
		# kind of arbitrary probabilities for now
		self.transition_probabilities = [[.001, 0, 1-(.001 + 0), 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [.4, 0, 1-(.4), 0, 0], [0,0,0,1,0],
										[.99, 0, 1 - (.99), 0, 0],[0, 0, 0, 1, 0],[1,0,0,0,0],[.1, 0, 0, 0, 1 - (.1)], [1,0,0,0,0]]

		# possible SNRs in the scenario
		self.snr_min = snr_min; self.snr_max=snr_max
		# vary scale at which you favor low snr in scenario generation
		base = 5
		self.snr_choices = self.snr_min + (self.snr_max - self.snr_min) / (base-1) * (np.logspace(0,1,num=1000,base=base) - 1)

		self.new_episode()
		self.update_protocol()								

	def update(self, state):
		"""Checks to see if getting jammed and decides what to do."""
		# 1. are we getting jammed?
		# 2. update protocol
		# 3. decide to transmit/hop (don't hop, but you can transmit)

		if self.dbg:
			print("TX-RX - Action Timer: %d"%self.action_timer)

		# increment the timestep
		self.t += 1

		# determines whether or not the tx-rx pair is currently jammed based on interference
		self.decide_jam(state)

		# only select an action if we are not currently performing another action
		if self.action_timer <= 0:
			# protocol is state dependent
			self.update_protocol()

			# See if we were jammed while we were busy
			self.is_jammed = self.is_jammed or self.was_jammed
			self.was_jammed = False

			# select an action according to the protocol
			action = np.random.choice(range(len(self.protocol.keys())), p=[v["probability"] for k,v in self.protocol.items()])
			tmp = list(self.protocol.keys())
			action = tmp[action]
			state[self.on_channel,0] += self.protocol[action]["power"]
			self.action_timer = self.protocol[action]["length"]

			if action == "ack":
				# ack packet
				if not self.is_jammed:
					self.increment_throughput()
			elif action == "data":
				# data packet
				if not self.is_jammed:
					self.increment_throughput()
			else:
				# Update the throughput since time is rolling by
				self.decrement_throughput()		

			if self.dbg:
				print("TX-RX - Taking action: " + action)

			self.previous_action = action
		else:
			# remember if we were ever jammed
			# TODO - update this based on some throughput threshold (# of bad packets, etc)
			self.was_jammed = self.is_jammed or self.was_jammed
			if self.dbg:
				print("TX-RX - Continuing to take action: " + self.previous_action)
			# Continue performing the last action
			state[self.on_channel,0] += self.protocol[self.status]["power"]
			if self.status == "ack":
				# ack packet
				if not self.is_jammed:
					self.increment_throughput()
			elif self.status == "data":
				# data packet
				if not self.is_jammed:
					self.increment_throughput()
			else:
				# Update the throughput since time is rolling by
				self.decrement_throughput()					

		self.action_timer += -1
		
		return state

	def update_protocol(self):
		"""Updates the probabilities/characteristics of the protocol according to the environment."""
		state_probabilities = int(self.possible_states.index(self.status)) + len(self.possible_states) * int(self.is_jammed)
		state_probabilities = self.transition_probabilities[state_probabilities][:]

		# get linear power, assuming N_o = 1, BPSK, etc..
		power = np.power(10,self.snr/10)

		self.protocol = {
			"hop": {
				"probability": state_probabilities[0],
				"length": 0,
				"power": 0
			},
			"ack": {
				"probability": state_probabilities[1], 
				"length": np.random.randint(2,3),
				"power": power # more important so higher power, they just need some sort of distinct feature
			},
			"data": {
				"probability": state_probabilities[2],
				"length": np.random.randint(1,4),
				"power": power / self.alpha
			},
			"wait_ack": {
				"probability": state_probabilities[3],
				"length": 2,
				"power": 0
			},
			"re_data": {
				"probability": state_probabilities[4],
				"length": np.random.randint(1,4),
				"power": power / self.alpha
			}
		}
		# jam threshold is that it needs to be at least as loud as the tx-rx pair
		self.jam_threshold = np.max([v["power"] for k,v in self.protocol.items()]) 
		self.leeway = np.minimum(self.jam_threshold / 2, 3.0)	

class Jam_Game:
	def __init__(self, n_channels, n_powers, n_memory, protocol="control", dbg=False):
		self.dbg = dbg
		# Arbitrary parameters
		self.n_channels = n_channels; self.n_powers = n_powers
		self.n_memory = n_memory # size of time dimension in state
		self.jam_props = {"power": 10, "length": 1}
		self.blank_col = np.zeros((1,self.n_channels))
		self.n_jams_finished = 5
		self.protocol = protocol

		# throughput based, rather than categorical based reward
		self.rewards_base = 1.0
		self.rewards_power = self.rewards_base / 3

		self.snr_min=12
		self.snr_max=15
		# cooldown after selecting to jam on a certain channel
		self.length_jam = 3
		# tracks how many timesteps left until it can make another action
		self.action_timer = 0

		self.power_map = {}
		for el in range(n_powers):
			# right now 3.16 -> 31.62 for tx-rx pair
			self.power_map[el] = 3 + 3.3 * el

	def account_for_own_power(self, channel):
		"""Subtract our own transmissions from the state so as to not confuse the agent."""
		self.state[channel,0] -= self.jam_props["power"]

	def calculate_reward(self, channel, decision_type="categorical"):
		"""Determines the reward based on the action and the current state."""

		# Determine the reward string - for visualizations
		reward_str = "miss"
		if channel-1 == self.tx_rx_pair.on_channel:
			if self.protocol == "control":
				if self.tx_rx_pair.get_status() == "no_data" and self.tx_rx_pair.is_jammed:
					reward_str = "jam - ctrl"
				elif self.tx_rx_pair.get_status() == "data" and self.tx_rx_pair.is_jammed: 
					# data packet - whooopdy doo
					reward_str = "jam - data"
			elif self.protocol == "ack":
				if self.tx_rx_pair.get_status() == "ack" and self.tx_rx_pair.is_jammed:
					reward_str = "jam - ACK"
				elif (self.tx_rx_pair.get_status() == "data" or self.tx_rx_pair.get_status() == "re_data") and self.tx_rx_pair.is_jammed: 
					# data packet - whooopdy doo
					reward_str = "jam - data"	
		elif channel == 0:
			reward_str = "waiting"

		if decision_type == "categorical":
			# direct state->reward mapping
			if channel-1 == self.tx_rx_pair.on_channel:
				if self.tx_rx_pair.get_status() == "no_data" and self.tx_rx_pair.is_jammed:
					reward = self.rewards["non_data"]
					reward_str = "jam - ctrl"
				elif self.tx_rx_pair.get_status() == "data" and self.tx_rx_pair.is_jammed: 
					# data packet - whooopdy doo
					reward = self.rewards["data"]
					reward_str = "jam - data"
				else:
					reward = 0 # need more power
				# if reward != 0:
					# # account for wasted power
					# if self.tx_rx_pair.get_status() != "hop":
					# 	reward += np.minimum(1.0 / (np.abs(self.jam_props["power"] - self.tx_rx_pair.jam_threshold)  * .1), 3)
			elif channel == 0:
				if self.tx_rx_pair.get_status() == "hop":
					reward = self.rewards["wait"]
				else:
					# missing a transmission when the tx-rx pair is visible in the spectrum
					reward = 0
				reward_str = "waiting"
			else:
				reward = self.rewards["miss"]
		elif decision_type == "throughput":
			# reward is proportional to the throughput achieved by 
			# the tx-rx pair
			reward = self.rewards_base - self.tx_rx_pair.throughput
			if channel == 0:
				# if we are waiting, add a little bonus for not using power
				reward += self.rewards_power
		else:
			raise ValueError("Decision type %s for reward function not implemented.")

		return reward, reward_str 

	def continue_last_action(self):
		"""Does the last action, and rolls the environment so as to simulate the
			time required to make actions."""
		[channel,power] = self.last_action
		# make the action
		if channel != 0:
			if self.dbg:
				print("JAMMER - Continuing to jam on channel %d"%(channel-1))
			self.jam_on_channel(channel-1)
		else:
			# 0 power if choosing to not transmit
			print("JAMMER - Continuing to not transmit.")
			self.jam_props["power"] = 0
		self.state = self.tx_rx_pair.update(self.state)
		self.state_raw[:,0] = np.copy(self.state[:,0])
		if channel != 0:
			# employ self-cancellation at the receiver
			self.account_for_own_power(channel-1)

		self.roll_state() # rolls the spec-gram one time step 
		if self.dbg:
			print("Is currently jammed: %d"%self.tx_rx_pair.is_jammed)
		self.tx_rx_pair.update_status() 
		self.t += 1
		self.action_timer += -1

	def new_episode(self):
		# Generate a new scenario
		if self.protocol == "control":
			self.tx_rx_pair = Control_Protocol_Pair(self.n_channels,self.snr_min,self.snr_max,self.dbg)
		elif self.protocol == "ack":
			self.tx_rx_pair = ACK_Protocol_Pair(self.n_channels,self.snr_min,self.snr_max,self.dbg)
		else:
			raise ValueError("Unrecognized protocol: %s, exiting."%self.protocol)
			exit(0)
		noise = np.random.normal(size=(self.n_channels,self.n_memory))
		self.state = np.zeros((self.n_channels, self.n_memory)) + noise
		# roll to a random point in the tx-rx correspondence so we don't always start at 0
		for i in range(self.n_memory):
			self.state = self.tx_rx_pair.update(self.state)
			self.state_raw = np.copy(self.state)
			self.roll_state()
			self.tx_rx_pair.update_status()
		self.t_last_jam = 0
		self.t = 0
		self.n_jams = 0

	def get_state(self):
		""" Getter method for the state. """
		return self.state

	def is_episode_finished(self):
		"""Determines whether the current episode is done or not."""
		return self.tx_rx_pair.is_jammed and (self.n_jams == self.n_jams_finished)

	def jam_on_channel(self, channel):
		# TODO how does the length of jam factor into this
		self.state[channel,0] += self.jam_props["power"]

	def make_action(self, channel, power):
		"""Makes an action, updates the current state based on the action made and returns a reward for the action."""

		# Check to make sure we are not in a cooldown phase
		if self.action_timer > 0:
			# functions that are required to make the scenario progress
			self.continue_last_action()
			return None, None
		# record the last action in case we are making an action that has a cooldown
		self.last_action = [channel,power]

		self.jam_props["power"] = self.power_map[9]#self.power_map[power]
		if self.dbg:
			print("JAMMER - Taking action: %d"%channel)
		# make the action
		if channel != 0:
			self.jam_on_channel(channel-1)
			# number of timesteps you must jam for
			self.action_timer = self.length_jam
		else:
			# 0 power if choosing to not transmit
			self.jam_props["power"] = 0
		self.state = self.tx_rx_pair.update(self.state)
		# un-cancelled view of the spectrum (including our transmissions)
		self.state_raw[:,0] = np.copy(self.state[:,0])
		if channel != 0:
			# employ self-cancellation at the receiver
			self.account_for_own_power(channel-1)
		
		if self.tx_rx_pair.is_jammed:
			self.t_last_jam = self.t
			self.n_jams += 1

		self.roll_state()

		reward, reward_str = self.calculate_reward(channel, decision_type="throughput")

		self.tx_rx_pair.update_status()
		self.t += 1

		# self.reward tracks how good the action was over time
		if self.dbg:
			print("Is currently jammed: %d"%self.tx_rx_pair.is_jammed)
			print("JAMMER - Giving total reward of %d"%reward)
		return reward, reward_str

	def roll_state(self):
		# roll the state 1 time step
		# N_o = 1
		noise = np.random.normal(size=(1,self.n_memory))

		self.state = np.insert(self.state[:,0:self.n_memory-1],0,self.blank_col + noise,axis=1)
		self.state_raw = np.insert(self.state_raw[:,0:self.n_memory-1],0,self.blank_col + noise,axis=1)
