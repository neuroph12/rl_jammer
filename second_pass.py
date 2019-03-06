# possible modes:
# basic_train - trains a simple model that recognizes it should jam things with high power
# normal - trains a model 
# gather_xp - saves experiences without updating weights
# auto_run - gathers experience and trains in a way that gathers a bunch of xp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--auto_run', action='store', default=False)
parser.add_argument('--mode', action='store', default='normal')
parser.add_argument('--protocol', action='store', default='control')
parser.add_argument('--rn', action='store', default=False)
parser.add_argument('--pkl_uid', action='store', default='default_pkl_id')
args = parser.parse_args()
PROTOCOL = args.protocol #possiblities are ack, control
AUTO_RUN = int(args.auto_run)
RUNTIME_MODE = args.mode
PKL_UID = args.pkl_uid
USE_REWARD_NET = int(args.rn) # true or false - use learned reward net
print("\n\n\nYou are running this agent using mode: %s, protocol: %s, auto run: %d, and are using reward net: %d. \n\n\n"%(RUNTIME_MODE, PROTOCOL, AUTO_RUN, USE_REWARD_NET))

if AUTO_RUN and not USE_REWARD_NET:
    RUNTIME_MODE = "gather_xp"
else:
    RUNTIME_MODE = RUNTIME_MODE

import threading, os, multiprocessing, scipy.signal, imageio
import numpy as np, pickle, matplotlib
np.set_printoptions(suppress=True)
matplotlib.use('Agg')
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 7}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper import *
from time import sleep, time
from subprocess import *

from jam_game import Jam_Game
from reward_function import Reward_Function
from constants import *
from importlib import reload

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(s):
    s = np.reshape(s,[np.prod(s.shape)]) / 5.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    np.random.seed(4)
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class Reward_Network():
    def __init__(self, s_size):
        # Relating to the reward function
        self.raw_state = tf.placeholder(shape=[None, 10, 10],dtype=tf.float32)
        # Infer class from model
        model = Reward_Function(tf.reshape(self.raw_state, shape=[-1, 10, 10]))
        self.reward_logits = model.logits
        self.reward_prediction = tf.argmax(input=self.reward_logits, axis=1)

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,10,10,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, #looking for where the fucker is currently transmitting
                inputs=self.imageIn,num_outputs=10,
                kernel_size=[1,4],stride=[1,1],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv1), 128, activation_fn=tf.nn.elu)

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden,a_size, # a size is n_channels * n_power choices
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy,1e-5,1e5)))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
class Worker():
    def __init__(self,game,name,s_size,a_size,trainer,model_path,global_episodes,rn):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_throughputs = []
        self.episode_lengths = []
        self.episode_mean_values = []

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.local_RN = rn
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.actions = np.identity(a_size,dtype=bool).tolist()
        # Environment set up
        self.env = game
        # variables relating to using the learned reward function
        self.reward_net = USE_REWARD_NET # use the learned reward function ?
        self.reward_miss = -1
        self.classes = np.insert(np.linspace(0,self.env.rewards_base + self.env.rewards_power,n_classes-1), 0, self.reward_miss)
        # Dictionary responsible for holding reward network experiences
        self.state_rewards = {}
        self.state_rewards[0,0] = []
        # Confusion matrix of reward network
        self.misses = np.zeros((len(self.classes),len(self.classes)))
        self.RUNTIME_MODE = RUNTIME_MODE
        

    def add_to_state_reward_pair(self, r, s):
        # Append state-reward pairs to data structure for offline learning 
        # Of the reward function
        sr_r = np.argmin(np.abs(r-self.classes))
        try:
            self.state_rewards[sr_r,0]
        except KeyError:
            self.state_rewards[sr_r,0] = []
        self.state_rewards[sr_r,0].append(s)
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        smr=0; v_l = 0; p_l = 0; e_l = 0; g_n = 0; v_n = 0; stopped_training = 20e3
        #print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            summary_file = "train_%s/%s"%(self.name[-1], PKL_UID)
            if not os.path.exists(summary_file):
                os.makedirs(summary_file)        
            self.summary_writer = tf.summary.FileWriter(summary_file,
                graph=tf.get_default_graph())         
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []; 

                # Viz variables
                self.viz_dict  = {
                    "frames": [], # holds spectral images
                    "jammer_frames": [],
                    "rewards": [], # holds all rewards received, even when stuck
                    "powers": [], # holds power decisions
                    "descriptions": [], # holds descriptions of jammer action
                    "times_of_application": [],
                    "descriptions_applied": [],
                    "rewards_applied": [], # holds all rewards applied
                    "rn_rewards_applied": []
                }
                first_episode_state = np.zeros((self.env.n_channels,self.env.n_memory))
                viz_counter = -1
                #
                episode_reward = 0
                episode_throughput = 0
                episode_step_count = 0
                d = False
                # Get a new frame (state)
                self.env.new_episode()
                s = self.env.get_state()
                self.viz_dict["frames"].append(s)
                s = process_frame(s) #flatten
        
                experience_buffer_last = None; first_time = True
                np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                while not self.env.is_episode_finished():
                    viz_counter += 1
                    if experience_buffer_last is not None:
                        first_time = False
                    #Take an action using probabilities from policy network output.
                    action_dist,v = sess.run([self.local_AC.policy,self.local_AC.value], 
                        feed_dict={self.local_AC.inputs:[s]})
                    control = np.random.choice(action_dist[0],p=action_dist[0])
                    control = np.argmax(action_dist == control)
                    # chan_action = int(control / self.env.n_channels)
                    # pow_action = int(control % self.env.n_powers)
                    chan_action = control
                    pow_action = 0

                    # Update the environment based on the state, and calculate the reward
                    # r is reward, r_str describes what happened for viz
                    r,r_str = self.env.make_action(chan_action, pow_action)
                    if r is None:
                        self.viz_dict["rewards"].append((self.env.calculate_reward(chan_action,"throughput"))[0])
                        self.viz_dict["descriptions"].append("stuck")
                        # We are waiting for an action to complete
                        # visualization
                        s1 = self.env.get_state()
                        self.viz_dict["frames"].append(s1)
                        s1 = process_frame(s1) #flatten
                        if self.env.last_action[0] != 0:
                            first_episode_state[self.env.last_action[0]-1,0] = self.env.jam_props["power"];
                        self.viz_dict["jammer_frames"].append(first_episode_state)
                        self.viz_dict["powers"].append(self.env.power_map[pow_action])
                        first_episode_state = np.insert(first_episode_state[:,0:self.env.n_memory-  1],0,np.zeros((1,self.env.n_channels)),axis=1) 
                        s = s1
                        continue

                    #variables that we only copy over if we aren't skipping frames    
                    self.viz_dict["descriptions"].append(r_str)
                    self.viz_dict["rewards"].append(r)
                    ## # Save for visualizations 
                    if chan_action != 0:
                        first_episode_state[chan_action-1,0] = self.env.jam_props["power"];
                    self.viz_dict["jammer_frames"].append(first_episode_state)
                    self.viz_dict["powers"].append(self.env.power_map[pow_action])
                    first_episode_state = np.insert(first_episode_state[:,0:self.env.n_memory-  1],0,np.zeros((1,self.env.n_channels)),axis=1) 
                    ### 

                    ##### REWARDS NET
                    if self.reward_net:
                        r_actual = r
                        if r_str == "miss":
                            r_actual = self.reward_miss
                        what_choice_should_be = np.argmin(np.abs(r_actual-self.classes))
                        logits, rew = sess.run([self.local_RN.reward_logits,self.local_RN.reward_prediction], 
                            feed_dict={self.local_RN.raw_state: [np.array(np.random.uniform(-4,4,(10,10)) +self.env.state_raw, dtype=np.float32)]})
                        choice = rew[0]
                        r = self.classes[choice]
                        self.misses[what_choice_should_be,choice] += 1
                    ##### 


                    # check to see if the episode is finished (probably not applicable)
                    d = self.env.is_episode_finished() or episode_step_count == max_episode_length - 1

                    if not d:
                        s1 = self.env.get_state()
                        self.viz_dict["frames"].append(s1)
                        s1 = process_frame(s1) #flatten
                    else:
                        s1 = s


                    if RUNTIME_MODE == "basic_train":
                        if r_str == "miss" or r_str == "waiting":
                            r = 0
                        else:
                            r = 1
                        episode_buffer.append([s,control, r, s1, d, v[0,0]])
                        episode_values.append(v[0,0])
                    else:
                        ##### Check for Miss
                        if (r_str == "miss" and not self.reward_net) or (r == self.reward_miss and self.reward_net): # this is really bad and should never happen - handle it differently
                            r = self.reward_miss
                            episode_buffer.append([s,control, r, s1, d, v[0,0]])
                            episode_values.append(v[0,0])
                            episode_reward += r
                            s = s1                    
                            total_steps += 1
                            episode_step_count += 1
                            self.add_to_state_reward_pair(r, self.env.state_raw.flatten())
                            break
                        ##### 

                        self.viz_dict["times_of_application"].append(viz_counter)
                        self.viz_dict["descriptions_applied"].append(r_str)
                        # We are delaying our reward applications by 1 time step
                        if experience_buffer_last is not None:
                            experience_buffer_last_tmp = experience_buffer_last.copy()
                        experience_buffer_last = [s,control, s1, d, v[0,0]]

                        self.add_to_state_reward_pair(r, self.env.state_raw.flatten())

                        if first_time:
                            continue

                        self.viz_dict["rewards_applied"].append({"t": viz_counter, "reward":r})
                        if self.reward_net:
                            self.viz_dict["rn_rewards_applied"].append({"t": viz_counter, "reward": r_actual})
                        episode_buffer.append([experience_buffer_last_tmp[0],
                            experience_buffer_last_tmp[1], r, experience_buffer_last_tmp[2],
                            experience_buffer_last_tmp[3], experience_buffer_last_tmp[4]])
                        
                        episode_values.append(experience_buffer_last_tmp[4])

                    episode_reward += r
                    episode_throughput += self.env.tx_rx_pair.throughput
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and not d and episode_step_count != max_episode_length - 1 and self.RUNTIME_MODE != "gather_xp":
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_throughputs.append(episode_throughput/episode_step_count)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0 and self.RUNTIME_MODE != "gather_xp":
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
               
                if episode_count % 5 == 0 and episode_count != 0:   
                    if AUTO_RUN and not self.reward_net:
                        if episode_count > 5000 and self.RUNTIME_MODE == "gather_xp" and smr < EARLY_STOP_REWARD:
                            print("Done gathering bad experience, changing to train on %s."%self.name)
                            self.RUNTIME_MODE = "normal"
                    if self.name == 'worker_10' and episode_count % 250 == 0:
                        try:
                            self.save_frames_gif(episode_count)
                        except:
                            print("%s: Problem saving frames - index error. Too lazy to fix it right now."%self.name)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("%s: Saved Model for iteration: %d."%(self.name, episode_count))
                    if episode_count % 250 == 0:
                        #print("%s: Mean reward per action: %.2f"%(self.name, np.mean(np.divide(self.episode_rewards[-50:],self.episode_lengths[-50:]))))
                        if self.reward_net:
                            print("%s: Confusion matrix of reward net: \n"%self.name)
                            print(self.misses)
                        if not self.reward_net: 
                            if self.name == "worker_0": # only saving experiences on one thread bc of dumb random number generation
                                # save state-reward pairs for offline reward function learning
                                with open("state_rewards/trainer-%s-%s-%d.pkl"%(self.env.tx_rx_pair.name, PKL_UID, episode_count),'wb') as f:
                                    pickle.dump(self.state_rewards, f)
                            self.state_rewards = {}
                    
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    smr = .96 * smr + .04 * np.mean(self.episode_rewards[-5:]) # smoothed mean reward
                    if smr > EARLY_STOP_REWARD:
                        if self.RUNTIME_MODE == "normal":
                            stopped_training = episode_count
                            self.RUNTIME_MODE = "gather_xp"
                            if self.name != "worker_0":
                                print("Gathering xp on a different worker... exiting.")
                                exit(0)
                            else:
                                print("Done training, switching to gather xp on %s"%self.name)
                        if episode_count > stopped_training + 1e3 or self.reward_net:
                            print("\n\nReached a mean_reward of %.3f, stopping this round because the model is assumed to have been trained well.\n\n"%mean_reward)
                            exit(0)
                   
                    mean_throughput = np.mean(self.episode_throughputs[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Throughput', simple_value=float(mean_throughput))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    if self.RUNTIME_MODE != "gather_xp":
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

    def save_frames_gif(self,episode_count):
        plot_max = 35; reward_bounds = [-1.1, 1.5]
        rew_app_arr = []; desc = ""; rew_app_recent = 0
        if self.reward_net:
            rn_rew_app_arr = []; rn_rew_app_recent = 0
        imageio_images = []
        for i,episode in enumerate(self.viz_dict["frames"]):
            fig,ax_fig = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
            plt.subplots_adjust(hspace=1.2, wspace=1.2)
            # Show state of tx-rx pair
            img = ax_fig[0][0].imshow(episode,vmin=0,vmax=plot_max); ax_fig[0][0].set_title("TX-RX Pair, SNR: %.1f dB"%self.env.tx_rx_pair.snr)
            cb = fig.colorbar(img,ax=ax_fig[0][0],ticks=[0,self.env.tx_rx_pair.jam_threshold,plot_max])
            cb.ax.set_yticklabels(['0','thresh=%.1f'%self.env.tx_rx_pair.jam_threshold,'35+'])
            # show our jamming pattern
            img = ax_fig[0][1].imshow(self.viz_dict["jammer_frames"][i],vmin=0,vmax=plot_max); ax_fig[0][1].set_title("Jammer " + self.viz_dict["descriptions"][i])
            cb = fig.colorbar(img,ax=ax_fig[0][1],ticks=[0,self.env.tx_rx_pair.jam_threshold,plot_max])
            cb.ax.set_yticklabels(['0','thresh=%.1f'%self.env.tx_rx_pair.jam_threshold,'35+'])
            # show rewards received for these actions
            if i in self.viz_dict["times_of_application"]:
                # an experience was saved for the buffer at this time
                # show the reward that will be applied for this experience
                el_num = self.viz_dict["times_of_application"].index(i)
                try:
                    reward_applied = self.viz_dict["rewards_applied"][el_num]
                    desc = self.viz_dict["descriptions_applied"][el_num]
                    rew_app_recent =  reward_applied["reward"]
                    rew_app_arr.append([i, reward_applied["reward"]])
                except IndexError:
                    break
                if self.reward_net:
                    try:
                        rn_reward_applied = self.viz_dict["rn_rewards_applied"][el_num]
                        rn_rew_app_recent =  rn_reward_applied["reward"]
                        rn_rew_app_arr.append([i, rn_reward_applied["reward"]])
                    except IndexError:
                        break

            ax_fig[0][2].set_title("Reward Applications - %.2f, %s"%(rew_app_recent,desc))
            tmp_arr = np.array(rew_app_arr)
            if rew_app_arr != []:
                img = ax_fig[0][2].plot(tmp_arr[:,0], tmp_arr[:,1],c='b',marker='o'); ax_fig[0][2].set_xlim([0,len(self.viz_dict["frames"])]); ax_fig[0][2].set_ylim([0,1.5])
            else:
                img = ax_fig[0][2].plot(tmp_arr,c='b',marker='o'); ax_fig[0][2].set_xlim([0,len(self.viz_dict["frames"])]); ax_fig[0][2].set_ylim(reward_bounds)

            # show power
            ax_fig[1][0].set_title("Jammer Power, Mean: %.1f"%np.mean(self.viz_dict["powers"][0:i]))
            ax_fig[1][0].plot(self.viz_dict["powers"][0:i]); ax_fig[1][0].set_xlim([0,len(self.viz_dict["frames"])]); ax_fig[1][0].set_ylim([0,plot_max])
            # show reward received
            ax_fig[1][1].set_title("Reward (Not Applied)")
            ax_fig[1][1].plot(self.viz_dict["rewards"][0:i]); ax_fig[1][1].set_xlim([0,len(self.viz_dict["frames"])]); ax_fig[1][1].set_ylim(reward_bounds)

            if self.reward_net:
                # show actual rewards, in case of reward net
                ax_fig[1][2].set_title("Actual Rewards (Labels) - %.2f, %s"%(rn_rew_app_recent,desc))
                tmp_arr = np.array(rn_rew_app_arr)
                if rn_rew_app_arr != []:
                    img = ax_fig[1][2].plot(tmp_arr[:,0], tmp_arr[:,1],c='b',marker='o'); ax_fig[1][2].set_xlim([0,len(self.viz_dict["frames"])]); ax_fig[1][2].set_ylim([0,1.5])
                else:
                    img = ax_fig[1][2].plot(tmp_arr,c='b',marker='o'); ax_fig[1][2].set_xlim([0,len(self.viz_dict["frames"])]); ax_fig[1][2].set_ylim(reward_bounds)

            # Save the figure so we can add it to the array of images for our gif
            # Note - I don't see an easier way to convert a pyplot to an imageio object
            # Pretty big bottleneck in terms of time spent
            plt.savefig("tmp.png")
            imageio_images.append(imageio.imread("tmp.png"))
            plt.clf()
            plt.close('all')
        imageio.mimsave('./frames/image'+str(episode_count)+'.gif', imageio_images, duration=1)

if __name__ == "__main__":

    load_model = True
    model_path = './jammer_model/%s'%PKL_UID

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        if load_model:
            # copy the basic model to the dir so we have something to start with
            call("cp jammer_model_basic_start/* %s"%model_path, shell=True)

    #Create a directory to save episode playback gifs to
    if os.path.exists("./frames") and not load_model:
        call("rm -rf ./frames", shell=True)
        call("rm -rf ./train_*", shell=True)
        os.makedirs('./frames')

    rn = Reward_Network(s_size)
    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
        if RUNTIME_MODE == "gather_xp" and not AUTO_RUN:
            num_workers = 1
        else:
            num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(Jam_Game(n_channels, power_a_size, n_memory, protocol=PROTOCOL),i,s_size,a_size,trainer,model_path,global_episodes, rn))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)

            reward_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=reward_fn_scope), 
                max_to_keep=100)
            latest_reward_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
            print('Restoring reward model from: ', latest_reward_checkpoint)
            reward_saver.restore(sess, latest_reward_checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
            
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)