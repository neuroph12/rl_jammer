# reward function learning parameters
PKL_DIRECTORY = "state_rewards"
TFRECORDS_DIRECTORY = "reward_records"
n_classes = 6
learning_rate = .001
checkpoints_dir = "reward_fn_checkpoints"
summaries_dir = "reward_summaries"
dropout = 1.0
n_epochs = 20
batch_size = 128
reward_fn_scope = "REWARD_WEIGHTS"

# Parameters associated with the jam game
n_channels = 10; n_memory = 10
JAM_GAME_CONFIG = "runtime_config.json"
EARLY_STOP_REWARD = 10.0 # mean reward the jammer needs to achieve to declare the model trained (for auto-running)

max_episode_length = 40
gamma = .9 # discount rate for advantage estimation and reward discounting
s_size = n_channels * n_memory 
power_a_size = 10 # how many power levels there are
channel_a_size = n_channels + 1 # Agent can do nothing, or jam on one of the channels
# a_size = channel_a_size (ignoring power control for now)
a_size = channel_a_size

# sense jammer constants
# ack - gets .48 throughput, .55 reward
# control - gets .73 throughput, .33 reward