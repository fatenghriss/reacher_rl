# Report about solving the Reacher environment


In this report, we will discuss the following:
* The learning algorithm
* The network architecture
* The reward plot

## Learning algorithm
In order to solve the reacher environment, we implemented the DDPG (Deep Deterministic Policy Gradient) algorithm.

### Hyperparameters:
The following hyperparameters have been used to train the model:
* BUFFER_SIZE = int(1e6)  # replay buffer size
* BATCH_SIZE = 64         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-4         # learning rate of the actor
* LR_CRITIC = 3e-4        # learning rate of the critic
* WEIGHT_DECAY = 0.0001   # L2 weight decay

### Actor Network architecture:
For the actor network architecture, we decided to go with the following:
* First fully connected layer with input's size = state space size and output's size = 256
* Second fully connected layer with input's size = 256 and the output's size = action space size = 2

### Critic Network Architecture:
For the critic network architecture, we decided to go with the following:
* First fully connected layer with input's size = state space size and output's size = 256
* Second fully connected layer with input's size = 256 + action space size and the output's size = 256
* Third fully connected network with input's size = 256 and the output's size = 128
* Fourth fully connected network with input's size = 128 and the output's size = 1 to map states and actions to Q-values

## Reward plot:
It took 176 episodes to DDPG to solve the environment.
The following is the plot of the rewards

![Rewards per episode](reward_plot.png)

For more information, please refer to the following files:
* [model.py](training/model.py) to get the code for the network architecture
* [ddpg.py](agents/ddpg.py) to get the code for the agent's implementation
* [train.py](training/train.py) to get the code for training
