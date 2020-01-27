# Reacher

This project consists in developing a reinforcement learning algorithm to train a double-jointed arm to move to target locations.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of our agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this version of the environment, we will have one agent. The task is episodic, and in order to solve the environment, our agent must get an average score of +30 over 100 consecutive episodes.

## Installation
First, and in order to have a working environment that is as clean as possible, let's:
1. Create and activate a new environment with Python 3.6
- __Linux__ or __Mac__:
```
conda create --name reacher python=3.6
source activate reacher
```
- __Windows__:
```
conda create --name reacher python=3.6
activate reacher
```
2. Install dependencies:

```
pip install .
```

## Training

In order to train your agent and test your agent, all you have to do is the following:

```
python main.py
```

## Benchmarking

The current agent is able to solve the environment in 176 episodes.
```python
Episode 1	Average Score: 0.63
Episode 2	Average Score: 0.63
Episode 3	Average Score: 0.66
Episode 4	Average Score: 0.67
Episode 5	Average Score: 0.84
Episode 6	Average Score: 0.96
Episode 7	Average Score: 1.04
Episode 8	Average Score: 1.10
Episode 9	Average Score: 1.14
Episode 10	Average Score: 1.18
Episode 11	Average Score: 1.33
Episode 12	Average Score: 1.45
Episode 13	Average Score: 1.58
Episode 14	Average Score: 1.68
Episode 15	Average Score: 1.82
Episode 16	Average Score: 1.93
...
Episode 173	Average Score: 29.45
Episode 174	Average Score: 29.68
Episode 175	Average Score: 29.91
Episode 176	Average Score: 30.13
Enviroment solved in @ i_episode=176, w/ avg_score=30.13
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
