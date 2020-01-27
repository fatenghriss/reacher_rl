import torch
from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt


def train(env, agent, n_episodes=1800, max_t=1000):
    """DDPG

    Params
    ======
        env (UnityEnvironment): the environment on which the agent will train
        agent : defines the agent
        n_episodes (int): number of training episodes
        max_t (int): maximum value of repetitions in an episode

    """
    avg_score = []
    scores_deque = deque(maxlen=100)
    scores = np.zeros(num_agents)
    time_steps = 20
    update = 10

    env_info = env.reset(train_mode=True)[brain_name]

    states = env_info.vector_observations

    agent_tuple = {"state_size": state_size, "action_size": action_size, "random_seed": 2, }
    agents = [DDPGAgent(**agent_tuple) for _ in range(num_agents)]
    action = [agent.act(states[i]) for i, agent in enumerate(agents)]

    for i_episode in range(1, n_episodes + 1):
        states = env_info.vector_observations
        for agent in agents:
            agent.reset()

        for t in range(max_t):
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            step_t = zip(agents, states, actions, rewards, next_states, dones)

            for agent, state, action, reward, next_step, done in step_t:
                agent.memory.add(state, action, reward, next_step, done)
                if t % time_steps == 0:
                    agent.step(state, action, reward, next_step, done, update)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score = np.mean(scores)
        avg_score.append(score)
        scores_deque.append(score)
        avg = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg, ), end="\n")

        if np.mean(scores_deque) > 30.:
            print(
                "\r\rEnviroment solved in @ i_episode={i_episode}, w/ avg_score={avg:.2f}\r".format(i_episode=i_episode,
                                                                                                    avg=avg))

            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

            break

    return avg_score
