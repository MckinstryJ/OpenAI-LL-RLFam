import numpy as np
import random
import rl.printer
import gym
import matplotlib.pyplot as plt


class DoubleQL(object):
    Qa = []
    Qb = []
    state_space = None
    action_space = None
    groups = None
    alpha = .1
    gamma = .99
    explore = 1.0
    explore_min = .01
    explore_decay = .001

    def __init__(self, state_space, action_space, group=5):
        self.state_space = state_space
        self.action_space = action_space
        self.groups = group

        shap = [self.groups for i in range(self.state_space)]
        shap.append(3)
        shap = tuple(shap)

        self.Qa = np.zeros(shape=shap)
        self.Qb = np.zeros(shape=shap)

    def continous_2_descrete(self, Q, obs):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """

        location = Q
        for i in range(len(obs)):
            value = int(obs[i] * 100) % self.groups
            location = location[value]

        return location

    def take_action(self, obs):
        """
            Explore if random number is greater than #
            Otherwise, return action with highest known reward
        :param obs: env state converted to 0 or 1 then summed as the Q state index
        :return: action
        """
        actions_a = self.continous_2_descrete(self.Qa, obs)
        actions_b = self.continous_2_descrete(self.Qb, obs)
        if max(actions_a) < max(actions_b):
            return np.argmax(actions_b) + 1
        return np.argmax(actions_a) + 1

    def update(self, Q, action, obs, next_obs, reward):
        if Q == 0:
            a_star = np.argmax(self.continous_2_descrete(self.Qa, next_obs))
            Qa = self.continous_2_descrete(self.Qa, obs)
            Qb = self.continous_2_descrete(self.Qb, next_obs)
            Qa[action] = Qa[action] + self.alpha * (reward + self.gamma * Qb[a_star] - Qa[action])
        else:
            b_star = np.argmax(self.continous_2_descrete(self.Qb, next_obs))
            Qb = self.continous_2_descrete(self.Qa, obs)
            Qa = self.continous_2_descrete(self.Qb, next_obs)
            Qb[action] = Qb[action] + self.alpha * (reward + self.gamma * Qa[b_star] - Qb[action])


def run(episodes=1000):
    loss = []
    ql = DoubleQL(state_space=8, action_space=env.action_space.n)

    obs = env.reset()
    for epoch in range(episodes):
        if (epoch + 1) % (episodes / 10) == 0:
            print("EPOCH ------> {}".format(epoch+1))

        done = False
        total_reward = 0
        while not done:
            if (epoch + 1) % (episodes / 10) == 0:
                env.render()
            action = ql.take_action(obs)
            next_obs, reward, done, info = env.step(action)

            ql.update(np.random.randint(0, 1), action - 1, obs, next_obs, reward)
            obs = next_obs

            total_reward += reward

            if done:
                obs = env.reset()

        loss.append(total_reward)

        if epoch % 100 == 0:
            last_ = np.mean(loss[-100:])
            print("Average Reward: {0:.2f} \n".format(last_))
    env.close()
    return [np.mean(loss[i - 100:i]) for i in range(100, len(loss))]


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    episodes = 1000
    loss = run(episodes=episodes)

    plt.title("Avg Reward for Agent at {} Episodes".format(episodes))
    plt.ylabel("Avg Reward")
    plt.plot([i + 1 for i in range(0, len(loss), 2)],
             loss[::2])
    plt.plot([i + 1 for i in range(0, len(loss), 2)],
             [200 for i in range(0, len(loss), 2)], 'g--')
    plt.show()