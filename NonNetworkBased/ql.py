import numpy as np
import random
import rl.printer
import gym
import matplotlib.pyplot as plt
from matplotlib import animation

full_obs = []


class QL(object):
    Q = []
    state_space = None
    action_space = None
    groups = None
    alpha = .1
    gamma = 1.0
    explore = 1.0
    explore_min = .01
    explore_decay = .001

    def __init__(self, state_space, action_space, group=5):
        self.state_space = state_space
        self.action_space = action_space
        self.groups = group

        shap = [self.groups for i in range(self.state_space)]
        shap.append(self.action_space)
        shap = tuple(shap)

        self.Q = np.zeros(shape=shap)

    def continous_2_descrete(self, obs):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """

        location = self.Q
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
        if random.random() < self.explore:
            return np.random.randint(0, self.action_space)
        return np.argmax(self.continous_2_descrete(obs))

    def update(self, action, obs, next_obs, reward):
        obs = self.continous_2_descrete(obs)
        next_obs = self.continous_2_descrete(next_obs)

        obs[action] = (1 - self.alpha) * obs[action] + self.alpha * (reward + self.gamma * max(next_obs))


def run(episodes=1000):
    loss = []
    ql = QL(state_space=8, action_space=env.action_space.n)

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

            ql.update(action, obs, next_obs, reward)
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

    print("X Position Min - Max: ({} - {})".format(min([i[0] for i in full_obs]), max([i[0] for i in full_obs])))
    print("Y Position Min - Max: ({} - {})".format(min([i[1] for i in full_obs]), max([i[1] for i in full_obs])))
    print("X Velo Min - Max: ({} - {})".format(min([i[2] for i in full_obs]), max([i[2] for i in full_obs])))
    print("Y Velo Min - Max: ({} - {})".format(min([i[3] for i in full_obs]), max([i[3] for i in full_obs])))
    print("Angle Min - Max: ({} - {})".format(min([i[4] for i in full_obs]), max([i[4] for i in full_obs])))
    print("Angle Velo Min - Max: ({} - {})".format(min([i[5] for i in full_obs]), max([i[5] for i in full_obs])))