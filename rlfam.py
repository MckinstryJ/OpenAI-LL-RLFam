import numpy as np
import random


class QL(object):
    Q = []
    state_space = None
    action_space = None
    groups = None
    alpha = 1.0
    gamma = 1.0
    explore = 1.0

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

        next_obs[action] = (1 - self.alpha) * obs[action] \
            + self.alpha * (reward + self.gamma * max(next_obs))


class Printer(object):
    rewards = []

    def __init__(self):
        pass

    def print_state(self, obs, reward, done, info):
        print("\nState --->")
        print("   X Position: {}".format(obs[0]))
        print("   Y Position: {}".format(obs[1]))
        print("   X Velocity: {}".format(obs[2]))
        print("   Y Velocity: {}".format(obs[3]))
        print("   Angle: {}".format(obs[4]))
        print("   Angle Velocity: {}".format(obs[5]))
        print("   Left Leg Contact: {}".format(obs[6]))
        print("   Right Leg Contact: {}".format(obs[7]))
        print("Reward: {}".format(reward))
        if done:
            print("Done? {}".format(done))
        if info:
            print("Info: {}".format(info))

    def print_results(self):
        print("---> Avg Reward: {}\n".format(round(np.average(self.rewards), 3)))