import numpy as np
import random
import rl.printer
import gym


class SARSA(object):
    Q = []
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

    def update(self, action, next_action, obs, next_obs, reward):
        obs = self.continous_2_descrete(obs)
        next_obs = self.continous_2_descrete(next_obs)

        obs[action] = (1 - self.alpha) * obs[action] + self.alpha * (reward + self.gamma * next_obs[next_action])


def run(epochs=1000):
    printer = rl.printer.printer()
    ql = SARSA(state_space=8, action_space=env.action_space.n)

    obs = env.reset()
    for epoch in range(epochs):
        if (epoch + 1) % (epochs / 10) == 0:
            print("EPOCH ------> {}".format(epoch+1))
            printer.print_results()

        done = False
        total_reward = 0
        while not done:
            if (epoch + 1) % (epochs / 10) == 0:
                env.render()
            action = ql.take_action(obs)
            next_obs, reward, done, info = env.step(action)
            next_action = ql.take_action(next_obs)

            ql.update(action, next_action, obs, next_obs, reward)
            obs = next_obs

            total_reward += reward

            if done:
                obs = env.reset()

        printer.rewards.append(total_reward)
        if ql.explore > ql.explore_min:
            ql.explore -= ql.explore_decay

    env.close()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    run()