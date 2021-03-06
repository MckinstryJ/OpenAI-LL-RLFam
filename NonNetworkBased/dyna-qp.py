import numpy as np
import random
import rl.printer
import gym


class DynaQ(object):
    Q = []
    M = []
    state_space = None
    action_space = None
    groups = None
    alpha = .1
    gamma = .99
    explore = 1.0
    explore_min = .01
    explore_decay = .001
    k = 1.0

    def __init__(self, state_space, action_space, group=5):
        self.state_space = state_space
        self.action_space = action_space
        self.groups = group

        shap = [self.groups for i in range(self.state_space)]
        shap.append(self.action_space)
        shap = tuple(shap)

        self.Q = np.zeros(shape=shap)

    def continous_2_descrete(self, obs):
        location = self.Q
        for i in range(len(obs)):
            value = int(obs[i] * 100) % self.groups
            location = location[value]

        return location

    def take_action(self, obs):
        if random.random() < self.explore:
            return np.random.randint(0, self.action_space)
        return np.argmax(self.continous_2_descrete(obs))

    def update(self, obs, action, next_obs, reward, last=None, halu=False):
        q_obs = self.continous_2_descrete(obs)
        q_next_obs = self.continous_2_descrete(next_obs)

        if not halu:
            q_obs[action] = (1 - self.alpha) * q_obs[action] + self.alpha * (reward + self.gamma * max(q_next_obs))
            self.M.append([obs, action, next_obs, reward, 0, True])
        elif halu:
            bonus_reward = reward + self.k * np.sqrt(last)
            q_obs[action] = (1 - self.alpha) * q_obs[action] + self.alpha * (bonus_reward + self.gamma * max(q_next_obs))
            return 0


def run(epochs=1000, hallucinations=1000):
    printer = rl.printer.printer()
    ql = DynaQ(state_space=8, action_space=env.action_space.n)

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

            ql.update(obs, action, next_obs, reward)
            obs = next_obs

            total_reward += reward

            if done:
                obs = env.reset()

        # Increasing the 'last used' metric for all in memory
        for mem in ql.M:
            mem[-2] += 1
        for n in range(hallucinations):
            rand_up = random.choice(ql.M)
            rand_up[-2] = ql.update(*rand_up)

        printer.rewards.append(total_reward)
        if ql.explore > ql.explore_min:
            ql.explore -= ql.explore_decay

    env.close()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.seed(120)
    run(epochs=1000, hallucinations=1000)