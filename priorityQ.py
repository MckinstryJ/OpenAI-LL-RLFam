import numpy as np
import random
import rl.printer
import gym
import matplotlib.pyplot as plt


class priorityQ(object):
    Q = []
    P = []
    M = []
    state_space = None
    action_space = None
    groups = None
    alpha = .1
    gamma = .99
    theta = 100
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

    def continuous_2_discrete(self, obs):
        location = self.Q
        for i in range(len(obs)):
            value = int(obs[i] * 100) % self.groups
            location = location[value]

        return location

    def take_action(self, obs):
        if random.random() < self.explore:
            return np.random.randint(0, self.action_space)
        return np.argmax(self.continuous_2_discrete(obs))

    def add_to_queue(self, obs, action, next_obs, reward, index=None):
        q_obs = self.continuous_2_discrete(obs)
        q_next_obs = self.continuous_2_discrete(next_obs)

        p = abs(reward + self.gamma * max(q_next_obs) - q_obs[action])
        if p > self.theta:
            self.P.append([obs, action, next_obs, reward])
            if index is not None:
                self.M.pop(index)
        else:
            self.M.append([obs, action, next_obs, reward])

    def update(self):
        q_obs = self.continuous_2_discrete(self.P[0][0])
        q_next_obs = self.continuous_2_discrete(self.P[0][2])

        q_obs[self.P[0][1]] = (1 - self.alpha) * q_obs[self.P[0][1]] \
                              + self.alpha * (self.P[0][3]
                                              + self.gamma * max(q_next_obs))

        for i in range(len(self.M)):
            for p in self.P:
                if all(self.M[i][2] == p[0]):
                    self.add_to_queue(*self.M[i], index=i)

        self.P = self.P[1:]


def run(episodes=1000, hallucinations=100):
    loss = []
    n = int(episodes / 10)

    printer = rl.printer.printer()
    ql = priorityQ(state_space=8, action_space=env.action_space.n)

    obs = env.reset()
    for epi in range(episodes):
        if (epi + 1) % (episodes / 10) == 0:
            print("EPOCH ------> {}".format(epi+1))
            printer.print_results()

        done = False
        score = 0
        while not done:
            if (epi + 1) % (episodes / 10) == 0:
                env.render()
            action = ql.take_action(obs)
            next_obs, reward, done, info = env.step(action)

            ql.add_to_queue(obs, action, next_obs, reward)
            obs = next_obs

            score += reward

            if done:
                obs = env.reset()
        loss.append(score)

        n = 0
        ql.P = sorted(ql.P, key=lambda x: x[3], reverse=True)
        while n < hallucinations and len(ql.P):
            ql.update()
            n += 1

        printer.rewards.append(score)
        if ql.explore > ql.explore_min:
            ql.explore -= ql.explore_decay

    env.close()
    return loss


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.seed(120)
    loss = run(episodes=1000, hallucinations=100)

    plt.title("Avg Reward for Agent at {} Episodes".format(1000))
    plt.ylabel("Avg Reward")
    plt.plot([i + 1 for i in range(0, len(loss))],
             loss)
    plt.show()