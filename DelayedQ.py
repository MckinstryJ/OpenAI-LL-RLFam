import numpy as np
import random
import gym
import matplotlib.pyplot as plt


class DelayedQ(object):
    state_space = None
    action_space = None
    groups = None
    gamma = .99
    m = 30.
    e = 1.
    explore = 1.0
    explore_decay = .001

    def __init__(self, state_space, action_space, group=5):
        self.state_space = state_space
        self.action_space = action_space
        self.groups = group

        shap = [self.groups for i in range(self.state_space)]
        shap.append(self.action_space)
        shap = tuple(shap)

        self.Q = np.full(shap, 1 / (1 - self.gamma))
        self.U = np.zeros(shap)
        self.l = np.zeros(shap)
        self.b = np.zeros(shap)
        self.learn = np.full(shap, True)

    def continous_2_descrete(self, obj, obs):
        """
            Convert Observation to Discrete
        :param obs: env state
        :return: transformed obs
        """
        for i in range(len(obs)):
            obj = obj[int(obs[i] * 100) % self.groups]

        return obj

    def adjust(self, obj, obs, value):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        obj[obs] = value

    def adjustL(self, obs):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        self.l[obs] += 1

    def adjustU(self, obs, r, next_obs):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        self.U[obs] += r + self.gamma * np.argmax(self.continous_2_descrete(self.Q, next_obs))

    def adjustQ(self, obs):
        temp = []
        for i in range(len(obs)):
            v = int(obs[i] * 100) % self.groups
            temp.append(v)
        obs = tuple(temp)

        self.Q[obs] = self.continous_2_descrete(self.U, obs) / self.m + self.e

    def take_action(self, obs):
        """
            Explore if random number is greater than #
            Otherwise, return action with highest known reward
        :param obs: env state converted to 0 or 1 then summed as the Q state index
        :return: action
        """
        if random.random() < self.explore:
            return np.random.randint(0, self.action_space)
        return np.argmax(self.continous_2_descrete(self.Q, obs))


def run(episodes=1000):
    loss = []
    ql = DelayedQ(state_space=8, action_space=env.action_space.n)

    obs = env.reset()
    for epi in range(episodes):
        if (epi + 1) % (episodes / 10) == 0:
            print("EPISODES ------> {}".format(epi+1))

        done = False
        total_reward = 0
        t, t_star = 0, 0
        while not done:
            if (epi + 1) % (episodes / 10) == 0:
                env.render()
            action = ql.take_action(obs)
            next_obs, reward, done, info = env.step(action)

            state_action = np.append(obs, action)
            if ql.continous_2_descrete(ql.b, state_action) <= t_star:
                ql.adjust(ql.learn, state_action, True)

            if ql.continous_2_descrete(ql.learn, state_action):
                if ql.continous_2_descrete(ql.l, state_action) == 0:
                    ql.adjust(ql.b, state_action, t)
                ql.adjustL(state_action)
                ql.adjustU(state_action, reward, next_obs)

                if ql.continous_2_descrete(ql.l, state_action) >= ql.m:
                    if ql.continous_2_descrete(ql.Q, state_action) \
                            - ql.continous_2_descrete(ql.U, state_action) / ql.m >= 2 * ql.e:
                        ql.adjustQ(state_action)
                        t_star = t
                    elif ql.continous_2_descrete(ql.b, state_action) > t_star:
                        ql.adjust(ql.learn, state_action, False)
                    ql.adjust(ql.U, state_action, 0)
                    ql.adjust(ql.l, state_action, 0)
            t += 1
            obs = next_obs

            total_reward += reward

            if done:
                obs = env.reset()

        loss.append(total_reward)

        if epi % 100 == 0:
            last_ = np.mean(loss[-100:])
        if (epi + 1) % (episodes / 10) == 0:
            print("Average Reward: {0:.2f} \n".format(last_))
        if ql.explore > .001:
            ql.explore -= ql.explore_decay
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