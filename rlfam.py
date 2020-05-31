import numpy as np

class RL(object):
    def __init__(self):
        pass


class Printer(object):
    rewards = []

    def __init__(self):
        pass

    def printState(self, obs, reward, done, info):
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

    def printResults(self, totalReward):
        print("---> Total Reward: {}".format(round(totalReward, 3)))
        if len(self.rewards) > 2:
            print("---> Improvement: {}%".format(round(100 * (totalReward / self.rewards[-1] - 1), 3)))
        self.rewards.append(totalReward)
        print("---> Avg Reward: {}\n".format(round(np.average(self.rewards), 3)))