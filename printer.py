import numpy as np


class printer(object):
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