import gym
from rlfam import Printer, RL

env = gym.make('LunarLander-v2')
env.reset()


def run(epochs=10):
    printer = Printer()
    for epoch in range(epochs):
        print("EPOCH ------> {}".format(epoch+1))
        done = False
        totalReward = 0
        while not done:
            env.render()
            action = env.action_space.sample()  # Random Action
            obs, reward, done, info = env.step(action)
            totalReward += reward

            # printer.printState(obs, reward, done, info)

            if done:
                obs = env.reset()
        printer.printResults(totalReward)
        env.close()


if __name__ == "__main__":
    run()
