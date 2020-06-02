import gym
from rlfam import Printer, QL


def run(epochs=1000):
    printer = Printer()
    ql = QL(state_space=8, action_space=env.action_space.n)

    obs = env.reset()
    for epoch in range(epochs):
        if (epoch + 1) % (epochs / 10) == 0:
            print("EPOCH ------> {}".format(epoch+1))

        done = False
        total_reward = 0
        while not done:
            if (epoch + 1) % (epochs / 10) == 0:
                env.render()
            action = ql.take_action(obs)
            new_obs, reward, done, info = env.step(action)

            ql.update(action, obs, new_obs, reward)
            obs = new_obs

            total_reward += reward

            if done:
                obs = env.reset()

        printer.rewards.append(total_reward)
        if (epoch + 1) % (epochs / 10) == 0:
            ql.explore -= .01

            if ql.explore < .01: ql.explore = .01

            printer.print_results()
    env.close()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    run()
