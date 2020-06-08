import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import numpy as np


class DQN:

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0  # Exploration Rate
        self.gamma = .99  # Discount Factor
        self.batch_size = 64  # Training Batch Size
        self.epsilon_min = .01  # Min to Exploration
        self.lr = 0.0005  # Learning Rate
        self.epsilon_decay = .005  # Exploration Decay
        self.memory = deque(maxlen=1000000)  # Memory limit of stored states
        self.model = self.build_model()  # Deep Learning Model

    def build_model(self):
        # Sequential Net
        model = Sequential()
        # 150 Nodes for first layer
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        # 120 Nodes for second layer
        model.add(Dense(120, activation=relu))
        # 4 Nodes for third layer -> Actions
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.lr))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        # Model is predicting rewards for each action
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)  # squeeze combines all to a single dimension
        next_states = np.squeeze(next_states)

        # Target value = reward + discount * (max action -> reward on next states)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        # Target value = predicted reward for current states
        targets_full = self.model.predict_on_batch(states)
        # For every index in batch, adjust rewards of current as a portion of the rewards in the future
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        # Retraining Model with states and rewards
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay


def run(episodes=100):
    loss = []
    n = int(episodes / 10)
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0

        if e > 500 and agent.gamma > .5:
            agent.gamma -= agent.epsilon_decay * 10

        max_steps = 2000
        for i in range(max_steps):
            if (e + 1) % int(episodes / 10) == 0:
                env.render()

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))

            # Storing the set for retraining purposes
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            # Retraining the Agent
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e + 1, episodes, score))
                break
        loss.append(score)

        last_ = np.mean(loss[-n:])
        print("Average Reward: {0:.2f} \n".format(last_))
    return [np.mean(loss[i-n:i]) for i in range(n, len(loss))]


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    episodes = 1000
    loss = run(episodes=episodes)

    plt.title("Avg Reward for Agent at {} Episodes".format(episodes))
    plt.ylabel("Avg Reward")
    plt.plot([i+1 for i in range(0, len(loss), 2)],
             loss[::2])
    plt.plot([i+1 for i in range(0, len(loss), 2)],
             [200 for i in range(0, len(loss), 2)], 'g--')
    plt.show()