import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pyformulas as pf

# Configs
seed = 0
gamma = .99
learning_rate = .01
max_steps = 5000
fig = plt.figure()
canvas = np.zeros((480, 640))
screen = pf.screen(canvas, "Agent")

env = gym.make('LunarLander-v2')
env.seed(seed)

# Model Creation
input_size = int(env.observation_space.shape[0])
action_size = env.action_space.n
num_hidden = 128

inputs = layers.Input(shape=(input_size,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(action_size, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

# Training Model
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
_loss = keras.losses.SquaredHinge()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = [0]
eps = np.finfo(np.float32).eps.item()
episodes = 3274

# Stepping through Lunar Lander
for epi in range(episodes):
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps):
            if epi % 100 == 0:
                env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predicting action probabilities and estimated future rewards from state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(action_size, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check if solved
        running_reward.append(0.05 * episode_reward + (1 - 0.05) * running_reward[-1])

        # Calculate expected value from rewards
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)

            critic_losses.append(
                _loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    if epi % 100 == 0:
        message = "EPISODE {} ---> Running Reward: {:.2f}"
        print(message.format(epi, running_reward[-1]))

        plt.title("ActorCritic - Running Reward")
        plt.ylabel("Reward")
        plt.plot(running_reward)
        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        screen.update(image)

    if running_reward[-1] > 200:
        message = "EPISODE {} ---> Average Reward: {:.2f}"
        print(message.format(epi, round(np.average(running_reward[-100:]), 2)))
        print("---> Solved!")
        break