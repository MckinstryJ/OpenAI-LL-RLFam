# OpenAI-LL-RLFam
<h2>Applying all known RL models to the OpenAI LunarLanding-v2 Environment</h2>

<h3>Setting up the Environment:</h3>
<ul>
  <li>Using Python v3.7 (Anaconda)</li>
</ul>

    pip install gym
    pip install Box2D

<h3>Starter Code (to render the ship and take random actions):</h3>

    import gym
    
    env = gym.make('LunarLander-v2')
    env.reset()


    def run(epochs=10):  # Epochs: number of rounds the ship should take.
        printer = Printer()
        for epoch in range(epochs):
            print("EPOCH ------> {}".format(epoch+1))
            done = False
            while not done:
                env.render()  # Allows you to view the ship
                
                action = env.action_space.sample()  # Random Action
                obs, reward, done, info = env.step(action)

                if done:
                    obs = env.reset()
        env.close()


    if __name__ == "__main__":
        run()

<h3>Applied Models</h3>
<ol>
  <li><b>Q Learning</b></li>
</ol>

Report:
[Reinforcement Learning Family](https://docs.google.com/document/d/1COi30Slc3M_qka92b-udfsq7BWqYjuI-Hp2-IjEQrKg/edit?usp=sharing).
