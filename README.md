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

<h3>Values Provided in env.step(action):</h3>
<ul>
  <li>Observation (from ship's perspective):</li>
  <ul>
    <li>obs[0] -> x position</li>
    <li>obs[1] -> y position</li>
    <li>obs[2] -> x velocity</li>
    <li>obs[3] -> y velocity</li>
    <li>obs[4] -> angle</li>
    <li>obs[5] -> angle velocity</li>
    <li>obs[6] -> left leg contact w/ ground</li>
    <li>obs[7] -> right leg contact w/ ground</li>
  </ul>
  <li>reward -> is ship between flags?</li>
  <li>done? -> has the ship made contact w/ ground?</li>
</ul>
  
