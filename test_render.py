import gym
env = gym.make("procgen:procgen-coinrun-v0", render_mode="human")
obs = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        break