from maze_env import Maze
from DQN_modified import DeepQNetwork
import numpy as np
episode_reward = np.zeros(300, dtype=float)

def run_maze():
    step = 0

    for episode in range(300):
        episode_reward[episode] = 0
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            episode_reward[episode] += reward

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    # print(RL.memory)
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(episode_reward)), episode_reward)
    plt.ylabel('episode_reward')
    plt.xlabel('run steps')
    plt.show()
