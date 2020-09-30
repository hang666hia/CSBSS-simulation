import bss_mod
import dqn_brain
import my_fun
import numpy as np


num_episodes = 10000
max_steps = 24
num_of_charging_slots = 8
ini_full_battery = 20
arrival_rate = 6
e_price_table = range(24)
episode_rewards = np.arange(num_episodes)

env = bss_mod.BSSEnvironment(
    max_steps=max_steps,
    num_of_charging_slots=num_of_charging_slots,
    ini_full_battery=ini_full_battery,
    arrival_rate=arrival_rate,
    e_price_table=e_price_table
)

RL = dqn_brain.DeepQNetwork(
    n_actions=env.n_actions,
    n_features=env.n_features,
    learning_rate=0.01,
    reward_decay=0.9,
    replace_target_iter=200,
    memory_size=2000,
)
step = 0
for episode in range(num_episodes):
    print('env reset!')
    env.reset()
    episode_reward = 0
    observation = env.get_system_state()
    for t in range(env.max_steps):
        env.swap()

        # action = env.random_action()
        epsilon = 1 * (0.99 ** episode)
        if np.random.uniform(0, 1) >= epsilon:
            # print('RL action:')
            action = RL.choose_action(observation)
        else:
            # print('random action')
            action = env.random_action()
        # print(my_fun. get_vector(action, num_of_charging_slots))

        # 当前action是一个无效的action
        if not env.action_valid(action):
            observation_ = np.zeros(RL.n_features, dtype=int)

            reward = -10000

            RL.store_transition(observation, action, reward, observation_)

            print('episode finished with invalid action')
            break

        env.put_battery_to_pile(action)

        env.finish_time_slot()

        observation_, reward = env.step()

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_

        step += 1
        if step % 200 == 0:
            print(step)

print('over')
RL.plot_cost()
