import bss_mod
import dqn_brain
import my_fun
import numpy as np


num_episodes = 200
max_steps = 200
num_of_charging_slots = 10
ini_full_battery = 20
arrival_rate = 6
e_price_table = range(24)
episode_rewards = np.arange(num_episodes)

env = bss_mod.BSSEnvironment(
    max_steps=max_steps,
    num_of_charging_slots=num_of_charging_slots,
    ini_full_battery=ini_full_battery,
    arrival_rate=arrival_rate
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
        a = RL.choose_action(observation)
        env.swap()
        action = env.random_action()
        '''
        epsilon = 0.5
        if np.random.uniform(0, 1) >= epsilon:
            action = env.random_action()
        else:
            action = env.valid_random_action()

        if not env.action_valid(action):
            observation_ = np.zeros(RL.n_features, dtype=int)

            reward = 0

            RL.store_transition(observation, action, reward, observation_)

            env.put_battery_to_pile(action)
            env.finish_time_slot()
            ob, r = env.step()
            b = RL.choose_action(ob)

            print('episode finished with invalid action')
            break
        '''

        env.put_battery_to_pile(action)

        env.finish_time_slot()

        observation_, reward = env.step()

        RL.store_transition(observation, action, reward, observation_)

        # swap observation
        observation = observation_

        step += 1
        if step % 200 == 0:
            print(step)

print('over')
RL.plot_cost()
