import bss_mod
import dqn_brain
import my_fun
import numpy as np


num_episodes = 100
max_steps = 24
num_of_charging_slots = 10
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
    e_greedy=0.9,
    replace_target_iter=200,
    memory_size=2000,
)

for episode in range(num_episodes):
    step = 0
    print('env reset!')
    env.reset()
    episode_reward = 0
    observation = env.get_system_state()
    for t in range(env.max_steps):
        print('--------------------------------------------------------')
        print('beginning of time slot')
        print('time:', env.t, '| new_arrival_EV_num:', env.new_arrival_num, '| depleted_battery:', env.depleted_battery,
              '| full_battery:', env.full_battery, '| charging_battery',  env.charging_battery)
        print('state of charging_slots: ', env.charging_slots)
        env.swap()
        print('--------')
        print('after swapping:')
        print('time:', env.t, '| new_arrival_EV_num:', env.new_arrival_num, '| depleted_battery:', env.depleted_battery,
              '| full_battery:', env.full_battery, '| charging_battery', env.charging_battery)
        print('state of charging_slots: ', env.charging_slots)
        # action = env.random_action()
        action = RL.choose_action(observation)
        print('selected action:  ', action)
        env.put_battery_to_pile(action)
        print('--------')
        print('after put_battery_to_pile')
        print('time:', env.t, '| new_arrival_EV_num:', env.new_arrival_num, '| depleted_battery:', env.depleted_battery,
              '| full_battery:', env.full_battery, '| charging_battery', env.charging_battery)
        print('state of charging_slots: ', env.charging_slots)
        env.finish_time_slot()
        print('--------')
        print('finish_time_slot')
        print('time:', env.t, '| new_arrival_EV_num:', env.new_arrival_num, '| depleted_battery:', env.depleted_battery,
              '| full_battery:', env.full_battery, '| charging_battery', env.charging_battery,
              '| new_full_battery',  env.new_full_battery)
        print('state of charging_slots: ', env.charging_slots)
        observation_, episode_finished, reward = env.step()

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_


        if episode_finished:
            print('episode_finished')
            break

        step += 1

print('over')
