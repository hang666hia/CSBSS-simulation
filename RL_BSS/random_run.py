import bss_mod
import my_fun
import numpy as np
num_episodes = 3
max_steps = 200
num_of_charging_slots = 10
ini_full_battery = 20
arrival_rate = 6
table1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
table2 = [0.5, 1.5, 1.5, 1.5, 1.5, 1]
table3 = [1, 1, 1, 1, 1, 1.5]
table4 = [1.5, 1.5, 1.5, 0.5, 0.5, 0.5]
e_price_table = table1 + table2 + table3 + table4
episode_rewards = np.arange(num_episodes)
env = bss_mod.BSSEnvironment(
    max_steps=max_steps,
    num_of_charging_slots=num_of_charging_slots,
    ini_full_battery=ini_full_battery,
    arrival_rate=arrival_rate,
    e_price_table=e_price_table
)
step = 0
for episode in range(num_episodes):
    print('env reset!')
    env.reset()
    episode_reward = 0
    for t in range(env.max_steps):

        state = env.get_system_state()
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
        action = env.random_action()
        print('selected action:  ', my_fun.get_vector(int_action=action, len_of_vector=num_of_charging_slots))
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
        next_state, reward = env.step()
        episode_reward += reward
        step += 1
        print(step)

print('over')
