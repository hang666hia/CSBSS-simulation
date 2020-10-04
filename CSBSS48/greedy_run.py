import env_time_price48
import numpy as np
num_episodes = 500
max_steps = 48
num_of_swap_server = 100
num_of_charging_slot = 15
initial_FB = 30
soc_threshold = 0.6
soc_max = 0.9
car_arrival_rate = 6
customer_arrival_rate = 6
episode_rewards = np.zeros(num_episodes, dtype=int)
service_loss = np.zeros(num_episodes)
electricity_cost = np.zeros(num_episodes)
income = np.zeros(num_episodes)
ch_power = np.zeros((500, 48))

env = env_time_price48.BSSEnvironment(
    max_steps=max_steps,
    num_of_swap_server=num_of_swap_server,
    num_of_charging_slot=num_of_charging_slot,
    initial_FB=initial_FB,
    soc_threshold=soc_threshold,
    soc_max=soc_max,
    car_arrival_rate=car_arrival_rate,
    customer_arrival_rate=customer_arrival_rate
)

step = 0
for episode in range(num_episodes):
    print("episode:", episode)
    env.reset()
    for t in range(env.max_steps):
        env.customer_arrival()
        env.car_arrival()
        env.check_charging_slot()
        env.swap_battery()
        env.pick_up_car()
        env.leave_queue()
        action = env.greedy_action()
        env.charging_battery(action)
        env.update_charging_slots()
        env.update_waiting_time()

        service_loss[episode] += env.num_leave_customer()

        electricity_cost[episode] += env.cost_buy_electricity()

        income[episode] += env.income - env.cost_buy_electricity()

        ch_power[episode, t] = env.Ct

        s, reward = env.step()
        episode_rewards[episode] += reward

import matplotlib.pyplot as plt
# plt.plot(np.arange(len(episode_rewards)), episode_rewards)
# plt.ylabel('greedy_run_episode_reward')
# plt.xlabel('episode')
# plt.show()

# plt.plot(np.arange(len(service_loss)), service_loss)
# plt.ylabel('greedy_run_service_loss')
# plt.xlabel('episode')
# plt.show()
#
# plt.plot(np.arange(len(electricity_cost)), electricity_cost)
# plt.ylabel('greedy_run_electricity_cost')
# plt.xlabel('episode')
# plt.show()

plt.plot(np.arange(48), np.mean(ch_power, axis=0))
plt.ylabel('ch_power_av')
plt.xlabel('episode')
plt.show()

print("episode reward", np.mean(episode_rewards))
print("service_loss", np.mean(service_loss))
print("electricity cost", np.mean(electricity_cost))
print("income", np.mean(income))
