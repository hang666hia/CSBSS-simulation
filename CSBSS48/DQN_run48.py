import env_time_price48
import RL_brain
import numpy as np

num_episodes = 6000
max_steps = 48
num_of_swap_server = 100
num_of_charging_slot = 15
initial_FB = 40
soc_threshold = 0.6
soc_max = 0.9
car_arrival_rate = 6
customer_arrival_rate = 6
episode_rewards = np.zeros(num_episodes, dtype=int)
service_loss = np.zeros(num_episodes, dtype=int)
electricity_cost = np.zeros(num_episodes)

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

RL = RL_brain.DeepQNetwork(
    n_actions=env.n_actions,
    n_features=env.n_features,
    learning_rate=0.01,
    reward_decay=0.8,
    replace_target_iter=300,
    memory_size=1000,
    batch_size=32,
)
step = 0
for episode in range(num_episodes):
    print('env reset!', 'episode:', episode)
    env.reset()
    observation = env.get_system_state()
    for t in range(env.max_steps):
        env.customer_arrival()
        env.car_arrival()
        env.check_charging_slot()
        env.swap_battery()
        env.pick_up_car()
        env.leave_queue()

        # epsilon = 0.5 * (0.99 ** episode)
        # if np.random.uniform(0, 1) >= epsilon:
        #     num = min(env.idle_slot, env.DB.qsize())
        #     action = RL.choose_action(observation, num)
        # else:
        #     action = env.valid_random_action()

        num = min(env.idle_slot, env.DB.qsize())
        action = RL.choose_action(observation, num)
        env.charging_battery(action)
        env.update_charging_slots()
        env.update_waiting_time()

        service_loss[episode] += env.num_leave_customer()

        electricity_cost[episode] += env.cost_buy_electricity()

        observation_, reward = env.step()

        episode_rewards[episode] += reward

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        observation = observation_

        step += 1

        if step % 200 == 0:
            print(step, episode_rewards[episode])


print('over')
# run with q_table
service_loss_run = np.zeros(1000, dtype=int)
electricity_cost_run = np.zeros(1000)
income = np.zeros(1000)
ch_power = np.zeros((1000, 48))
for run_episode in range(1000):
    print("run period:", run_episode)
    env.reset()
    observation = env.get_system_state()
    for t in range(env.max_steps):
        env.customer_arrival()
        env.car_arrival()
        env.check_charging_slot()
        env.swap_battery()
        env.pick_up_car()
        env.leave_queue()

        num = min(env.idle_slot, env.DB.qsize())
        action = RL.choose_action(observation, num)
        env.charging_battery(action)
        env.update_charging_slots()
        env.update_waiting_time()

        service_loss_run[run_episode] += env.num_leave_customer()

        electricity_cost_run[run_episode] += env.cost_buy_electricity()

        income[run_episode] += env.income - env.cost_buy_electricity()

        ch_power[run_episode, t] = env.Ct

        observation_, reward = env.step()
        observation = observation_




RL.plot_cost()
import matplotlib.pyplot as plt

plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.ylabel('DQN_episode_reward')
plt.xlabel('episode')
plt.show()

plt.plot(np.arange(len(service_loss)), service_loss)
plt.ylabel('DQN_run_service_loss')
plt.xlabel('episode')
plt.show()

plt.plot(np.arange(len(electricity_cost)), electricity_cost)
plt.ylabel('DQN_run_electricity_cost')
plt.xlabel('episode')
plt.show()

plt.plot(np.arange(48), np.mean(ch_power, axis=0))
plt.ylabel('ch_power_av')
plt.xlabel('episode')
plt.show()

print("episode rewards", np.mean(episode_rewards[num_episodes - 1000:num_episodes]))
print("service_loss", np.mean(service_loss[num_episodes - 1000:num_episodes]))
print("electricity cost", np.mean(electricity_cost[num_episodes - 1000:num_episodes]))
print("run period:")
print("service_loss_run", np.mean(service_loss_run))
print("electricity cost_run", np.mean(electricity_cost_run))
print("income run", np.mean(income))

