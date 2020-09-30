import CHenv
import env_time_price
import numpy as np
num_episodes = 2
max_steps = 90
num_of_swap_server = 10
num_of_charging_slot = 15
initial_FB = 30
soc_threshold = 0.6
soc_max = 0.9
car_arrival_rate = 5
customer_arrival_rate = 5
episode_rewards = np.zeros(num_episodes, dtype=int)

env = env_time_price.BSSEnvironment(
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
        print("time : ", t)
        print("beginning----------------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.customer_arrival()
        print("customer_arrive----", "customer arrive:", env.customer_arrive, "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.car_arrival()
        print("car_arrive----", "car arrive:", env.car_arrive, "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        print(env.charging_slots)
        env.check_charging_slot()
        print(env.charging_slots)
        print("check_charging_slot------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.swap_battery()
        print("swap_battery-------------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.pick_up_car()
        print("pick_up_car--------------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.leave_queue()
        print("leave_queue--------------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        action = env.valid_random_action()
        print("valid_random_action:", action)
        env.charging_battery(action)
        print(env.charging_slots)
        print("charging_battery---------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.update_charging_slots()
        print(env.charging_slots)
        print("update_charging_slots----", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        env.update_waiting_time()
        print("update_waiting_time------", "customer queue:", env.Q1.qsize(), "EV to swap queue:", env.Q2.qsize(),
              "num of car:", len(env.parking_car), "DBs:", env.DB.qsize(), "FBs:", env.FB.qsize(),
              "CBs:", env.Ct, "idle slots:", env.idle_slot)
        s, r = env.step()
        print("---------------------------------------------------------------------------")
