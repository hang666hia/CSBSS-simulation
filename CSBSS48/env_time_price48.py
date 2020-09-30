import numpy as np
import random
from queue import Queue


class BSSEnvironment:

    def __init__(self, max_steps, num_of_swap_server, num_of_charging_slot, initial_FB, soc_threshold, soc_max, car_arrival_rate,
                 customer_arrival_rate):
        self.t = 0
        self.e_price_table = np.zeros(48)
        for e in range(48):
            if 0 <= e <= 11:
                self.e_price_table[e] = 1.5
            elif 12 <= e <= 43:
                self.e_price_table[e] = 1
            else:
                self.e_price_table[e] = 0.5
        self.soc_threshold = soc_threshold
        self.soc_max = soc_max
        self.num_of_swap_server = num_of_swap_server
        self.num_of_charging_slot = num_of_charging_slot
        self.initial_FB = initial_FB
        self.action_space = np.arange(0, self.num_of_charging_slot + 1)
        self.car_arrival_rate = car_arrival_rate
        self.customer_arrival_rate = customer_arrival_rate
        self.arr_low = 6
        self.arr_high = 10
        self.car_arrive = None
        self.customer_arrive = None
        self.charging_rate = 0.1
        # customer queue
        self.Q1 = Queue()
        # max length of customer queue
        self.M = 20
        # create array to record customers' waiting time
        self.waiting_time = np.array([])
        self.max_wait = 4
        # EV to swap battery queue
        self.Q2 = Queue()
        # batteries to charge queue
        self.DB = Queue()
        # fully charged battery queue,it need to initialize
        self.FB = Queue()
        # number of idle charging slot
        self.idle_slot = None
        # number of being charged battery
        self.Ct = None

        self.parking_car = np.array([0.85, 0.9, 0.95, 0.7, 0.75])
        self.charging_slots = np.zeros(self.num_of_charging_slot, dtype=float)

        self.cost = None
        self.U_1 = None
        self.U_2 = None
        self.service_customer = None

        self.n_actions = len(self.action_space)
        # n_feature 由 def get_system_state(self)确定
        self.n_features = 7
        self.max_steps = max_steps
        self.episode_finished = False

    def reset(self):
        self.t = 0
        self.Q2 = Queue()
        self.Q1 = Queue()
        self.DB = Queue()
        # FB need to initialize
        self.FB = Queue()
        self.idle_slot = self.num_of_charging_slot
        self.Ct = self.num_of_charging_slot - self.idle_slot
        self.service_customer = 0
        self.charging_slots = np.zeros(self.num_of_charging_slot, dtype=float)
        for i in range(self.initial_FB):
            self.FB.put(0.95)
        self.parking_car = np.array([0.85, 0.9, 0.95, 0.7, 0.75])

    def customer_arrival(self):

        # self.customer_arrive = np.random.poisson(lam=self.customer_arrival_rate)
        self.customer_arrive = np.random.randint(self.arr_low, self.arr_high)
        # user enter waiting queue
        count = 0
        for i in range(self.customer_arrive):

            distance = random.uniform(0.3, 0.7)
            if self.Q1.qsize() < self.M:
                self.Q1.put(distance)
                count += 1
        self.U_1 = self.customer_arrive - count

    def car_arrival(self):
        # self.car_arrive = np.random.poisson(lam=self.car_arrival_rate)
        self.car_arrive = np.random.randint(self.arr_low, self.arr_high)
        # check SOC of ev
        for i in range(self.car_arrive):
            ran_soc = random.uniform(0.3, 0.7)
            if ran_soc >= self.soc_threshold:
                self.parking_car = np.append(self.parking_car, ran_soc)
            else:
                self.Q2.put(ran_soc)

    def check_charging_slot(self):
        for i in range(self.num_of_charging_slot):
            if self.charging_slots[i] >= self.soc_max:
                self.FB.put(self.charging_slots[i])
                self.charging_slots[i] = 0
                self.idle_slot += 1
# at the end of time slot, we need to update the status of charging slots

    def swap_battery(self):
        num_swap = min(self.num_of_swap_server, self.FB.qsize(), self.Q2.qsize())
        self.reward_num_swap = - max(0, self.Q2.qsize()-num_swap)
        for i in range(num_swap):
            low_soc = self.Q2.get()
            self.DB.put(low_soc)
            high_soc = self.FB.get()
            self.parking_car = np.append(self.parking_car, high_soc)

    def pick_up_car(self):
        self.parking_car = np.sort(self.parking_car)
        self.income = 0
        profit_v = 7
        count = 0
        for i in range(self.Q1.qsize()):
            if len(self.parking_car) == 0:
                pass
            else:
                demand = self.Q1.get()
                if (len(self.parking_car) == 1) and self.parking_car[0] < demand:
                    self.Q1.put(demand)
                elif (len(self.parking_car) == 1) and self.parking_car[0] >= demand:
                    self.parking_car = np.delete(self.parking_car, 0)
                    count += 1
                    self.income += demand * profit_v
                elif demand <= self.parking_car[0]:
                    self.parking_car = np.delete(self.parking_car, 0)
                    count += 1
                    self.income += demand * profit_v
                else:
                    found = False
                    for j in range(len(self.parking_car) - 1):
                        if self.parking_car[j] < demand <= self.parking_car[j + 1]:
                            found = True
                            self.parking_car = np.delete(self.parking_car, j + 1)
                            count += 1
                            self.income += demand * profit_v
                    if not found:
                        self.Q1.put(demand)
            self.service_customer = count

    def leave_queue(self):
        leave_index = np.array([])
        for i in range(len(self.waiting_time)):
            if self.waiting_time[i] == self.max_wait:
                leave_index = np.append(leave_index, i)
        count = 0
        for i in range(self.Q1.qsize()):
            a = self.Q1.get()
            if (len(leave_index) != 0) and (leave_index[0] == i):
                print(i)
                leave_index = np.delete(leave_index, 0)
                self.waiting_time = np.delete(self.waiting_time, i - count)
                count += 1
            else:
                self.Q1.put(a)
        self.U_2 = count

    def greedy_action(self):
        a = min(self.DB.qsize(), self.idle_slot)
        return a

    def random_action(self):
        action = random.randint(0, self.num_of_charging_slot)
        return action

    def valid_random_action(self):
        action = random.randint(0, min(self.DB.qsize(), self.idle_slot))
        return action

    # 判断action是否可行
    def action_valid(self, action):
        valid = True
        if action > self.Q2.qsize() or action > self.num_of_swap_server:
            valid = False
        return valid

    def charging_battery(self, action):
        to_charge = action
        for i in range(self.num_of_charging_slot):
            if self.charging_slots[i] == 0 and to_charge > 0:
                self.charging_slots[i] = self.DB.get()
                to_charge -= 1
                self.idle_slot -= 1
        self.Ct = self.num_of_charging_slot - self.idle_slot

    def update_charging_slots(self):
        for i in range(self.num_of_charging_slot):
            if self.charging_slots[i] != 0:
                self.charging_slots[i] += 0.1

    def update_waiting_time(self):
        for i in range(len(self.waiting_time)):
            self.waiting_time[i] += 1

    def num_leave_customer(self):
        return self.U_1 + self.U_2

    def cost_buy_electricity(self):
        return self.Ct * self.e_price_table[self.t]

    def step(self):
        s_ = self.get_system_state()
        reward = - self.cost_buy_electricity() - self.num_leave_customer()*16

        # 完成这步，进入下一步
        self.t = self.t + 1
        return s_, reward

    def get_system_state(self):
        s1 = self.Q1.qsize()
        s2 = self.Q2.qsize()
        s3 = len(self.parking_car)
        s4 = self.idle_slot
        s5 = self.DB.qsize()
        s6 = self.FB.qsize()
        s7 = self.e_price_table[self.t]
        s = np.array([s1, s2, s3, s4, s5, s6, s7])
        return s
