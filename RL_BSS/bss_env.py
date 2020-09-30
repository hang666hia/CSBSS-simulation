import numpy as np
import my_fun
no_charge = 0
slow_charge = 1
fast_charge = 2

ACTIONS = [no_charge, slow_charge , fast_charge]


class BSSEnvironment:

    def __init__(self, max_steps, num_of_charging_slots, ini_full_battery, arrival_rate, e_price_table):
        self.t = 0
        self.num_of_charging_slots = num_of_charging_slots
        self.action_space = np.arange(0, np.power(3, self.num_of_charging_slots))
        self.n_actions = len(self.action_space)
        self.n_features = None
        # 设置各种奖励的权重w
        self.w_delta_DB = 0.5
        self.w_ch = 0.5
        # 设置电池数量
        self.ini_full_battery = ini_full_battery
        self.depleted_battery = 0
        self.full_battery = self.ini_full_battery
        self.charging_battery = 0
        self.new_full_battery = 0

        # 泊松到达率
        self.arrival_rate = arrival_rate
        self.new_arrival_num = None
        self.max_steps = max_steps

        # 分时电价
        self.e_price_table = e_price_table

        self.charging_slots = np.zeros(self.num_of_charging_slots, dtype=np.int)

    def reset(self):
        self.t = 0
        self.new_arrival_num = np.random.poisson(lam=self.arrival_rate)
        self.depleted_battery = 0
        self.full_battery = self.ini_full_battery
        self.charging_battery = 0
        self.charging_slots = np.zeros(self.num_of_charging_slots, dtype=np.int)

    def step(self, action):

        Q1 = self.depleted_battery
        service_ev = min(self.new_arrival_num, self.full_battery)
        # 更新full_battery,depleted_battery
        self.full_battery -= service_ev
        self.depleted_battery += service_ev
        # 当前动作
        action_vector = my_fun.get_vector(action, self.num_of_charging_slots)
        # 放入充电槽的电池数量
        start_charging_battery = my_fun.count_nonzero(action_vector)
        self.depleted_battery -= start_charging_battery
        self.charging_slots = self.charging_slots + action_vector

        count_new_full_battery = 0
        for i in range(self.num_of_charging_slots):
            if (self.charging_slots[i] == 2) or (self.charging_slots[i] == -1):
                count_new_full_battery += 1
                self.charging_slots[i] = 0
            elif self.charging_slots[i] == 1:
                self.charging_slots[i] = -1
        self.full_battery += count_new_full_battery
        self.charging_battery = self.ini_full_battery - self.depleted_battery - self.full_battery
        Q2 = self.depleted_battery
        self.t += 1
        # 泊松到达
        self.new_arrival_num = np.random.poisson(lam=self.arrival_rate)
        s_ = self.get_system_state
        episode_finished = self.t >= self.max_steps
        # 此处reward还未完成
        reward = self.w_delta_DB*(Q2 - Q1)
        return s_, episode_finished, reward

    def get_system_state(self):

        return self.t, self.new_arrival_num, self.depleted_battery, self.full_battery, self.charging_battery, self.charging_slots

    def random_action(self):
        action = 0
        count = 0
        for i in range(self.num_of_charging_slots):
            if self.charging_slots[i] == 0:
                count += 1
                if count > self.depleted_battery:
                    break
                action += np.power(3, i)*np.random.choice(ACTIONS)
        return action
'''
    def set_system_state(self):
        self.charging_slots =
'''




