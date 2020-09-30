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
        self.n_features = 4 + self.num_of_charging_slots
        # 设置各种奖励的权重w
        self.service_reward = 0
        self.e_price_reward = 0
        self.w_service_reward = 0.5
        self.w_e_price_reward = 0.5
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
        # 所有充电槽的状态
        self.charging_slots = np.zeros(self.num_of_charging_slots, dtype=np.int)

        self.episode_finished = None

    def reset(self):
        self.t = 0
        self.new_arrival_num = np.random.poisson(lam=self.arrival_rate)
        self.depleted_battery = 0
        self.full_battery = self.ini_full_battery
        self.charging_battery = 0
        self.charging_slots = np.zeros(self.num_of_charging_slots, dtype=np.int)

    def swap(self):
        # 被服务的电动车车数量
        service_ev = min(self.new_arrival_num, self.full_battery)
        '''
        if self.new_arrival_num == 0:
            self.service_reward = 0
        else:
            self.service_reward = service_ev / self.new_arrival_num
        '''
        # 更新full_battery,depleted_battery
        self.full_battery -= service_ev
        self.depleted_battery += service_ev

    def random_action(self):
        action = 0
        count = 0
        for i in range(self.num_of_charging_slots):
            if self.charging_slots[i] == 0:
                count += 1
                if count > self.depleted_battery:
                    break
                action += np.power(3, i) * np.random.choice(ACTIONS)
        return action

    def action_valid(self, action):
        valid = True
        action_vector = my_fun.get_vector(action, self.num_of_charging_slots)
        for i in range(self.num_of_charging_slots):
            if (action_vector[i] != 0) and (self.charging_slots[i] != 0):
                valid = False
                break
        return valid

    def put_battery_to_pile(self, action):
        action_vector = my_fun.get_vector(action, self.num_of_charging_slots)
        # 放入充电槽的电池数量
        start_charging_battery = my_fun.count_nonzero(action_vector)
        self.service_reward = start_charging_battery
        self.depleted_battery -= start_charging_battery
        # 根据action更新充电槽的状态
        self.charging_slots = self.charging_slots + action_vector
        e_price = self.e_price_table[self.t]
        for i in range(self.num_of_charging_slots):
            self.e_price_reward += (2 - e_price) * abs(self.charging_slots[i])

        self.charging_battery = self.ini_full_battery - self.depleted_battery - self.full_battery

    def finish_time_slot(self):
        count_new_full_battery = 0
        for i in range(self.num_of_charging_slots):
            if (self.charging_slots[i] == 2) or (self.charging_slots[i] == -1):
                count_new_full_battery += 1
                self.charging_slots[i] = 0
            elif self.charging_slots[i] == 1:
                self.charging_slots[i] = -1
        self.new_full_battery = count_new_full_battery
        self.full_battery += count_new_full_battery
        self.charging_battery = self.ini_full_battery - self.depleted_battery - self.full_battery

    def step(self):
        # 完成这步，进入下一步
        self.t = (self.t + 1) % 24
        self.new_arrival_num = np.random.poisson(lam=self.arrival_rate)
        s_ = self.get_system_state()
        reward = self.service_reward + self.e_price_reward
        return s_, reward

    def get_system_state(self):
        s1 = [self.t, self.new_arrival_num, self.depleted_battery, self.full_battery]
        s2 = self.charging_slots
        s = s1 + list(s2)
        s = np.array(s)
        return s







