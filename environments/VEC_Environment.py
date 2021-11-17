import copy
import random
from collections import namedtuple
import gym
import torch
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint
from scipy.optimize import fsolve

class VEC_Environment(gym.Env):
    environment_name = "Vehicular Edge Computing"

    def __init__(self, num_vehicles=50, task_num=30):
        self.num_vehicles = num_vehicles
        self.task_num_per_episode = task_num
        self.vehicle_count = 0
        self.maxR = 500 #m, max relative distance between request vehicle and other vehicles
        self.maxV = 30 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 50 # maximum vehicles in the communication range of request vehicle
        self.max_local_task = 10
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(5,11)  #GHz
        self.data_size = [0.05, 0.1, 0.15, 0.2] #MBytes
        self.comp_size = [0.2, 0.4, 0.6, 0.8, 1] #GHz
        self.tau = [0.5, 1, 1.5, 2] #s
        self.max_datasize = max(self.data_size)
        self.max_compsize = max(self.comp_size)
        self.max_tau = max(self.tau)
        self.price = 0.1
        self.max_price = np.log(1+self.max_tau)/20
        self.price_level = 10
        self.sample_price = torch.distributions.Categorical(torch.tensor([float(i) for i in range(1, self.price_level+1)]))

        self.action_space = spaces.Discrete(self.num_vehicles*self.price_level)
        self.observation_space = spaces.Dict({
            "snr":spaces.Box(0,self.snr_ref,shape=(self.max_v,),dtype='float32'),
            "time_remain":spaces.Box(0,100,shape=(self.max_v,),dtype='float32'),
            "freq_remain":spaces.Box(0,6,shape=(self.max_v,),dtype='float32'),
            "u_max":spaces.Box(0,self.max_local_task*self.max_tau,shape=(self.max_v,),dtype='float32'),
            "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau),shape=(3,),dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "VEC"
        self.finish_count = [0,0,0,0]
        self.finish_delay = [0,0,0,0]
        self.utility = 0
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading
        self.init_vehicles()
        # self.generate_offload_tasks()
        self.generate_local_tasks()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        # for _ in range(random.randint(1,10)):
        #     self.add_vehicle()
        # self.move_vehicles()
        # self.generate_local_tasks()
        # self.generate_offload_tasks()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        for v in self.vehicles:
            v["freq"] = v["freq_init"]
            v["freq_remain"] = max(0, v["freq_init"] - sum([i[1]/i[2] for i in v["tasks"]]))
            alpha_max = v["freq_remain"]/v["freq"]
            v["u_max"] = sum([np.log(1+alpha_max*i[2]) for i in v["tasks"]])
            v["position"] = v["position_init"]
        with open("../finish_count.txt",'a') as f:
            f.write(str(self.utility)+' '+' '.join([str(i) for i in self.finish_count])+' '+' '.join([str(i) for i in self.finish_delay])+'\n')
        self.finish_count = [0,0,0,0]
        self.finish_delay = [0,0,0,0]
        self.utility = 0
        task = self.tasks[0]
        self.s = {
            "snr":np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "time_remain":np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "freq_remain":np.array([v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "u_max":np.array([v["u_max"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "task":np.array(task)}
        return spaces.flatten(self.observation_space, self.s)

    def step(self, action):
        self.step_count += 1
        # print("action=",action)
        self.reward = self.compute_reward(action)
        self.utility += self.reward
        v_id = action//self.price_level
        self.s["freq_remain"][v_id] = self.vehicles[v_id]["freq_remain"]
        self.s["u_max"][v_id] = self.vehicles[v_id]["u_max"]
        self.move_vehicles()
        self.s["snr"] = np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
        self.s["time_remain"] = np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
        if self.step_count >= self.task_num_per_episode: 
            self.done = True
        else: 
            self.done = False
            task = self.tasks[self.step_count]
            self.s["task"] = np.array(task)
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        task = self.s["task"]
        v_id = action//self.price_level
        u_max = self.s["u_max"][v_id]
        u_alpha = u_max - (action%self.price_level+1)/self.price_level*u_max
        cost = u_max - u_alpha + self.price*task[1]
        # cost = (1 + action%self.price_level)*self.price*task[1]
        reward = -np.log(1+self.max_tau)
        v = self.vehicles[v_id]
        if v["freq_remain"]==0:
            return reward
        alpha_max = v["freq_remain"]/v["freq"]
        alpha = fsolve(lambda a:sum([np.log(1+a*i[2]) for i in v["tasks"]])-u_alpha, 0.001)[0]
        alpha = min(max(0,alpha), alpha_max)
        freq_alloc = v["freq"]-(v["freq"]-v["freq_remain"])/(1-alpha)
        if freq_alloc <= 0:
            return reward
        snr = self.s["snr"][v_id]
        t_total = task[0]/(self.bandwidth*np.log2(1+snr)) + task[1]/freq_alloc
        if t_total <= min(task[2],self.s["time_remain"][v_id]):
            reward = np.log(1+task[2]-t_total) - cost
            v["freq"] -= freq_alloc
            v["freq_remain"] = max(0, v["freq"] - sum([i[1]/i[2] for i in v["tasks"]]))
            alpha_max = v["freq_remain"]/v["freq"]
            v["u_max"] = sum([np.log(1+alpha_max*i[2]) for i in v["tasks"]])
            self.finish_count[int(task[2]/0.5)-1] += 1
            self.finish_delay[int(task[2]/0.5)-1] += t_total
            # if reward <= 0:
            #     print("t_total=",t_total,"reward=",reward)
        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR*0.9,self.maxR*0.9)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[], "u_max":0})

    def add_vehicle(self):
        if len(self.vehicles) <= self.num_vehicles:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[], "u_max":0})

    def move_vehicles(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]/3.6*0.1
            # if abs(self.vehicles[i]["position"]) >= self.maxR:
            #     self.vehicles.pop(i)
            #     self.add_vehicle()

    def generate_local_tasks(self):
        for v in self.vehicles:
            v["tasks"] = []
            for _ in range(random.randint(1,self.max_local_task)):
                data_size = random.choice(self.data_size)
                compute_size = random.choice(self.comp_size)
                max_t = random.choice(self.tau)
                v["tasks"].append([data_size, compute_size, max_t])
    
    def generate_offload_tasks(self, file, task_num, group_num):
        with open(file,'w+') as f:
            for _ in range(group_num):
                f.write("tasks:\n")
                for _ in range(task_num):
                    data_size = random.choice(self.data_size)
                    compute_size = random.choice(self.comp_size)
                    max_t = random.choice(self.tau)
                    task = [str(data_size), str(compute_size), str(max_t)]
                    f.write(' '.join(task)+'\n')
        

    def produce_action(self, action_type):
        if action_type=="random":
            v_id = self.action_space.sample()//self.price_level
        elif action_type=="greedy":
            v_id = np.argmax(self.s["freq_remain"])
        task = self.s["task"]
        rewards = []
        for price in range(self.price_level):
            u_max = self.s["u_max"][v_id]
            u_alpha = u_max - (price+1)/self.price_level*u_max
            cost = u_max - u_alpha + self.price*task[1]
            v = self.vehicles[v_id]
            if v["freq_remain"]==0:
                reward = -np.log(1+self.max_tau)
                rewards.append(reward)
                continue
            alpha_max = v["freq_remain"]/v["freq"]
            alpha = fsolve(lambda a:sum([np.log(1+a*i[2]) for i in v["tasks"]])-u_alpha, 0.001)[0]
            alpha = min(max(0,alpha), alpha_max)
            freq_alloc = v["freq"]-(v["freq"]-v["freq_remain"])/(1-alpha)
            if freq_alloc <= 0:
                reward = -np.log(1+self.max_tau)
                rewards.append(reward)
                continue
            snr = self.s["snr"][v_id]
            t_total = task[0]/(self.bandwidth*np.log2(1+snr)) + task[1]/freq_alloc
            if t_total <= min(task[2],self.s["time_remain"][v_id]):
                reward = np.log(1+task[2]-t_total) - cost
            else:
                reward = -np.log(1+self.max_tau)
            rewards.append(reward)
        return v_id*self.price_level+np.argmax(rewards)

    def load_offloading_tasks(self, file, index):
        a = []
        self.tasks = []
        with open(file) as f:
            a = f.read().split("tasks:\n")[index].split('\n')
        for i in a[:-1]:
            tmp = i.split(' ')
            self.tasks.append([float(k) for k in tmp])