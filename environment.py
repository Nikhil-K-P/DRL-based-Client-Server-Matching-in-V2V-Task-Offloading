import numpy as np
import math
from scipy.stats import nakagami
from vehicular_environment import VehicularNetwork
class Environment:
    def __init__(self,N):

        self.N=N
        self.network=VehicularNetwork(self.N)
        #replace env.observation.shape in main.py with this
        self.state_space_shape=(5+2*self.N,)
        self.action_space=[i for i in range(self.N)]
        self.min_action=0
        self.max_action=self.N-1
        self.reward_range=(0,1)
        self.request_queue=[]

        self.time_step=0
        self.Flag=False
        self.no_time_steps=100

        self.curr_state=np.zeros(self.state_space_shape)

        #analysis parameters 
        self.no_of_tasks=0
        self.cumulative_delays=0
        self.no_of_tasks_utility_1=0
        self.no_of_task_failures=0
        self.no_of_task_success=0

    def reset(self):
        self.time_step=0
        self.Flag=False
        self.request_queue=[]
        self.network.computation_resources()
        self.network.initial_position()
        self.network.initial_speed()

        #analysis parameters 
        self.no_of_tasks=0
        self.cumulative_delays=0
        self.no_of_tasks_utility_1=0
        self.no_of_task_failures=0
        self.no_of_task_success=0

        #generating path loss matrix
        #path loss will be stored in self.network.path_loss
        self.network.path_loss_matrix()

        #generating set of clients
       # self.network.activity()

        self.network.activity_history = self.network.active + 0.001
        self.network.activity_history_normalized=self.network.active + 0.001

        #count variable to count no of clients
        #count=np.count_nonzero(self.network.active == 1)

        while not self.request_queue:
            self.network.activity()
            count=np.count_nonzero(self.network.active == 1)
            #obtain Client Vehicle IDs
            for i in range(self.N):
                #check if vehicle is client
                if self.network.active[i]:

                    task_profile=self.network.task()
                    count_norm=count/self.N
                    state=np.concatenate(([count_norm],task_profile,self.network.activity_history_normalized,self.network.path_loss_normalized[i].flatten()))
                    self.request_queue.append((i,state))
                    print(f"Appending state for client {i}")

                    #update action space as: Action Space - Client Vehicle ID
                    if i in self.action_space:
                        self.action_space.remove(i)

        initial_state = self.request_queue[0][1] if self.request_queue else np.zeros(self.state_space_shape)
        print(f"Initial state after reset: {initial_state}")

        self.curr_state = initial_state

        return self.curr_state

    def step(self,action):

        Client_ID = int(self.request_queue[0][0])
        print(Client_ID)
        Server_ID = int(action*(self.N-1)) # Resizing action
        No_Clients = int(self.curr_state[0]*self.N)
        task_data_size = self.curr_state[1]
        task_computation_cycles = self.curr_state[2]
        delay = self.network.calculate_delay( No_Clients, Client_ID, Server_ID, task_data_size,task_computation_cycles)

        self.no_of_tasks+=1
        self.cumulative_delays+=delay
        print("delay",delay)

        lower_tolerance=self.curr_state[3]
        upper_tolerance=self.curr_state[4]

        print("Lower tolerance",lower_tolerance)
        print("Upper tolerance",upper_tolerance)

        reward=0

        if delay < lower_tolerance:
            reward = 1
        elif delay >= lower_tolerance and delay < upper_tolerance:
            reward = (upper_tolerance - delay) / (upper_tolerance - lower_tolerance)
        else:
            reward = 0

        if Client_ID == Server_ID:
            reward = 0
        
        if reward<=1 and reward>0:
            self.no_of_task_success+=1
        #for analysis 
        if reward == 1:
            self.no_of_tasks_utility_1+=1
        elif reward == 0:
            self.no_of_task_failures+=1

        # popping state of the Cient whose request is satisfied
        self.request_queue.pop(0)

        #following code is logic for generating next state
        #if task queue is not empty then next state is queue[0]
        if self.request_queue:

            next_state=self.request_queue[0][1]

        else:
            self.time_step+=1
            if self.time_step==self.no_time_steps:
                self.Flag=True
                next_state=np.zeros(self.state_space_shape)
                return next_state,reward,self.Flag

            #update action space
            self.action_space=[i for i in range(self.N)]

            self.network.update_position()
            self.network.update_speed()
            self.network.path_loss_matrix()

            self.network.activity()

            self.network.update_activity_history()

            count=np.count_nonzero(self.network.active == 1)

            for i in range(self.N):
                if self.network.active[i]:
                    task_profile=self.network.task()
                    count_norm=count/self.N
                    state=np.concatenate(([count_norm],task_profile,self.network.activity_history_normalized+0.001,self.network.path_loss_normalized[i].flatten()))
                    self.request_queue.append((i,state))
                    print(f"Appending state for client {i}")
                    #updating action space
                    if i in self.action_space:
                        self.action_space.remove(i)

            #set next state as top of queue
            if self.request_queue:
                next_state=self.request_queue[0][1]
            else:
                self.time_step+=1
                if self.time_step<self.no_time_steps:
                    while not self.request_queue:
                        self.network.activity()
                        self.network.update_activity_history()
                        count=np.count_nonzero(self.network.active == 1)
                        self.action_space=[i for i in range(self.N)]
                        for i in range(self.N):
                            if self.network.active[i]:
                                task_profile=self.network.task()
                                count_norm=count/self.N
                                state=np.concatenate(([count_norm],task_profile,self.network.activity_history_normalized+0.001,self.network.path_loss_normalized[i].flatten()))
                                self.request_queue.append((i,state))
                                print(f"Appending state for client {i}")
                                if i in self.action_space:
                                    self.action_space.remove(i)
                    next_state=self.request_queue[0][1]
                else:
                    self.Flag=True
                    next_state=np.zeros(self.state_space_shape)

        self.curr_state=next_state
        print("TIME STEP",self.time_step)

        return next_state, reward, self.Flag
