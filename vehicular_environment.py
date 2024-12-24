# Final Code, all chnages made, everything works
# V5.0
import numpy as np
import math
from scipy.stats import nakagami
class VehicularNetwork:
    def __init__(self,N):
        self.N=N
        self.computation_cycles=np.zeros(self.N)
        self.active=np.zeros(self.N)
        self.position=np.zeros(self.N)
        self.speed=np.zeros(self.N)
        self.path_loss=np.zeros((self.N,self.N))
        self.path_loss_normalized=self.path_loss
        self.time_slot_duration=2 # Duration of time slot: 2s
        self.Transmitter_Power=0.1
        self.bandwidth=10e9 #10MHz
        self.V2V_bandwidth=180e3 #180KHz
        self.activity_history=np.zeros(self.N)
        #self.activity_history.fill(0.001)
        self.activity_history_normalized=self.activity_history

    def computation_resources(self):

        self.computation_cycles=np.round(np.random.uniform(1.5,3,self.N),1)
        for i in range(self.N):
            cycles=self.computation_cycles[i]
            self.computation_cycles[i]=cycles* (10**9)
        return self.computation_cycles

    def activity(self):

        self.active=np.random.choice([0,1],p=[0.80,0.20], size=self.N)
        return self.active

    def initial_position(self):

        self.position=np.random.uniform(0,500,self.N)
        return self.position

    def initial_speed(self):

        self.speed=np.random.uniform(3,6,self.N)
        return self.speed

    def update_speed(self):

        for i in range(self.N):
            self.speed[i]+=np.random.normal(0,0.5)
        return self.speed

    def update_position(self):

        for i in range(self.N):
            self.position[i]+=self.speed[i]*self.time_slot_duration
        return self.position

    def calculate_path_loss(self,distance):

        #in case distance is 0, avoids any math errors
        if distance == 0:
            distance=1

        frequency=5.9e9
        d0=1
        n=3 #path loss exponent
        m=3
        FSPL=20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55
        DPL=10 * n * np.log10(distance/d0)
        nakagami_envelope = nakagami.rvs(m)
        phase = np.random.uniform(0, 2 * np.pi)
        nakagami_fading = nakagami_envelope * (np.cos(phase) + 1j * np.sin(phase))
        nakagami_fading_db = 10 * np.log10(np.abs(nakagami_fading)**2)
        total_path_loss = FSPL + DPL + nakagami_fading_db

        return total_path_loss
    
    def normalize_matrix(self, matrix):

        min_val = np.min(matrix)
        max_val = np.max(matrix)

        if max_val - min_val == 0:
            return matrix + 0.0001

        normalized_matrix = (matrix - min_val) / (max_val - min_val)

        return normalized_matrix

    def path_loss_matrix(self):

        for i in range(self.N):
            for j in range(self.N):
                if i==j:
                    self.path_loss[i][j]=0
                    continue
                distance=np.linalg.norm(self.position[i]-self.position[j])
                self.path_loss[i][j]=self.calculate_path_loss(distance)

        # Normalizing Path Loss Matrix
        self.path_loss_normalized = self.normalize_matrix(self.path_loss)

        return self.path_loss

    def calculate_delay(self, Total_Clients, client, server, task_data_size,task_computation_cycles):

        # Converting Normalized values to Original Values for Calculation
        task_data_size*=10e6
        task_computation_cycles*=1e9

        path_loss = 10 ** (self.path_loss[client][server] / 10)
        noise_power_spectral_density = 4.0e-21
        transmission_rate = (50//Total_Clients) * self.V2V_bandwidth * np.log2(1+ (self.Transmitter_Power* path_loss) / ( noise_power_spectral_density *self.V2V_bandwidth))
        transmission_delay = task_data_size / transmission_rate
        computation_delay = task_computation_cycles/self.computation_cycles[server] 
        total_delay=computation_delay + transmission_delay

        print("transmission_rate",transmission_rate)
        print("transmission delay",transmission_delay)
        print("Computation_delay",computation_delay)
        print("Total_delay",total_delay)

        return total_delay

    def task(self):

        data=np.random.choice([5e6,7e6,10e6]) #in Mbits
        computation_cycles=np.random.choice([0.2e9,0.3e9,0.4e9,0.5e9,1e9]) #in GHz
        #lower_delay=np.round(np.random.uniform(0.35,0.5),2) #in seconds
        lower_delay=np.round(np.random.uniform(0.15,0.35),2)
        upper_delay=lower_delay + 0.15

        # Normalizing Task profile parameters
        data=data/10e6
        computation_cycles=computation_cycles/1e9

        return [data,computation_cycles,lower_delay,upper_delay]

    def update_activity_history(self):

        alpha = 0.65
        for i in range(self.N):
            self.activity_history[i]= alpha * self.active[i] + (1-alpha) * self.activity_history[i]

        self.activity_history_normalized= self.normalize_matrix(self.activity_history)

        return self.activity_history