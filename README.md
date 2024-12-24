# DRL-based-Client-Server-Matching-in-V2V-Task-Offloading
Project Description:
With developments in Autonomous Vehicles, new applications have emerges that facilitate a smooother and safer driving experience, but these applications tend to be computing intensive and have strict deadlines for completion(maximum tolerable delay).However,due to the limited
onboard computing and storage resources, some computation-intensive tasks cannot be performed within the deadlines which needs the assistance of central cloud or an edge server. Offloading the task to a central cloud incurs a high transmission delay and offlading to edge server might not be effective. V2V Task offloading is a new paradigm where a resource constrained vehicles(Client Vehicles) offloads its task to a vehicle with idle resources(Server Vehicle)
This project address key issues in V2V Computation offlading. 
Code Description
buffer.py - Code for replay buffer for model training 
networks.py - Code for the Actor Critic and the Target Neural Networks
vehicular_environment.py - models the vehicular network
ddpg_torch.py - Code for the Agent and Network Training 
environment_py - Code that models the vehiclular network environment at each time step 
utils.py- Code to plot graph of scores 
NOTE: Here the vehicular_environment.py just sets up the environment, but enviornment.py models the network and its dynamics at each time step. 
