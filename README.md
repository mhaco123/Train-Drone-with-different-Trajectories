# Train-Drone-with-different-Trajectories
In order to examine and improve the learning process from simple to complex, a criterion has been set. In the first step, using PPO learning, we teach the aim of floating a bird with three degrees of freedom at a certain point. In the next step, the learning agent is changed from PPO to DDPG so that the efficiency of the algorithms can be compared. Finally, we change the environment to train a bird agent with six degrees.
In the first step, the only change that has been made compared to the previous training is the change of reward in line with the training of the bird to float. At this point, we change the reward function.
%% Reward customize
current_position = sqrt(x_^2 + y_^2);
target_position = 20;
reward = 1 - .1*((abs(current_position - target_position)))^2;
At first, it is assumed that the starting point of the agent is a random point. D and the assumed motor have the ability to move in the direction of the circular axes, and the movement of the operator is done only in the direction of the Y axis. After learning and applying environmental conditions, simulation has been done. As a result, trust actions include the following pairs of actions. Here, L=0, M=5.0 and H=0.1 are the normalized thrust values for each engine.
