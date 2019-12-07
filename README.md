# Reinforcement-Learning-chatbot

• Team Neurons

<b>Abhishek Prabhudesai 

Pooja Agarwal 

Rohini Shimpatwar

Saurabh Aggarwal</b>


Reference:
https://github.com/KGSands
https://learning.oreilly.com/library/view/deep-reinforcement-learning/9781788834247/ch12.html
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter12

We would like to thank <b>Keelan KGSands</b> to provide us a baseline for our project. The fundamental of this chatbot implementation was inspired by his work and we were able to give it a shape according our need which was to provide the user of our application (crowd couting at a public plcae, Mall, stores, DMV etc.) an interface to give a crowd footfall estimation  using an DQN learned chatbot.



The application comprises of a chatbot, that aims to give a user appropriate answers based on their queries. A learning agent uses a deep Q learning neural network to get the maximum reward and reply at a given point of time. 

The application will be trained on a conversation csv, that will have all the possible conversations between chatbot and the user. The input to the neural network is a vector [S1, S2] which will be of size 2. The S1 will have the value of the current sentence or the question that will be entered by the user. To maintain the context, S2 will be storing the previous question entered by the user. 

An action space, will be an array of [0, 1 ...21] and an action will be selected out of this action space. The actions will be one of the possible answers from [A0 ...A21]. The maximum Q-value will be selected for the answer during the testing phase out of [Q0…Q21]. The Q-value will give us an estimate of the aggregated future reward . The reward will be an integer value, that will be sent over to the agent when it gets an action. 

• Environment

The main purpose of the environment class is to generate the random user conversation to train the chatbot agent and develop a reward function which takes current question and actions as the input and provides rewards as the output.

These randomly generated conversations will be used for training the neural network.

• Replay Memory

This is used to store the historical data so that it can be used during exploitation. 

A number of 2-dimensional arrays are maintained with action to q value mapping containing historic data.

Q-values are updated with the formula:
State Value = Reward + Discount_Rate* Max(Q value of next State)

• Neural Network

Structure of the Neural network required for estimating the Q value for the actions 

Input layer: 2 neurons corresponding to state(Current State and Previous State)
Hidden layer1: Dense layer of 20 neurons with Relu as an activation function 
Hidden layer2: Dense layer of 20 neurons with Relu as an activation function

Output Layer: Layer with 22 Neurons

• Agent

The main purpose of the Agent is to initialize the environment, train the Neural Network and take action - that is to choose the correct reply for the user question. 

All the hyperparameters required for training the neural network like learning rate, exploration parameter epsilon, Replay memory size are initialized in the agent class.




