#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:17:55 2019

@author: pagarwal
"""

import sys

sys.path.append('/Users/pagarwal/Downloads/DRL-Chatbot-master/')
from Environment import Environment
from ReplayMemory import ReplayMemory
from NeuralNetwork import NeuralNetwork

import os
import time
import pandas as pd
import numpy as np

os.chdir("/Users/pagarwal/Downloads/DRL-Chatbot-master/Conversations")
textdata = pd.read_csv('Conversations.csv')
greetings = textdata['Greetings'].values.tolist()
greetings_answer = textdata['Greetings_A'].values.tolist()
place = textdata['Place'].values.tolist()
place_answer = textdata['Place_A'].values.tolist()
location = textdata['Location'].values.tolist()
answers = textdata['Answers'].values.tolist()

# Remove NaN values in columns
greetings = [x for x in greetings if str(x) != 'nan']
greetings_answer = [x for x in greetings_answer if str(x) != 'nan']
place = [x for x in place if str(x) != 'nan']
place_answer = [x for x in place_answer if str(x) != 'nan']
location = [x for x in location if str(x) != 'nan']

class Agent:


	def __init__(self, training):

		# Create the environment
		self.environment = Environment()

		# Training or testing
		self.training = training

		# Set the initial training epsilon
		self.epsilon = 0.10

		# Get the number of actions for storing memories and Q-values etc.
		total_actions = self.environment.total_actions()
        
		# Training or testing
		if self.training:
			# Training : Set a learning rate
			self.learning_rate = 1e-2

			# Training: Set up the replay memory
			self.replay_memory = ReplayMemory(size=1000, num_actions=total_actions)

		else:
			# Testing: These are not needed
			self.learning_rate = None
			self.replay_memory = None

		# Create the neural network
		self.neural_network = NeuralNetwork(num_actions=total_actions, replay_memory=self.replay_memory)

		# This stores the rewards for each episode
		self.rewards = []


	def get_action(self, q_values, curr_question):
		"""
		Description:
			Use the current epsilon greedy to select an action
		Parameters:
			q_values: q_values at current state
			iteration_count: count of processed states
			training: training or testing
		Return:
			action: the selected reply
		"""

		# This is used when a random reply is selected. It encourages more efficient interactions when selecting randomly
		# as it only lets the chatbot select from the correct column (this speeds up training a bit by letting the 
		# agent find the correct answers a little more easily).
		if(curr_question) == 0:
			low = 0
			high = len(greetings)
		elif(curr_question) == 1:
			low = len(greetings)
			high = len(greetings) + len(place)
		else:
			low = len(greetings) + len(place)
			high = len(greetings) + len(place) + len(answers)

		# self.epsilon = probability of selecting a random action
		if np.random.random() < self.epsilon:
			# Random sentence reply in the correct column
			action = np.random.randint(low=low, high=high)
		else:
			# Select highest Q-value
			action = np.argmax(q_values)

		return action

	def get_testing_action(self, q_values):

		# During testing, always select the maximum Q-value
		action = np.argmax(q_values)

		return action

	def run(self, num_episodes=1000000):
		"""
		Description:
			Run the agent in either training or testing mode
		Parameters:
			num_episodes: The number of episodes the agent will run for in training mode
		"""

		if self.training:

			# Reset following loop
			end_episode = True

			# Counter for the states processed so far
			count_states = 0

			# Counter for the episodes processed so far
			count_episodes = 0

			# Counter for which step of the conversation (question)
			conversation_step = 0

			while count_episodes <= num_episodes:
				if end_episode:
					# Generate new conversation for the new episode
					conversation = self.environment.create_conversation()

					# The number of questions for this episode
					num_questions = len(conversation)

					# Reset conversation step
					conversation_step = 0				

					# Increment count_episodes as it is the end of the conversation
					count_episodes += 1

					# Reset episode reward
					reward_episode = 0.0

					if count_episodes > num_episodes:
						self.neural_network.save(count_states)

				if(conversation_step == 0):
					# First step in conversation. No previous question
					state = self.environment.get_state(curr_question = conversation[conversation_step])
				else:
					# Pass in the prev
					prev_question_idx = conversation_step - 1
					prev_question = conversation[prev_question_idx]
					state = self.environment.get_state(curr_question = conversation[conversation_step], prev_question = prev_question)

				# Estimate Q-Values for this state
				q_values = self.neural_network.get_q_values(states=[state])[0]

				# Determine the action
				action = self.get_action(q_values=q_values, curr_question = conversation_step)
				
				# Use action to take a step / reply
				reward, end_episode = self.environment.step(curr_question = conversation_step, action=action)

				# Increment to the next conversation step
				conversation_step += 1

				# Add to the reward for this episode
				reward_episode += reward

				# Increment the episode counter for calculating the control parameters
				count_states += 1

				# Add this memory to the replay memory
				self.replay_memory.add_memory(state=state,q_values=q_values,action=action,reward=reward,end_episode=end_episode)

				if self.replay_memory.is_full():
					# If the replay memory is full, update all the Q-values in a backwards sweep
					self.replay_memory.update_q_values()

					# Improve the policy with random batches from the replay memory
					self.neural_network.optimize(learning_rate=self.learning_rate, current_state=count_states)

					# Reset the replay memory
					self.replay_memory.reset_size()

				# Add the reward of the episode to the rewards array 
				if end_episode:
					self.rewards.append(reward_episode)

				# Reward from previous episodes (mean of last 30)
				if len(self.rewards) == 0:
					# No previous rewards
					reward_mean = 0.0
				else:
					# Get the mean of the last 30
					reward_mean = np.mean(self.rewards[-30:])

				if end_episode:
					# Print statistics
					statistics = "{0:4}:{1}\tReward: {2:.1f}\tMean Reward (last 30): {3:.1f}\tQ-min: {4:5.7f}\tQ-max: {5:5.7f}"
					print(statistics.format(count_episodes, count_states, reward_episode, reward_mean, np.min(q_values), np.max(q_values)))


		# TESTING
		else:
			# Clear cmd window and print chatbot intro
			clear = lambda: os.system('cls')
			clear()

			# Load the conversation checkpoint generated by training
			self.neural_network.load()

			# Set the previous question to blank so it returns a word vector of 0
			previous_question = ""

			# Current question counter
			curr_question = 0

			while True:

				user_input = input("Me: ").lower()
				try:

					# Get the state for this input
					if(previous_question == ""):
						# First question
						state = self.environment.get_state(curr_question=user_input)
					else:
						state = self.environment.get_state(curr_question=user_input, prev_question=previous_question)

					print("STATE:",state)

					# Input this question into the neural network
					q_values = self.neural_network.get_q_values(states=[state])[0]

					# Store previous question
					previous_question = user_input

					# Possible actions of agent (replies)
					possible_actions = greetings_answer + place_answer + answers

					print("Q VALUES:",q_values)

					# Select an action based on the q-values
					action = self.get_testing_action(q_values = q_values)

					print("Chatbot: ", possible_actions[action])
					if(curr_question < 2):
						curr_question += 1
					else:
						print("*****END OF CONVERSATION. RESTARTING...*****")
						# Reset
						curr_question = 0
						previous_question = ""

				except:
					print("Sorry, I don't understand you.")

				
				if user_input == "bye":
					print("Chatbot signing off in 5...")
					time.sleep(5)
					break