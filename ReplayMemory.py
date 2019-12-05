#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:16:20 2019

@author: pagarwal
"""


import numpy as np

class ReplayMemory:

	def __init__(self, size, num_actions, discount_rate=0.95):

		# Size of the states (last 2 input sentences)
		self.state_shape = [2]

		# Previous states
		self.states = np.zeros(shape=[size] + self.state_shape, dtype=np.float)

		# Array for the Q-values corresponding to the states
		self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

		# Old Q-value array for comparing
		self.old_q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

		# Holds the actions corresponding to the states
		self.actions = np.zeros(shape=size, dtype=np.int)

		# Holds the rewards corresponding to the states
		self.rewards = np.zeros(shape=size, dtype=np.float)

		# Whether the conversation has ended (end of episode)
		self.end_episode = np.zeros(shape=size, dtype=np.bool)

		# Number of states
		self.size = size

		# Discount factor per step
		self.discount_rate = discount_rate

		# Reset the current size of the replay memory
		self.current_size = 0

	def is_full(self):
		#Used to check if the replay memory is full

		return self.current_size == self.size

	def reset_size(self):
		# Empty the replay memory
		self.current_size = 0

	def add_memory(self, state, q_values, action, reward, end_episode):
		# Add a state to the replay memory. The parameters are all the things we want to store

		# Move to current index and increment size
		curr = self.current_size
		self.current_size += 1

		# Store
		self.states[curr] = state
		self.q_values[curr] = q_values
		self.actions[curr] = action
		self.end_episode[curr] = end_episode
		self.rewards[curr] = reward
		# Clip reward to between -1.0 and 1.0
		#self.rewards[curr] = np.clip(reward, -1.0, 1.0)

	def update_q_values(self):
		# Update the Q-values in the replay memory

		# Keep old q-values
		self.old_q_values[:] = self.q_values[:]

		# Update the Q-values in a backwards loop
		for curr in np.flip(range(self.current_size-1),0):

			# Get data from curr
			action = self.actions[curr]
			reward = self.rewards[curr]
			end_episode = self.end_episode[curr]

			# Calculate Q-Value
			if end_episode:
				# No future steps therefore it is just the observed reward
				value = reward
			else:
				# Discounted future rewards
				value = reward + self.discount_rate * np.max(self.q_values[curr + 1])

			# Update Q-values with better estimate
			self.q_values[curr, action] = value

	def get_batch_indices(self, batch_size):
		# Get random indices from the replay memory (number = batch_size)

		self.indeces = np.random.choice(self.current_size, size=batch_size, replace=False)

	def get_batch_values(self):
		# Get the states and q values for these indeces

		batch_states = self.states[self.indeces]
		batch_q_values = self.q_values[self.indeces]

		return batch_states, batch_q_values