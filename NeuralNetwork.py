#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:17:04 2019

@author: pagarwal
"""
import tensorflow as tf
import os

# Set up the checkpoint and log directories
checkpoint_directory = '/Users/pagarwal/Downloads/DRL-Chatbot-master/Checkpoints'
log_directory = '/Users/pagarwal/Downloads/DRL-Chatbot-master/Logs'

class NeuralNetwork:
	
	def __init__(self, num_actions, replay_memory):

		# Reset default graph
		tf.reset_default_graph()

		# Path for saving checkpoints during training for testing
		self.checkpoint_dir = os.path.join(checkpoint_directory, "checkpoint")

		# Size of state (2x Word2Vec vectors)
		self.state_shape = [2]

		# Sample random batches
		self.replay_memory = replay_memory

		# Inputting states into the neural network
		with tf.name_scope("inputs"):
			self.states = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_shape, name='state')

		# Learning rate placeholder
		self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

		# Input the new Q-values placeholder that we want the states to map to
		self.q_values_new = tf.placeholder(dtype=tf.float32, shape=[None, num_actions], name='new_q_values')

		# Initialise weights close to 0
		weights = tf.truncated_normal_initializer(mean=0.0, stddev=1e-1)

		# Hidden Layer 1
		# Note: This takes the word2vec state as input
		layer1 = tf.layers.dense(inputs=self.states, name='hidden_layer1', units=20, kernel_initializer=weights, activation=tf.nn.relu)

		# Hidden Layer 2
		layer2 = tf.layers.dense(inputs=layer1, name='hidden_layer2', units=20, kernel_initializer=weights, activation=tf.nn.relu)

		# Output layer - estimated Q-values for each action
		output_layer = tf.layers.dense(inputs=layer2, name='output_layer', units=num_actions, kernel_initializer=weights, activation=None)

		# Set the Q-values equal to the output from the output layer
		with tf.name_scope('Q-values'):
			self.q_values = output_layer
			tf.summary.histogram("Q-values", self.q_values)

		# Get the loss
		# Note: This is the mean-squared error between the old and new Q-values (L2-Regression)
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.q_values - self.q_values_new), axis = 1))
			tf.summary.scalar("loss", self.loss)

		# Optimiser for minimising the loss (to learn better Q-values)
		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		# Create TF session for running NN
		self.session = tf.Session()

		# Merge all summaries for Tensorboard
		self.merged = tf.summary.merge_all()

		# Create Tensorboard session
		self.writer = tf.summary.FileWriter(log_directory, self.session.graph)

		# Initalialise all variables and run
		init = tf.global_variables_initializer()
		self.session.run(init)

		# For saving the neural network at the end of training (for testing)
		self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

	def close(self):
		# Close TF Session
		self.session.close()

	def get_q_values(self, states):
		# Calculate and return estimated Q-values for the given states
		
		# Get estimated Q-values from the neural network
		q_values = self.session.run(self.q_values, feed_dict={self.states: states})

		return q_values

	def optimize(self, current_state, learning_rate, batch_size=50):
		
		print("Optimizing the Neural Network with learning rate {0}".format(learning_rate))
		
		# Get random indices from the replay memory
		self.replay_memory.get_batch_indices(batch_size)

		# Get the corresponding states and q values for the indices
		batch_states, batch_q_values = self.replay_memory.get_batch_values()

		# Feed these values into the neural network and run one optimization and get the loss value
		current_loss, _ = self.session.run([self.loss, self.optimizer], feed_dict = {self.states: batch_states, self.q_values_new: batch_q_values, self.learning_rate: learning_rate})

		# Send the results to tensorboard
		result = self.session.run(self.merged, feed_dict={self.states: batch_states, self.q_values_new: batch_q_values, self.learning_rate: learning_rate})
		print("Current loss: ", current_loss)
		self.writer.add_summary(result, current_state)

	def save(self, count_states):
		# Save the completed trained network
		self.saver.save(self.session, save_path=self.checkpoint_dir, global_step=count_states)
		print("Checkpoint saved")

	def load(self):
		# Load the network for testing
		try:
			latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_directory)
			self.saver.restore(self.session, save_path=latest_checkpoint)
			print("\n\nLoaded neural network successfully. You can start talking to the chatbot!:")
		except:
			print("Could not find checkpoint")