"""
############################################################################################
Main Classes:
	Environment				The environment for the agent to explore
	ReplayMemory				The memory which holds the states, action and Q-values	
	NeuralNetwork			The neural network for estimating Q-values
	Agent					The agent to learn the environment		
"""

import os

import pandas as pd
import numpy as np
import gensim

import argparse

from timeit import default_timer as timer
import matplotlib.pyplot as plt
from Agent import Agent

import sys

sys.path.append('/Users/pagarwal/Downloads/DRL-Chatbot-master/*')

############################################################################################

# Set up the checkpoint and log directories
checkpoint_directory = '/Users/pagarwal/Downloads/DRL-Chatbot-master/Checkpoints'
log_directory = '/Users/pagarwal/Downloads/DRL-Chatbot-master/Logs'

# Load pre-trained Word2Vec model created by the create_model class
os.chdir("/Users/pagarwal/Downloads/DRL-Chatbot-master/Vec_Models")
model = gensim.models.Word2Vec.load('conversations')

# Load the conversation headers from the CSV file
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



if __name__ == '__main__':
	# Running the system from command line

	# Parsing for command line
	description = "Q-Learning chatbot"

	# Create parser and add arguments
	parser = argparse.ArgumentParser(description=description)

	# Training argument: add "-training" to run training 
	parser.add_argument("-training", required=False,
						dest='training', action='store_true',
						help="train or test agent")

	# Parse the args
	args = parser.parse_args()
	training = args.training

	# Take note of the time taken to train the system
	t1 = timer()

	# Create and run the agent
	agent = Agent(training=training)
	agent.run()

	# Calculate time taken
	t2 = timer()
	time_taken = ((t2 - t1)/60)

	# Get the rewards
	rewards = agent.rewards

	# Print statistics about the rewards
	if training:
		print("#############################################################")
		
		print("Observations for {0} episodes:".format(len(rewards)))
		
		print("No. of correct conversation sequences:\t", 	rewards.count(np.max(rewards)))
		
		print("First occurrence of max reward:\t\t", 			rewards.index(np.max(rewards)))
		
		print("Mean Reward:\t\t\t",				np.mean(rewards))
		
		print("Time taken(minutes):\t\t\t",			time_taken)

		# Plot the total average reward over time
		rewards_plot = []
		for x in range(len(rewards)):
			if (x>0 and (x % 10000 == 0)):
				# This gets the average reward of the last 10000 results
				values = np.mean(rewards[(x-10000):x])
				rewards_plot.append(values)


		# Plot the reward over time
		plt.title("Total Mean Reward Over Time")
		plt.ylabel("Mean Reward")
		plt.xlabel("Number of episodes /1000 ")
		plt.plot(rewards_plot, 'r')
		plt.show()
