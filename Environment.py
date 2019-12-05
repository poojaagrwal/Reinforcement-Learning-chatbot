#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:15:45 2019

@author: pagarwal
"""
import numpy as np
import os
import pandas as pd

import gensim
from nltk import word_tokenize

# Load pre-trained Word2Vec model created by the create_model class
os.chdir("/Users/pagarwal/Downloads/DRL-Chatbot-master/Vec_Models")
model = gensim.models.Word2Vec.load('conversations')
print(model)
print(model.vector_size)

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


class Environment:
	

	def create_conversation(self):
		# This creates the conversation for the user simulator

		# Get the greeting
		hello = greetings[np.random.randint(0,len(greetings))]

		# Select a location and a place
		place_n = np.random.randint(0,(len(place)))
		location_n = np.random.randint(0,(len(location)))

		# Select the relevant corresponding answer based on the place and the location
		# e.g. "Italian" and "Expensive" selects an "Expensive Italian Restaurant"
		# as the answer 
		answer_n = ((len(location)*place_n) + (location_n +1)) - 1

		# Set the answer in the conversation
		self.answer = answers[answer_n]

		# Create conversation array
		self.conversation = [hello, place[place_n], location[location_n], self.answer]

		# Count the correct answers in the conversation - for issuing a reward
		self.correct_answers = 0

		return self.conversation

	def get_state(self, curr_question, prev_question = ""):
		

		# For each modelled word in the sentence, sum the vectors
		# NOTE: NOT USED as vector size = 1
		vectors = [model.wv[w] for w in word_tokenize(curr_question.lower())
					if w in model.wv]
		state = np.zeros(model.vector_size)
		# Get the average of the vector sum
		state = (np.array([sum(x) for x in zip(*vectors)])) / state.size
		
		# If there is a previous question
		if(prev_question != ""):
			# Same as above
			vectors2 = [model.wv[w] for w in word_tokenize(prev_question.lower())
					if w in model.wv]
			last_state = np.zeros(model.vector_size)
			last_state = (np.array([sum(x) for x in zip(*vectors2)])) / last_state.size

		# If not, just set the vector to 0 (first question in conversation)
		else:
			last_state = np.zeros([1])

		# Add the current state and the previous together to form one state
		this_state = np.concatenate([state, last_state])

		return this_state

	def total_actions(self):
		# Add the action columns
		num_actions = (len(greetings_answer) + len(place_answer) + len(answers))

		return num_actions

	def action_space_size(self, curr_question):
		# Get the number of actions (replies)

		# Add the answer columns in conversations.csv
		num_actions = 0
		if(curr_question == 0):
			num_actions = len(greetings_answer)
		elif(curr_question == 1):
			num_actions = len(place_answer)
		else:
			num_actions = len(answers)

		return num_actions

	def step(self, curr_question, action):
		
		# The possible actions of the chatbot
		possible_actions = greetings + place + answers

		
		if(self.conversation[curr_question] == possible_actions[action]):
			self.correct_answers+=1
			reward = 0.2
			end_episode = False

		# When the action taken == 2 (location)
		elif(curr_question == 2):\
			# Correct answer for the ENTIRE conversation
			if(self.conversation[3] == possible_actions[action]):
				reward = 0.2
				# Check if the rest of the conversation was also correct
				if(self.correct_answers == 2):
					# Fully correct sequence. +1 reward total
					reward = 0.6
				end_episode = True
			else:
				# Incorrect
				reward = 0.0
				end_episode = True
				
		# If the answer was incorrect
		else:
			reward = 0.0
			end_episode = False

		return reward, end_episode
