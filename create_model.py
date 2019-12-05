
import os
import nltk
import gensim
import pandas as pd

nltk.download('punkt')

# Move to the conversations directory and read the file
os.chdir("/Users/pagarwal/Downloads/DRL-Chatbot-master/Conversations")
textdata = pd.read_csv("Conversations.csv")

# Create word corpus
greetings = textdata['Greetings'].values.tolist()
place = textdata['Place'].values.tolist()
location = textdata['Location'].values.tolist()
corpus = greetings + place + location

# Tokenize the words in the corpus
tokenise = [nltk.word_tokenize(str(sent).lower()) for sent in corpus]

# Create a word2vec model for the tokenized words
model = gensim.models.Word2Vec(tokenise, min_count=1, size=1)

# Save the model so it can be loaded in the main program
os.chdir("/Users/pagarwal/Downloads/DRL-Chatbot-master/Vec_Models")
model.save('conversations')