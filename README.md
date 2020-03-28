# BotBuddy-small_conversational_chatbot
Driven by a nostalgic chatbot of a chat messenger that was called Nimbuzz, i tried to build a small conversational chatbot that acts as a 
buddy for chitchating or killing time with a funny bot that knows about AI, COMPUTERS and bunch of other stuff.

# Overview
It is a retrieval bot that uses keras deep learning model to first classify the category or intent of the incoming message, and then choose 
a random response of a set of responses to the question in the dataset having the highest cosine similarity with the incoming message. 

# Dataset 
ChatterBot English corpus, source: https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data                       

I did modify a little on the dataset, its up to you to modify and add whatever you want
# Usage
run the following files in the same order:
* Create_Dataset ; which creates dataset of dictionaries from YAML files for convenience of data structures
* Train_Chatbot ; training the model
* Chat_With_Bot ; initialize the chatbot gui
