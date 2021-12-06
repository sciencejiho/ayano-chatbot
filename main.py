import os
# os.environ['TF_CP_MIN_LOG_LEVEL'] = '2'
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)

# import NLTK, which is a Natural Language Tool Kit for building Pyhon programs to work with human language data.
import nltk
# import stemmers to remove morphological affixes from words, leaving only the word stem
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import time

# import the necessary libraries for discord support
import discord
from dotenv import load_dotenv

# import the necessary libraries for DNN visualization
# from tensorflow.keras.utils import plot_model
# import pydot
# import graphviz

with open("intents.json") as file:
    data = json.load(file)

# print(data)

# try to load the preexisting pickle data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            vocab = nltk.word_tokenize(pattern)
            words.extend(vocab)
            docs_x.append(vocab)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # print(words)
    # print(labels)
    # print(docs_x)
    # print(docs_y)

    # lowercase every words, and use stemmer to remove morphological affixes
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]

    # remove duplicate words, and make them a sorted list
    words = sorted(list(set(words)))
    # sort labels as well
    labaels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        # create a bag to track down which vocabulary exists in the given setence
        bag = []

        vocab = [stemmer.stem(w) for w in doc]

        for w in words:
            # if a vocabulary exists in the sentence, denote it as 1
            if w in vocab:
                bag.append(1)
            # else, denote it as 0
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    training - np.array(training)
    output = np.array(output)

# print(training)
# print(output)

print("Bot is loading...")
time.sleep(3)

# try to load the preexisting model
try:
    model.load("Ayano_Chatbot_model.tflearn")
except:
    # reset underlying data graph
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    # add hidden neural layer of 8 neurons
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    # use sofmax function (normalized exponential function) for our activation function in the output layer of neural network model
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net, tensorboard_verbose = 3)
    model.fit(training, output, n_epoch = 5000, batch_size = 8, show_metric = True, run_id = 'LOG_AYANO')
    model.save("Ayano_Chatbot_model.tflearn")

print("Bot is now loaded!")

def bag_of_words(sentence, words):
    # initialize list full of 0 for number of words given
    bag = [0 for _ in range(len(words))]

    # tokenize, lowercase, and stem the words in the given sentence
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            # if word exists in sentence, replace 0 to 1
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

# def chat_with_bot():
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "quit":
#             break
        
#         # calculate the probability of each possible tags
#         res = model.predict([bag_of_words(user_input, words)])[0]
#         # find the correct tag based on the probability
#         res_index = np.argmax(res)
#         tag = labels[res_index]

#         # set minimum confidence level
#         if res[res_index] > 0.7:
#             # using json file, print out suitable response
#             for t in data["intents"]:
#                 if t['tag'] == tag:
#                     responses = t['responses']

#             # print out one of possible responses in given tag randomly    
#             print(random.choice(responses))
#         else:
#             print("Hehe... I'm not smart after all, I don't quite understand. Sorry.")

def chat_with_bot_discord(message):
    # calculate the probability of each possible tags
    res = model.predict([bag_of_words(message, words)])[0]
    # find the correct tag based on the probability
    res_index = np.argmax(res)
    tag = labels[res_index]

    # set minimum confidence level
    if res[res_index] > 0.7:
        # using json file, print out suitable response
        for t in data["intents"]:
            if t['tag'] == tag:
                responses = t['responses']

        # print out one of possible responses in given tag randomly    
        final_answer = random.choice(responses)

    else:
        final_answer = "Hehe... I'm not smart after all, I don't quite understand. Sorry."

    # for debugging purposes only
    # converted_res = [str(elem) for elem in res]
    # final_answer = ", ".join(converted_res)    

    return final_answer

client = discord.Client()

load_dotenv()
TOKEN = os.getenv('TOKEN')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("$ayanobot"):
        response = chat_with_bot_discord(message.content[10:])
        await message.channel.send(response)

client.run(TOKEN)


# # clear screen before initiating the bot
# os.system('cls' if os.name == 'nt' else 'clear')

# print("Bot is ready!")
# print("")

# chat_with_bot()
