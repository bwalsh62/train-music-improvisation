# -*- coding: utf-8 -*-
"""
Improvise Solo with an LSTM Network
-----------------------------------
This implements a model that uses an LSTM network to learn musical style based 
on a library of music and generate new music based on the learned songs

Started April 2019
Initial draft upload 1.0 April 25, 2019

@author: benja

"""

#%% Import libraries
#-----------------
# Custom utilities for music processing
from utils.grammar import * 
from utils.music_utils import * 
from utils.data_utils import * 

# Custom functions for LSTM models built on Keras
from nn.nn_functions import *

# Keras for Neural Network Library
from keras.layers import Dense, LSTM, Reshape
from keras.optimizers import Adam

# Numpy for array math
import numpy as np

#%%------------------------- 
# Load music and pre-process
#---------------------------

print(".......................\n")
print("..Loading music.......")
print("\n.......................")

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

#%%-------------------------
# Build LSTM model
#---------------------------

# Define dimension of hidden layer
n_a = 64    

# Define layer objects in Keras

reshaper = Reshape((1, n_values))                        
LSTM_cell = LSTM(n_a, return_state = True)         
densor = Dense(n_values, activation='softmax')

# Implement LSTM model
#---------------------------
print(".......................\n")
print("..Building LSTM model .......")
print("\n.......................")
model = lstm_model(Tx = X.shape[1] , n_a = n_a, n_values = n_values) 

# Compile model
learning_rate = 0.1
beta1_adam = 0.9
beta2_adam = 0.999
decay=0.01
opt = Adam(lr=learning_rate, beta_1=beta1_adam, beta_2=beta2_adam, decay=decay)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#%%-------------------------
# Initialize model
#---------------------------
m = X.shape[0] # Number of training examples
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

# Fit model
#---------------------------
model.fit([X, a0, c0], list(Y), epochs=100)

# Create Model
#------------------
vals_to_gen = 50
inference_model = music_inference_model(LSTM_cell, densor, n_values = n_values, n_a = n_a, Ty = vals_to_gen )

#%% Generate Music
#------------------
print("........................\n")
print("..Generating music......")
print("\n........................")


bpm = 90       
song_fname_base = "mySong_gui_test"
song_fname = song_fname_base+"_bpm"+str(bpm)

out_stream = generate_music(inference_model, bpm=bpm, song_fname = song_fname)

# %% REFERENCES
'''

( Adapted from https://www.coursera.org/learn/nlp-sequence-models
  Week 1 assignment in Deep Learning Specialization by Andrew Ng )

The ideas presented in the original notebook came primarily from three computational music papers cited below. The implementation here also took significant inspiration and used many components from Ji-Sung Kim's github repository.

- Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
- Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
- Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
- François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)

François Germain also provided valuable feedback to Andrew Ng's Coursera assignment.
'''
