# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:33:34 2019

@author: benja

Implement the LSTM model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras model
"""

# Keras for Neural Network Library
from keras.layers import Dense, Input, LSTM, Reshape, Lambda
from keras.models import Model
from keras.utils import to_categorical

# Custom music utilities - from class but will remake later
from utils.music_utils import one_hot # borrowed from class - will recreate eventually
# Remake one_hot to be more general, not hard-code to 78, or just use from tensorflow

# Numpy for array math
import numpy as np

# Function define LSTM model

def lstm_model(Tx, n_a, n_values): 
    
    # Define layer objects in Keras
    reshaper = Reshape((1, n_values))                        
    LSTM_cell = LSTM(n_a, return_state = True)         
    densor = Dense(n_values, activation='softmax')
    
    # Define the input of your model with a shape 
    X = Input(shape=(Tx, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Initialize outputs to append to while iterating
    outputs = []
    
    # Loop through each time step
    for t in range(Tx):
        
        # Select the time step vector from X at time t
        x = Lambda(lambda x: X[:,t,:])(X)
        # Use reshaper to reshape x to be (1, n_values)
        x = reshaper(x)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Append output
        outputs.append(out)
        
    # Create model instance to return
    model = Model(inputs=[X,a0,c0], outputs=outputs)
        
    return model


def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer number of unique values
    n_a -- integer number of units in the LSTM_cell
    Ty -- integer number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of model 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # Initialize outputs to append to while iterating
    outputs = []
    
    # Loop through each time step
    for t in range(Ty):
        
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Apply Dense layer to the hidden state output of the LSTM_cell 
        out = densor(a)

        # Append the prediction to "outputs"
        outputs.append(out)
        
        # Select the next value according to "out" to eventually pass 
        # as the input to LSTM_cell on the next step.
        x = Lambda(one_hot)(out)#(n_values)
        
    # Create and return model instance  
    inference_model = Model(inputs=[x0,a0,c0], outputs=outputs)
        
    return inference_model

def predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cell
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    # Use inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, 2)
    # Convert indices to one-hot vectors
    results = to_categorical(indices, num_classes = None)
    
    return results, indices