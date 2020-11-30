#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:31:28 2020

@author: qingn
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle 
import timeit
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# Tensorflow 2.0 way of doing things
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential
import netCDF4
# Default plotting parameters
FONTSIZE = 18
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = FONTSIZE

# build model with keras includes three hidden layers and [8,8,7] neurals, default activation function is elu, loss function is mse...

def build_model(n_inputs, n_hidden, n_output, activation='elu', lrate=0.001):
    model = Sequential();
    model.add(InputLayer(input_shape=(n_inputs,)))
    model.add(Dense(n_hidden, use_bias=True, name="hidden_1", activation=activation))
    model.add(Dense(n_hidden, use_bias=True, name="hidden_2", activation=activation))
    model.add(Dense(n_hidden-1, use_bias=True, name="hidden_3", activation=activation))
    model.add(Dense(n_hidden, use_bias=True, name="hidden_4", activation=activation))
    model.add(Dense(n_hidden, use_bias=True, name="hidden_5", activation=activation))
#     model.add(Dense(n_hidden, use_bias=True, name="hidden_6", activation=activation))
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
    
    opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=opt)
    print(model.summary())
    return model
#%%
#fp = pd.read_csv("inputs.csv", "rb")
#foo = pickle.load(fp)
#fp.close()
data = netCDF4.Dataset('X_y.nc')
ins =  data['X'][:]
outts = np.array(data['y'])
outs = outts.reshape(2250,1)
#%%
start = timeit.default_timer()
history= model.fit(x=ins, y=outs, epochs=80, verbose=False)
end = timeit.default_timer()
print(str(end-start))
#%%
a = []
error = []
plt.figure()
for i in np.arange(7):

    start = timeit.default_timer() #Timming
    model = build_model(ins.shape[1], 8, outs.shape[1],activation='tanh')#, activation='sigmoid' # setup model
    history = model.fit(x=ins, y=outs, epochs=180, verbose=False) # run the model
    end = timeit.default_timer()
    print(str(end-start)) # How long has been used for each Independent run
# history = model.fit(x=ins, y=outs, epochs=8000, verbose=False,
#                     validation_data=(ins, outs),
#                    callbacks=[checkpoint_cb, early_stopping_cb])
    a.append(history.history['loss'])
    error_each = np.abs((outs-model.predict(ins))[:,0]) # error is defined in this way
    error.append(error_each)
    # Display
   
    plt.plot(history.history['loss'],label = 'independent_learning_run'+str(i))
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('epochs')