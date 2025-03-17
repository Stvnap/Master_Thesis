###################################################################################################################################
# File after Dataset_preprocessing.py
# This file is used to create a DNN model using the preprocessed dataset



import os
import numpy as np
import tensorflow as tf
import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# Load the preprocessed dataset

df=pd.read_csv('datatest1.csv')


# Split the dataset into training and testing sets




