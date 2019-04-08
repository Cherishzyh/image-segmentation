import h5py
import numpy as np
import os

file_path = 'H:/Input_1_Output_1/testing/data1.h5'
f = h5py.File(file_path, 'r')
f.keys()
