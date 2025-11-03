import h5py
import numpy as np 
import pandas as pd 
import datetime

def matlab_datenum_to_python_datetime(dn):
    """
    HELPER FUNCTION
    Accepts a variable of time matlab datenum and converts it to python datetime format.
    Returns python datetime variable for downstream processing
    """
    return datetime.datetime.fromordinal(int(dn))+datetime.timedelta(days=dn%1)-datetime.timedelta(days=366)

def read_ds(f, group, key):
    """
    HELPER FUNCTION
    Safely reads a dataset and returns none if key is missing
    """
    if key in group:
        ref = group[key][0][0]
        return np.array(f[ref]).flatten()
    return None

def read_matlab_string(f, ref):
    """
    HELPER FUNCTION
    Decodes MATLAB character array stored in HDF5 
    """"
    arr = f[ref][:]
    if arr.ndim == 2:
        arr = arr.flatten()
    return "".join(chr(x) for x in arr)
    
