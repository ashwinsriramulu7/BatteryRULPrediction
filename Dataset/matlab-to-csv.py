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

