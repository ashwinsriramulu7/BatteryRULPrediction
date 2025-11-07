import pandas as pd 
import numpy as np 
import scipy.io as sio
import os

datadir = '/home/ashwin-sriramulu/Documents/work/BatteryRULPrediction/Dataset/Raw_Mat_Data'
for folder in os.scandir(datadir):
    print(f"Directory Found{folder.path}")