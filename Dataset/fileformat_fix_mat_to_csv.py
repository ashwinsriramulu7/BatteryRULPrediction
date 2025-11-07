import pandas as pd 
import numpy as np 
import scipy.io as sio
import os

#print path of all subfolders
datadir = '/home/ashwin-sriramulu/Documents/work/BatteryRULPrediction/Dataset/Raw_Mat_Data'
for folder in os.scandir(datadir):
    print(f"Directory Found: {folder.path}")

#print files found per folder
for folder in os.scandir(datadir):
    print(f"Directory path: {folder.path}")
    for file in os.listdir(folder):
        print(f"\tfilename: {file}")

#use scipy.io to load one .mat file and understand dataset structure
mat_data = sio.loadmat('/home/ashwin-sriramulu/Documents/work/BatteryRULPrediction/Dataset/Raw_Mat_Data/1. BatteryAgingARC-FY08Q4/B0005.mat')
print(mat_data.keys())

#use scipy.io to convert to dataframe
print