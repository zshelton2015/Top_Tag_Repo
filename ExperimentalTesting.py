#/usr/bin/env

#This program is for testing a trained BTD
# By Zach Shelton
# 9/9/2021
# Running this will test on a f
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Data is stored in pandas -> Each 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from numpy.random import choice
import argparse
parser = argparse.ArgumentParser(description='run boosted decision tree on data, note this file grabs only the data not validation this is an experimental set')
parser.add_argument('file', metavar='f', type=str)
parser.add_argument('BTD',metavar='d', type=str)
parser.add_argument('result',metavar='d', type=str)
args=parser.parse_args()

xg_reg=xgb.load_model(args.BTD)

rawdata=pandas.read_csv(args.file)
etruth=rawdata[["event","truth"]]
cleandata=rawdata.drop(["event","truth"],axis=1)
Dexp=xgb.DMatrix(data=cleandata)
predictions=xg_reg.predict(Dexp)
preddf=pd.Series(predictions)
preddf.to_csv("ExperimentalPred/%s.csv"%args.result)