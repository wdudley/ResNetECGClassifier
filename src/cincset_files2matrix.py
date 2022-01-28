#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Convert multiple files from Physionet/Computing in Cardiology challenge into 
file single matrix. As input argument 

For more information visit: https://github.com/fernandoandreotti/cinc-challenge2017
 
 Referencing this work
   Andreotti, F., Carr, O., Pimentel, M.A.F., Mahdi, A., & De Vos, M. (2017). Comparing Feature Based 
   Classifiers and Convolutional Neural Networks to Detect Arrhythmia from Short Segments of ECG. In 
   Computing in Cardiology. Rennes (France).

--
 cinc-challenge2017, version 1.0, Sept 2017
 Last updated : 27-09-2017
 Released under the GNU General Public License

 Copyright (C) 2017  Fernando Andreotti, Oliver Carr, Marco A.F. Pimentel, Adam Mahdi, Maarten De Vos
 University of Oxford, Department of Engineering Science, Institute of Biomedical Engineering
 fernando.andreotti@eng.ox.ac.uk
   
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import scipy.io
import numpy as np
import glob

# Parameters
dataDir = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/" # <---- change!!
dataDir = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/training2017NoNoise/"
FS = 300
WINDOW_SIZE = 60*FS

'''
Idea on how to solve issue with loding the 12-lead dataset individually
1. Fix where the gen shit is saved because it bugs loading the .mat files in glob
2.  Create a loop to create the .mat files like in the loop below.
3. Save the .mat files with the naming convention I did when loading them into the other model
'''

## Loading time serie signals
files = sorted(glob.glob(dataDir+"*.mat"))
trainset = np.zeros((len(files),WINDOW_SIZE))
count = 0
for f in files:
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    #print('Loading record {}'.format(record))    
    data = mat_data['val'].squeeze()
    # Preprocessing
    #print('Preprocessing record {}'.format(record))       
    data = np.nan_to_num(data) # removing NaNs and Infs
    data = data - np.mean(data)
    data = data/np.std(data)
    trainset[count,:min(WINDOW_SIZE,len(data))] = data[:min(WINDOW_SIZE,len(data))].T # padding sequence
    count += 1
## Loading labels    
import csv
new_csv = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/new_REFERENCE-v3_2.csv"
#csvfile = list(csv.reader(open(dataDir+'REFERENCE.csv')))
csvfile = list(csv.reader(open(new_csv)))
traintarget = np.zeros((trainset.shape[0],3))
classes = ['N', 'A', 'O']
for row in range(len(csvfile)):
    traintarget[row,classes.index(csvfile[row][1])] = 1
    #print(traintarget)
            
# Saving both
scipy.io.savemat('/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/mat_saves/trainingset.mat',mdict={'trainset': trainset,'traintarget': traintarget})

# Stores train sets into an array. Should work just like above.
# PROBABLY NEED TO DELETE THE BELOW CODE
sigs_12_path = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/lead_sep/sigs/"
labels_12_path = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/lead_sep/labels/"

trainset_frame = []

for lead in range(12):
  data = np.load(sigs_12_path + "lead_" + str(lead+1) + "_sigs.npy")
  trainset = np.zeros((len(data),WINDOW_SIZE))
  trainset = []
  count = 0
  
  for arr in data:
    arr_new = np.nan_to_num(arr)
    arr_new = arr_new - np.mean(arr_new)
    arr_new = arr_new/np.std(arr_new)
    print(arr_new)
    print(min(WINDOW_SIZE,len(arr_new)))
    
    trainset[count,:min(WINDOW_SIZE,len(arr_new))] = arr_new[:min(WINDOW_SIZE,len(arr_new))].T # padding sequence
    
    count += 1
    
  trainset_frame.append(trainset)
  
for label_lead in range(12):
  data = np.load(labels_12_path + "lead_" + str(lead+1) + "_labels.npy")
  
  traintarget = np.zeros((trainset_frame[label_lead].shape[0],3))
  
  for row in range(len(trainset_frame[label_lead])):
    traintarget[row,classes.index(csvfile[row][1])] = 1
  
  scipy.io.savemat('/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/mat_saves/testingset_' + str(label_lead+1) + '.mat',mdict={'trainset': trainset_frame[label_lead],'traintarget': traintarget})
