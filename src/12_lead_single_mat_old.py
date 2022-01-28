import pandas as pd
import scipy.io
import numpy as np
import wfdb
import wfdb.processing as prcs
import glob

dataDir = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/WFDB/"
new_ds_path = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/WFDB_single_leads/"
FS = 300
WINDOW_SIZE = 60*FS

files = sorted(glob.glob(dataDir+"*.mat"))
trainset = np.zeros((len(files*12),WINDOW_SIZE))
count = 0
indv_data = [None] * 12
indv_labels = [None] * 12

np.seterr(all='raise')

for f in files:
    record = f[:-4]
    record = record[-6:]
    # Loading
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    #print('Loading record {}'.format(record))
    count_1 = 0
    for arr in mat_data['val']:
      data = arr.squeeze()
      # Preprocessing
      # print('Preprocessing record {}'.format(record))       
      data = np.nan_to_num(data) # removing NaNs and Infs
      data = data - np.mean(data)
      div = np.std(data)
      if(count == 152662):
        div = 1
      data = data/div
      data = prcs.resample_sig(data, 500, FS)[0]
      trainset[count,:min(WINDOW_SIZE,len(data))] = data[:min(WINDOW_SIZE,len(data))].T
      #indv_data[count_1].append(trainset)
      #trainset[count,:min(WINDOW_SIZE,len(data))] = data[:min(WINDOW_SIZE,len(data))].T # padding sequence
      count += 1
      count_1 += 1

import csv
curr_csv = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/backup_csv/12_lead_single_mat_old2.csv"
#csvfile = list(csv.reader(open(dataDir+'REFERENCE.csv')))
csvfile = list(csv.reader(open(curr_csv)))
traintarget = np.zeros((trainset.shape[0],3))
print(len(trainset))
print(len(traintarget))
print(len(csvfile))
classes = ['N', 'A','O']
count = 0
for row in range(len(csvfile)):
    traintarget[row,classes.index(csvfile[row][1])] = 1
    #print("thisone ", traintarget[count])
    count += 1

for i in range(10):
 print("asdasd", traintarget[i])

print(len(traintarget))
  
# Saving both
for i in range(12):
  sigs = []
  labels = []
  print(len(traintarget))
  for j in range(len(traintarget) // 12):
    labels.append(traintarget[j*12+i])
    print("for ", traintarget[j*12+i])
    sigs.append(trainset[j*12+i])
  
  scipy.io.savemat('/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/WFDB_testset_mat/12_testingset_' + str(i) + '.mat',mdict={'trainset': sigs, 'traintarget' : labels})
