'''
This function  function used for training and cross-validating model using. The database is not 
included in this repo, please download the CinC Challenge database and truncate/pad data into a 
NxM matrix array, being N the number of recordings and M the window accepted by the network (i.e. 
30 seconds).


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

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import gc
import itertools
from sklearn.metrics import confusion_matrix
import sys
sys.path.insert(0, './preparation')

# Keras imports
import keras
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
# from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.compat.v1 import ConfigProto, Session

from datetime import datetime
import helper_defs
import pandas as pd
now = str(datetime.now())

###################################################################
### Callback method for reducing learning rate during training  ###
###################################################################
class AdvancedLearnignRateScheduler(Callback):    
    '''
   # Arguments
       monitor: quantity to be monitored.
       patience: number of epochs with no improvement
           after which training will be stopped.
       verbose: verbosity mode.
       mode: one of {auto, min, max}. In 'min' mode,
           training will stop when the quantity
           monitored has stopped decreasing; in 'max'
           mode it will stop when the quantity
           monitored has stopped increasing.
   '''
    def __init__(self, monitor='val_loss', patience=0,verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__() 
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio
 
        #if mode not in ['auto', 'min', 'max']:
         #   warnings.warn('Mode %s is unknown, '
          #                'fallback to auto mode.'
           #               % (self.mode), RuntimeWarning)
           # mode = 'auto'
 
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
 
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        #if current is None:
         #   warnings.warn('AdvancedLearnignRateScheduler'
          #                ' requires %s available!' %
           #               (self.monitor), RuntimeWarning)
 
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0 
            self.wait += 1


###########################################
## Function to plot confusion matrices  ##
#########################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = np.around(cm, decimals=3)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion.eps', format='eps', dpi=1000)


#####################################
## Model definition              ##
## ResNet based on Rajpurkar    ##
################################## 
def ResNet_model(WINDOW_SIZE):
    # Add CNN layers left branch (higher frequencies)
    # Parameters from paper
    INPUT_FEAT = 1
    OUTPUT_CLASS = 3    # output classes

    k = 1    # increment every 4th residual block
    p = True # pool toggle every other residual block (end with 2^8)
    convfilt = 64
    convstr = 1
    ksize = 16
    poolsize = 2
    poolstr  = 2
    drop = 0.5
    
    # Modelling with Functional API
    #input1 = Input(shape=(None,1), name='input')
    input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')
    
    ## First convolutional block (conv,BN, relu)
    x = Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(input1)                
    x = BatchNormalization()(x)        
    x = Activation('relu')(x)  
    
    ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
    # Left branch (convolutions)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x)      
    x1 = BatchNormalization()(x1)    
    x1 = Activation('relu')(x1)
    x1 = Dropout(drop)(x1)
    x1 =  Conv1D(filters=convfilt,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)
    x1 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x1)
    # Right branch, shortcut branch pooling
    x2 = MaxPooling1D(pool_size=poolsize,
                      strides=poolstr)(x)
    # Merge both branches
    x = keras.layers.add([x1, x2])
    del x1,x2
    
    ## Main loop
    p = not p 
    for l in range(15):
        
        if (l%4 == 0) and (l>0): # increment k on every fourth residual block
            k += 1
             # increase depth by 1x1 Convolution case dimension shall change
            xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
        else:
            xshort = x        
        # Left branch (convolutions)
        # notice the ordering of the operations has changed        
        x1 = BatchNormalization()(x)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 =  Conv1D(filters=convfilt*k,
               kernel_size=ksize,
               padding='same',
               strides=convstr,
               kernel_initializer='he_normal')(x1)        
        if p:
            x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)                

        # Right branch: shortcut connection
        if p:
            x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
        else:
            x2 = xshort  # pool or identity            
        # Merging branches
        x = keras.layers.add([x1, x2])
        # change parameters
        p = not p # toggle pooling

    
    # Final bit    
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    x = Flatten()(x)
    #x = Dense(1000)(x)
    #x = Dense(1000)(x)
    out = Dense(OUTPUT_CLASS, activation='softmax')(x)
    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()
    #sequential_model_to_ascii_printout(model)
    plot_model(model, to_file='model.png')
    return model

###########################################################
## Function to perform K-fold Crossvalidation on model  ##
##########################################################
def model_eval(X,y):
    batch =64
    epochs = 30    #  CHANGE THIS LATER
    rep = 1         # K fold procedure can be repeated multiple times
    Kfold = 5
    Ntrain = 262044 // 12 # number of recordings on training set, 8243 for single-lead
    Nsamp = int(Ntrain/Kfold) # number of recordings to take as validation        
   
    # Need to add dimension for training
    X = np.expand_dims(X, axis=2)
    classes = ['N', 'A', 'O']
    Nclass = len(classes)
    cvconfusion = np.zeros((Nclass,Nclass,Kfold*rep))
    cvscores = []       
    counter = 0
    # repetitions of cross validation
    for r in range(rep):
        print("Rep %d"%(r+1))
        # cross validation loop
        for k in range(Kfold):
            print("Cross-validation run %d"%(k+1))
            # Callbacks definition
            callbacks = [
                # Early stopping definition
                EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                # Decrease learning rate by 0.1 factor
                AdvancedLearnignRateScheduler(monitor='val_loss', patience=1,verbose=1, mode='auto', decayRatio=0.1),            
                # Saving best model
                ModelCheckpoint('weights-best_k{}_r{}.hdf5'.format(k,r), monitor='val_loss', save_best_only=True, verbose=1),
                ]
            # Load model
            model = ResNet_model(WINDOW_SIZE)
            
            # split train and validation sets
            idxval = np.random.choice(Ntrain, Nsamp,replace=False)
            idxtrain = np.invert(np.in1d(range(X_train.shape[0]),idxval))
            ytrain = y[np.asarray(idxtrain),:]
            Xtrain = X[np.asarray(idxtrain),:,:]         
            Xval = X[np.asarray(idxval),:,:]
            yval = y[np.asarray(idxval),:]
            
            print(ytrain)
            
            # Train model
            model.fit(Xtrain, ytrain,
                      validation_data=(Xval, yval),
                      epochs=epochs, batch_size=batch,callbacks=callbacks)
            
            # Evaluate best trained model
            # THESE WEIGHTS ARE SAVED IN THE ROOT DIRECTORY WHERE MLP IS
            model.load_weights('weights-best_k{}_r{}.hdf5'.format(k,r))
            ypred = model.predict(Xval)
            ypred = np.argmax(ypred,axis=1)
            ytrue = np.argmax(yval,axis=1)
            cvconfusion[:,:,counter] = confusion_matrix(ytrue, ypred)
            F1 = np.zeros((3,1))
            for i in range(3):
                F1[i]=2*cvconfusion[i,i,counter]/(np.sum(cvconfusion[i,:,counter])+np.sum(cvconfusion[:,i,counter]))
                print("F1 measure for {} rhythm: {:1.4f}".format(classes[i],F1[i,0]))            
            cvscores.append(np.mean(F1)* 100)
            print("Overall F1 measure: {:1.4f}".format(np.mean(F1)))            
            K.clear_session()
            gc.collect()
            config = ConfigProto()
            config.gpu_options.allow_growth=True            
            sess = tf.compat.v1.Session(config=config)
            K.set_session(sess)
            counter += 1
    # Saving cross validation results 
    scipy.io.savemat('xval_results.mat',mdict={'cvconfusion': cvconfusion.tolist()})  
    return model

###########################
## Function to load data ##
###########################
def loaddata(WINDOW_SIZE, data_file = 'trainingset.mat'):    
    '''
        Load training/test data into workspace
        
        This function assumes you have downloaded and padded/truncated the 
        training set into a local file named "trainingset.mat". This file should 
        contain the following structures:
            - trainset: NxM matrix of N ECG segments with length M
            - traintarget: Nx4 matrix of coded labels where each column contains
            one in case it matches ['A', 'N', 'O', '~'].
        
    '''
    print("Loading data training set")        
    matfile = scipy.io.loadmat(data_file)
    X = matfile['trainset']
    y = matfile['traintarget']
    
    # Merging datasets    
    # Case other sets are available, load them then concatenate
    #y = np.concatenate((traintarget,augtarget),axis=0)     
    #X = np.concatenate((trainset,augset),axis=0)     

    X =  X[:,0:WINDOW_SIZE] 
    return (X, y)


#####################
# Main function   ##
###################

config = ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = Session(config=config)
seed = 7
np.random.seed(seed)

# Parameters
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
run_model = True

weight_save_loc = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/weight_saves/weights_" + now + "_.h5" # _30_train_on_12_____
temp_save = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/temp_cf_test/"

if run_model:
  # Loading data
  # IF YOU CHANGE THIS, CHANGE TRAINING SIZE VARIABLE IN EVAL_MODEL!!!!!
  (X_train,y_train) = loaddata(WINDOW_SIZE, data_file = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/WFDB_testset_mat/12_testingset_9.mat")
  
  # Training model
  model = model_eval(X_train,y_train)
  
  # Outputing results of cross validation
  matfile = scipy.io.loadmat('xval_results.mat')
  cv = matfile['cvconfusion']
  F1mean = np.zeros(cv.shape[2])
  for j in range(cv.shape[2]):
      classes = ['N', 'A', 'O']
      F1 = np.zeros((3,1))
      for i in range(3):
          F1[i]=2*cv[i,i,j]/(np.sum(cv[i,:,j])+np.sum(cv[:,i,j]))        
          print("F1 measure for {} rhythm: {:1.4f}".format(classes[i],F1[i,0]))
      F1mean[j] = np.mean(F1)
      print("mean F1 measure for: {:1.4f}".format(F1mean[j]))
  print("Overall F1 : {:1.4f}".format(np.mean(F1mean)))
  # Plotting confusion matrix
  cvsum = np.sum(cv,axis=2)
  for i in range(3):
      F1[i]=2*cvsum[i,i]/(np.sum(cvsum[i,:])+np.sum(cvsum[:,i]))        
      print("F1 measure for {} rhythm: {:1.4f}".format(classes[i],F1[i,0]))
  F1mean = np.mean(F1)
  print("mean F1 measure for: {:1.4f}".format(F1mean))
  plot_confusion_matrix(cvsum, classes,normalize=True,title='Confusion matrix')
  
  model.save_weights(weight_save_loc)
  
  helper_defs.test_model(model, cf_save_path_name = temp_save + 'cf.png', report_save_path_name = temp_save + 'report.csv', test_signals = X_train, test_labels = y_train)

test_12_lead = False
test_cf_path = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/12_test_cf/"
test_report_path = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/12_test_report/"
test_data_path = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/WFDB_testset_mat/"

weights = "/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/weight_saves/30_epoch_train_on_1"

if test_12_lead:
  model_2 = ResNet_model(WINDOW_SIZE)
  for i in range(4):
    data = test_data_path + '12_testingset_' + str(i+6) + '.mat'
    (X_train,y_train) = loaddata(WINDOW_SIZE, data)
    #for shit in y_train:
     # print(shit)
    #X_train = np.expand_dims(X_train, axis=2)
    #idxval = np.random.choice(Ntrain, Nsamp,replace=False)
    #idxtrain = np.invert(np.in1d(range(X_train.shape[0]),idxval))
    #ytrain = y_train[np.asarray(idxtrain),:]
    #Xtrain = X_train[np.asarray(idxtrain),:,:]
    
    # df_labels = pd.read_csv("/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/datasets/12_lead_single_mat.csv", usecols = [1])
    # df_labels = np.asarray(df_labels)
    # labels = []
    # for j in range(len(df_labels) // 12):
      # labels.append(df_labels[j*12 + i])
      
    #if run_model:
      #weights = weight_save_loc
      
    # print(len(labels))
    # print(len(X_train))
      
    helper_defs.test_model(model_2, 
                          cf_save_path_name = test_cf_path + "report_" + str(i+1) + "_" + now + ".png", 
                          report_save_path_name = test_report_path + "report_" + str(i+1) + "_" + now + ".csv",
                          test_signals = X_train,
                          test_labels = y_train)