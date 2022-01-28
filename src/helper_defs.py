import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def test_model(model, 
                cf_save_path_name, 
                report_save_path_name, 
                test_signals = None, 
                test_labels = None, 
                weights = None):
  if (weights == None):     
    model.load_weights("/home/SharedStorage2/NewUsersDir/aledhari/wdudley/mlp/old/cinc-challenge2017-master/deeplearn-approach/backup_weight_saves/weights-best_k4_r0.hdf5")
  
  #if test_signals == None:
   # test_signals = test_signals
    #test_labels = test_labels
  
  Y_pred = model.predict_generator(test_signals)
  y_pred = np.argmax(Y_pred,axis=1)
  test_labels = np.argmax(test_labels,axis=1)
  
  count = 0
  for i in range(len(test_labels)):
    if(test_labels[i] == y_pred[i]):
      count +=1
  print("correct count: ", count)
  
  #print(confusion_matrix(self.test_labels, y_pred))
  #print(classification_report(self.test_labels, y_pred))
  
  cf = confusion_matrix(test_labels, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix = cf)
  disp.plot()
  plt.savefig(cf_save_path_name)
  plt.clf()
  
  report = classification_report(test_labels, y_pred, output_dict=True)
  df = pd.DataFrame(report).transpose()
  df.to_csv(report_save_path_name, index=True)
  
def save_weights(model, save_name):
  self.model.save(save_name)
  
def load_weights(model, save_name):
  model.load_weights(save_name)
  
def show_loss_graph(model, save_path_name):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('model loss / epoch')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(save_path_name)
  plt.clf()
  
def show_acc_graph(model, save_path_name):
  plt.plot(hist.history['accuracy'])
  plt.plot(hist.history['val_accuracy'])
  plt.title('model acc / epoch')
  plt.ylabel('acc')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(save_path_name)
  plt.clf()