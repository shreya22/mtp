import numpy as np
import random
from random import shuffle
import scipy
import os
import scipy.io.wavfile as wavfile
import audioFeatureExtraction
import matplotlib.pyplot as plt
import pandas as pd
import fnmatch
import sklearn.preprocessing
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

names = ['F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean',
                                        'F10_mean','F11_mean','F12_mean','F13_mean','F14_mean','F15_mean','F16_mean','F17_mean',
                                        'F18_mean','F19_mean','F20_mean','F21_mean','F22_mean','F23_mean','F24_mean','F25_mean',
                                        'F26_mean','F27_mean','F28_mean','F29_mean','F30_mean','F31_mean','F32_mean','F33_mean','F34_mean','F35_median',
                                        'F36_median',
                                        'F37_median','F38_median','F39_median','F40_median','F41_median','F42_median','F43_median','F44_median',
                                        'F45_median','F46_median','F47_median',
                                        'F48_median','F49_median','F50_median','F51_median','F52_median','F53_median','F54_median','F55_median',
                                        'F56_median','F57_median','F58_median',
                                        'F59_median','F60_median','F61_median','F62_median','F63_median','F64_median','F65_median','F66_median','F67_median','F68_median','F69_std','F70_std',
                                        'F71_std','F72_std','F73_std','F74_std','F75_std','F76_std','F77_std','F78_std','F79_std',
                                        'F80_std','F81_std','F82_std','F83_std','F84_std','F85_std','F86_std','F87_std','F88_std','F89_std',
                                        'F90_std','F91_std','F92_std','F93_std','F94_std','F95_std','F96_std','F97_std','F98_std','F99_std',
                                        'F100_std','F101_std','F102_std','F103_max','F104_max','F105_max','F106_max','F107_mmax','F108_max','F109_max',
                                        'F110_max','F111_max','F112_max','F113_max','F114_max','F115_max','F116_max','F117_max','F118_max','F119_max',
                                        'F120_max','F121_max','F122_max','F123_max','F124_max','F125_max','F126_max','F127_max','F128_max','F129_max',
                                        'F130_max','F131_max','F132_max','F133_max','F134_max','F135_max','F136_max','F137_min','F138_min','F139_min',
                                        'F140_min','F141_min','F142_min','F143_min','F144_min','F145_min','F146_min','F147_min','F148_min','F149_min',
                                        'F150_min','F151_min','F152_min','F153_min','F154_min','F155_min','F156_min','F157_min','F158_min','F159_min',
                                        'F160_min','F161_min','F162_min','F163_min','F164_min','F165_min','F166_min','F167_min','F168_min','F169_min',
                                        'F170_min','Arousal', 'Valence', 'Labels']

speech_dataset= pd.read_csv('central_features.csv',names=names)
print('speech_dataset.shape', speech_dataset.shape)

array = speech_dataset.values[1:,]
X = array[:, :170]
Y = [array[:, 170], array[:, 171], array[:, 172]]

validation_size = 0.30
seed = 8

print ('array size', array.shape)
print ('X', X.shape)
print ('Y joint', len(Y))

# Adding column for arousal and valence
# 0 => low arousal, 2 => high arousal.
# 0 => -ve valence, 2 => +ve valence
     
# ----------------------------------------------------------
#              Label index         Arousal        Valence
# 1. Anger          0 W               2              0
# 2. Boredom        4 L               0              0
# 3. Disguist       3 E               1              1
# 4. Anxiety        2 A               1              0
# 5. Happiness      1 F               2              2
# 6. Sadness        6 T               1              0
# 7. Neutral        5 N               1              1

index_arr= np.arange(array.shape[0]).reshape(-1, 1)
X_train, X_validation, ind_train, ind_validation= model_selection.train_test_split(X, index_arr, test_size=validation_size, random_state=seed)

print 'training data shape', ind_train.shape
print 'testing data shape', ind_validation.shape

def predictionModel(x_train, x_validation, y_train, y_validation):
     svm = SVC(C=100, coef0=0.0,
    decision_function_shape='ovr', gamma='auto', kernel='rbf',
    max_iter=1000)
     svm.fit(x_train, y_train)
     predictions = svm.predict(x_validation)
     print 'aaaaaaaaaaaa', predictions
     # print(confusion_matrix(y_validation, predictions))
     return predictions, accuracy_score(y_validation, predictions)


prediction_accuracy= {}


print '-----------------------------------------------------------------------------------'
print 'Calculating over Arousal...'

# over arousal prediction
Y_trainA, Y_validationA= Y[0][ind_train], Y[0][ind_validation]
arousal_predictions, a_predAcc= predictionModel(X_train, X_validation, Y_trainA.ravel(), Y_validationA.ravel())
print 'arousal predicting accuracy: ', a_predAcc


# now, get all indices from training data with individual arousals
for i in range (3): 
     x_arousali_train= X[ ind_train[ np.where(Y_trainA.astype(int) == i)[0] ] ].reshape(-1, X.shape[1])
     x_arousali_validation= X[ ind_validation[ np.where(arousal_predictions.astype(int) == i)[0] ] ].reshape(-1, X.shape[1])

     y_arousali_train= Y[2][ ind_train[ np.where(Y_trainA.astype(int) == i)[0] ] ]
     y_arousali_validation= Y[2][ ind_validation[ np.where(arousal_predictions.astype(int) == i)[0] ] ]

     if i == 0:
          prediction_accuracy['arousal', i] = a_predAcc
     else:
          pred, acc= predictionModel (x_arousali_train, x_arousali_validation, y_arousali_train.ravel(), y_arousali_validation.ravel())
          prediction_accuracy['arousal', i] = a_predAcc * acc


print '-----------------------------------------------------------------------------------'
print 'Calculating over Valence...'

# over arousal prediction
Y_trainV, Y_validationV= Y[1][ind_train], Y[1][ind_validation]
valence_predictions, v_predAcc= predictionModel(X_train, X_validation, Y_trainV.ravel(), Y_validationV.ravel())
print 'valence predicting accuracy: ', v_predAcc

# now, get all indices from training data with individual arousals
for i in range (3): 
     x_valencei_train= X[ ind_train[ np.where(Y_trainV.astype(int) == i)[0] ] ].reshape(-1, X.shape[1])
     x_valencei_validation= X[ ind_validation[ np.where(valence_predictions.astype(int) == i)[0] ] ].reshape(-1, X.shape[1])

     y_valencei_train= Y[2][ ind_train[ np.where(Y_trainV.astype(int) == i)[0] ] ]
     y_valencei_validation= Y[2][ ind_validation[ np.where(valence_predictions.astype(int) == i)[0] ] ]

     if i == 2:
          prediction_accuracy['valence', i] = v_predAcc
     else:
          pred, acc= predictionModel (x_valencei_train, x_valencei_validation, y_valencei_train.ravel(), y_valencei_validation.ravel())
          prediction_accuracy['valence', i] = v_predAcc * acc

print prediction_accuracy