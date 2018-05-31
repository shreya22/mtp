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
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
import pprint

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

speech_dataset= pd.read_csv('central_features.csv', names=names)
print('speech_dataset.shape', speech_dataset.shape)

array = speech_dataset.values[1:,]
X = array[:, :170]
Y = [array[:, 170], array[:, 171], array[:, 172]]

validation_size = 0.20
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

def predictionModel(x_train, x_validation, y_train, y_validation, hidden_layer, alpha, lr):

     # label_encoder = LabelEncoder()
     # y_validation = label_encoder.fit_transform(y_validation)

     # mlp = MLPClassifier(hidden_layer_sizes= hidden_layer, max_iter=1000, alpha= alpha,
     #                solver='adam', tol=1e-4, random_state=1,
     #                learning_rate_init= lr)
     # mlp.fit(x_train, y_train)


     svm = SVC(C=100, coef0=0.0,
    decision_function_shape='ovo', gamma='auto', kernel='rbf',
    max_iter=100, probability=True)

     svm.fit(x_train, y_train)
     # predictions = svm.predict(x_validation)

     # print('cross validation score', cross_val_score(mlp, x_train, y_train, cv=10, scoring='accuracy').mean())

     predictions = svm.predict_proba(x_validation).astype('float32')
     pred_labels= svm.predict(x_validation)
     print accuracy_score(y_validation, pred_labels)
     return predictions, pred_labels

arousal_dict={
     0: ['L'],
     1: ['A', 'E', 'N', 'T'],
     2: ['F', 'W'] 
}

valence_dict={
     0: ['A', 'L', 'T', 'W'],
     1: ['E', 'N'],
     2: ['F'] 
}

print '-----------------------------------------------------------------------------------'
print 'Calculating over Arousal...'

index_predictionLabels= {}


# over arousal prediction
Y_trainA, Y_validationA= Y[0][ind_train], Y[0][ind_validation]
arousal_predictions, pred_labelsA = predictionModel(X_train, X_validation, Y_trainA.ravel(), Y_validationA.ravel(), (250, 250), 1e-5, 0.001)

# now, get all indices from training data with individual arousals

for i in range (3): 
     validation_indices= ind_validation[ np.where(pred_labelsA.astype(int) == i)[0] ]

     x_arousali_train= X[ ind_train[ np.where(Y_trainA.astype(int) == i)[0] ] ].reshape(-1, X.shape[1])
     x_arousali_validation= X[ validation_indices ].reshape(-1, X.shape[1])

     y_arousali_train= Y[2][ ind_train[ np.where(Y_trainA.astype(int) == i)[0] ] ]
     y_arousali_validation= Y[2][ validation_indices ]

     if len (arousal_dict[i]) == 1:
          for val in validation_indices:
               index_predictionLabels[val[0]] = ( arousal_dict[i][0], 1 )
     else:
          pred, pred_label_values= predictionModel (x_arousali_train, x_arousali_validation, y_arousali_train.ravel(), y_arousali_validation.ravel(), (250, 250), 1e-5, 0.001)
          
          for ind, val in enumerate(validation_indices):
               max_index= pred[ind].argmax(axis=0)

               if val[0] in index_predictionLabels:
                    if pred[ind][max_index] > index_predictionLabels[val[0]][1]:
                         index_predictionLabels[ val[0] ]= ( arousal_dict[i][max_index], pred[ind][max_index] )
               else:
                    index_predictionLabels[ val[0] ]= ( arousal_dict[i][max_index], pred[ind][max_index] )     

print 'Prediction dictionary filled in for arousal predictions..'
# print index_predictionLabels

print '-----------------------------------------------------------------------------------'
print 'Calculating over Valence...'

# over valence prediction
Y_trainV, Y_validationV= Y[1][ind_train], Y[1][ind_validation]
valence_predictions, pred_labelsV= predictionModel(X_train, X_validation, Y_trainV.ravel(), Y_validationV.ravel(), (500, 50), 1e-5, 0.0001)

# now, get all indices from training data with individual valence
for i in range (3): 

     validation_indices= ind_validation[ np.where(pred_labelsV.astype(int) == i)[0] ]

     x_valencei_train= X[ ind_train[ np.where(Y_trainV.astype(int) == i)[0] ] ].reshape(-1, X.shape[1])
     x_valencei_validation= X[ validation_indices ].reshape(-1, X.shape[1])

     y_valencei_train= Y[2][ ind_train[ np.where(Y_trainV.astype(int) == i)[0] ] ]
     y_valencei_validation= Y[2][ validation_indices ]

     if len (valence_dict[i]) == 1:
          for val in validation_indices:
               index_predictionLabels[val[0]] = ( valence_dict[i][0], 1 )
     else:
          pred, pred_label_values= predictionModel (x_valencei_train, x_valencei_validation, y_valencei_train.ravel(), y_valencei_validation.ravel(), (500, 50), 1e-5, 0.0001)

          for ind, val in enumerate(validation_indices):
               max_index= pred[ind].argmax(axis=0)

               if val[0] in index_predictionLabels:
                    if pred[ind][max_index] > index_predictionLabels[val[0]][1]:
                         index_predictionLabels[ val[0] ]= ( valence_dict[i][max_index], pred[ind][max_index] )
               else:
                    index_predictionLabels[ val[0] ]= ( valence_dict[i][max_index], pred[ind][max_index] )     

print 'Prediction dictionary filled in for valence predictions..'

for i in ind_validation:
     i= i[0]
     index_predictionLabels[i] = index_predictionLabels[i][0]

# print pprint.pprint(index_predictionLabels)

actual_predictionLables= {}
for i in ind_validation:
     i= i[0]
     actual_predictionLables[i] = Y[2][i]

print 'Comparing with actual validation labels...'

num_correctPredicted, num_wrongPredicted= 0, 0
for i in ind_validation:
     i= i[0]
     if index_predictionLabels[i] == actual_predictionLables[i]:
          num_correctPredicted+=1
     else:
          num_wrongPredicted+=1

print 'num_correctPredicted', num_correctPredicted
print 'num_wrongPredicted', num_wrongPredicted
print 'accuracy', float(num_correctPredicted) / (num_correctPredicted + num_wrongPredicted)