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
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

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
                                        'F170_min','Arousal','Valence', 'Labels']

speech_dataset= pd.read_csv('/media/shreya/New Volume2/mtp/2017/mfcc/central_features.csv',names=names)
print('speech_dataset.shape', speech_dataset.shape)

array = speech_dataset.values[1:,]
X = array[:, :170]
Y = array[:, 172]

validation_size = 0.1
seed = 8

print ('array size', array.shape)
print ('X', X.shape)
print ('Y', Y.shape)

_rf, _knn, _svm, _ann= np.array([]), np.array([]), np.array([]), np.array([])

for i in range(3):

     X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)
     print (i)

     # print 'Random Forest'
     logistic = RandomForestClassifier(n_estimators=100)
     logistic.fit(X_train, Y_train)
     predictions = logistic.predict(X_validation)
     _rf= np.append(_rf, accuracy_score(Y_validation, predictions))

     # print 'KNN'
     knn= KNeighborsClassifier(n_neighbors=5)
     knn.fit(X_train, Y_train)
     predictions= knn.predict(X_validation)
     _knn= np.append(_knn, accuracy_score(Y_validation, predictions))

     # print 'SVM'
     svm = SVC(C=100, coef0=0.0,
         decision_function_shape='ovo', gamma='auto', kernel='rbf',
         max_iter=1000)
     svm.fit(X_train, Y_train)
     predictions = svm.predict(X_validation)
     _svm= np.append(_svm, accuracy_score(Y_validation, predictions))

     # ANN
     ann = MLPClassifier(hidden_layer_sizes= (70, 50, 20), max_iter=1000, alpha= 1e-5,
                    solver='adam', tol=1e-4, random_state=1,
                    learning_rate_init= 0.001, momentum=0.5)
     ann.fit(X_train, Y_train)
     predictions = ann.predict(X_validation)
     _ann= np.append(_ann, accuracy_score(Y_validation, predictions))
     
print ('RF', _rf.mean())
print ('KNN', _knn.mean())
print ('SVM', _svm.mean())
print ('ANN', _ann.mean())