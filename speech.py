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
from pymir import AudioFile

def normalize_data(data):
    scaler=sklearn.preprocessing.MinMaxScaler()
    return scaler.fit(data)


def readAudioFile(path):
    extension = os.path.splitext(path)[1]
    try:
        if extension.lower() == '.wav':
            [Fs, x] = wavfile.read(path)
        elif extension.lower() == '.aif' or extension.lower() == '.aiff':
            s = aifc.open(path, 'r')
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            x = numpy.fromstring(strsig, numpy.short).byteswap()
            Fs = s.getframerate()
        else:
            print ("Error in readAudioFile(): Unknown file type!")
            return (-1,-1)
    except IOError: 
        print ("Error: file not found or other I/O error.")
        return (-1,-1)
    return (Fs, x)

lsf= []

def dirWav(dirName,fileExpression):
    types = ('*.wav', '*.aif',  '*.aiff')
    wavFilesList = []
    for root,dirnames,filenames in os.walk(dirName):
        for filename in fnmatch.filter(filenames,fileExpression):
            wavFilesList.append(os.path.join(root,filename))
    #print ("wavFilesList",wavFilesList)

    wavFilesList = sorted(wavFilesList)
    #print ("sorted wavFilesList",wavFilesList)
    emotions=[]
    signal=[]
    i=0
    for wavFile in wavFilesList:
        if(i>600):
            break
        else:
            i+=1
        fs,audio=readAudioFile(wavFile)
        #audio=np.float32(audio[:int(len(audio)/784)*784])
        signal.append(audio)
        fileo=os.path.basename(wavFile)
        emotions.append(fileo[5])

    print(set(emotions))
    dict_emotions={'W':0, 'F':1, 'A':2, 'E':3, 'L':4, 'N':5, 'T':6}


    #one_hot
    emotion_one_hot=[]
    for i in emotions:
        temp=[0]*7
        temp[dict_emotions[i]]=1
        #emotion_one_hot.append(temp)
        emotion_one_hot.append(int(dict_emotions[i]))
    return emotion_one_hot, signal, wavFilesList



height=20
k=0
col, avg, med, std, maxm, minm=[], [], [], [], [], []

emo_labels, signal_data, filename=dirWav('/media/shreya/New Volume1/datasets/EMO-DB/wav/','*.wav')

output=np.asarray(emo_labels)

features=[]
feat=[]
feature=[]
length_features=[]

for i in signal_data:
    temp=audioFeatureExtraction.stFeatureExtraction(i, 16000, 1024, 1024)
    print 'temp', temp.shape
    avg.append(temp.mean(axis=1))
    med.append(np.median(temp,axis=1))
    std.append(np.std(temp,axis=1))
    maxm.append(np.amax(temp,axis=1))
    minm.append(np.amin(temp,axis=1))
    
mean=np.asarray(avg)
median=np.asarray(med)
maximum=np.asarray(maxm)
minimum=np.asarray(minm)
standard_deviation=np.asarray(std)

def reduce_zeroOneNorm(arr):
    arr= arr.astype('float32')
    min= arr.min(axis=0)
    max= arr.max(axis=0)
    arr= (arr-min)/(max-min)
    return np.nan_to_num(arr)

mean= reduce_zeroOneNorm(mean)
median= reduce_zeroOneNorm(median)
maximum= reduce_zeroOneNorm(maximum)
minimum= reduce_zeroOneNorm(minimum)
standard_deviation= reduce_zeroOneNorm(standard_deviation)

print 'mean', mean.shape
print 'median', median.shape
print 'maximum', maximum.shape
print 'minimum', minimum.shape
print 'standard_deviation', standard_deviation.shape

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


labels, arousal, valence= [], [], []

for i in range(0,535):
    if(emo_labels[i]==0):
        labels.append('W')
        arousal.append(2)
        valence.append(0)

    elif(emo_labels[i]==1):
        labels.append('F')
        arousal.append(2)
        valence.append(2)

    elif(emo_labels[i]==2):
        labels.append('A')
        arousal.append(1)
        valence.append(0)
    
    elif(emo_labels[i]==3):
        labels.append('E')
        arousal.append(1)
        valence.append(1)

    elif(emo_labels[i]==4):
        labels.append('L')
        arousal.append(0)
        valence.append(0)

    elif(emo_labels[i]==5):
        labels.append('N')
        arousal.append(1)
        valence.append(1)

    elif(emo_labels[i]==6):
        labels.append('T')
        arousal.append(1)
        valence.append(0)

    else:
        print("unknown label")

# target_labels=np.asarray(labels)
target_labels=np.asarray(labels).reshape(-1,1)
Arousal= np.asarray(arousal).reshape(-1, 1)
Valence= np.asarray(valence).reshape(-1, 1)
filename=np.asarray(filename).reshape(-1,1)

central_feat=np.concatenate((mean,median,standard_deviation,maximum,minimum,Arousal,Valence,target_labels),axis=1)
print('central_feat.shape', central_feat.shape)
df = pd.DataFrame(central_feat,columns=['F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean',
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
                                        'F170_min', 'Arousal', 'Valence', 'Labels'])

# df = pd.DataFrame(central_feat,columns=['F1_mean','F2_mean','F3_mean','F4_mean','F5_mean','F6_mean','F7_mean','F8_mean','F9_mean',
#                                         'F10_mean','F11_mean','F12_mean','F13_mean','F14_mean','F15_mean','F16_mean','F17_mean',
#                                         'F18_mean','F19_mean','F20_mean','F21_mean','F22_mean','F23_mean','F24_mean','F25_mean',
#                                         'F26_mean','F27_mean','F28_mean','F29_mean','F30_mean','F31_mean','F32_mean','F33_mean','F34_mean','F35_median',
#                                         'F36_median',
#                                         'F37_median','F38_median','F39_median','F40_median','F41_median','F42_median','F43_median','F44_median',
#                                         'F45_median','F46_median','F47_median',
#                                         'F48_median','F49_median','F50_median','F51_median','F52_median','F53_median','F54_median','F55_median',
#                                         'F56_median','F57_median','F58_median',
#                                         'F59_median','F60_median','F61_median','F62_median','F63_median','F64_median','F65_median','F66_median','F67_median','F68_median','F69_std','F70_std',
#                                         'F71_std','F72_std','F73_std','F74_std','F75_std','F76_std','F77_std','F78_std','F79_std',
#                                         'F80_std','F81_std','F82_std','F83_std','F84_std','F85_std','F86_std','F87_std','F88_std','F89_std',
#                                         'F90_std','F91_std','F92_std','F93_std','F94_std','F95_std','F96_std','F97_std','F98_std','F99_std',
#                                         'F100_std','F101_std','F102_std','F103_max','F104_max','F105_max', 'Arousal', 'Valence', 'Labels'])

df.to_csv('central_features.csv')
print(len(df.columns))

# lsf= np.array(lsf)
# print ('lsf shape', lsf.shape)

# csv of features generated :)