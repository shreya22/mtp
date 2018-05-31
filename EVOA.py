import random
import math
import sys
import time
import pandas as pd
from sklearn import svm , metrics ,datasets, neighbors, linear_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# variables

Mf = 170
Mv = 20
MAXITER = 30
SWITCHPROB = 0.8
TH = 10
trainX = []
trainY = []
testX = []
testY = []
vult = []
fitness = []
gBest = -1
best = -1
bbb = -1
# input dataset
def readData() :

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
	Y = array[:, 172]

	validation_size = 0.1	
	seed = 8


	global trainX,trainY,testX,testY
	trainX, testX, trainY, testY= model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	# f = open('gi/train_inp.txt','rb')
	# for line in f :
	# 	trainX.append( [float(i) for i in line.split()])
	# f = open('gi/train_out.txt','rb')
	# for i in f :
	# 	trainY.append(int(i))
	# f = open('gi/test_inp.txt','rb')
	# for line in f :
	# 	testX.append( [float(i) for i in line.split()] )
	# f = open('gi/test_out.txt','rb')
	# for i in f :
	# 	testY.append(int(i))

	trainX = np.array(trainX)
	trainY = np.array(trainY)
	testX = np.array(testX)
	testY = np.array(testY)

	print ('trainX', trainX.shape)
	print ('trainY', trainY.shape)
	print ('testX', testX.shape)
	print ('testY', testY.shape)


	#normalization
	#trainX = preprocessing.normalize(trainX, norm='l2')
	#testX = preprocessing.normalize(testX, norm='l2')

	#scaling
	#min_max_scaler = preprocessing.MinMaxScaler()
	#trainX = min_max_scaler.fit_transform(trainX)
	#testX = min_max_scaler.fit_transform(testX)

	#print trainX[0]
#print trainX[0:5]


#print

def printf() :
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best
	#print trainX[0:5]
	#print len(trainX[0])
	#print trainY[0:5]
	#print len(trainY)
	#print testX[0:5]
	#print testY[0:5]
	print vult[best]
	print 'gbest %f ' % gBest


# initialize vultures

def init_sequence() :
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best
	for i in range(0,Mv) :
		x = random.randrange(1,Mf+1)
		v = []
		for i in range(0,x) :
			v.append(1)
		for i in range(x,Mf):
			v.append(0)

		random.shuffle(v);
		vult.append(v);

def calc_fitness(x) :
	#print 'calc fitness'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best
	trTrainX = trainX
	trTestX = testX

	trTrainX = np.array(trTrainX)
	trTestX = np.array(trTestX)

	#print len(x)
	ct = 0
	lx = len(x)
	for i in range(0,lx) :
		if x[i] == 0 :
			#print trTrainX.shape
			#print trTestX.shape
			#print 'i-ct ',
			#print i-ct
			trTrainX = np.delete(trTrainX,i-ct,1)
			trTestX = np.delete(trTestX,i-ct,1)

			ct = ct + 1

	if trTrainX.shape[1] == 0 :
		return 0.0

	#SVM
	# clf = svm.SVC(gamma = 0.01,C = 100)
	#clf.fit(trTrainX,trainY)
	#predicted = np.array(clf.predict(trTestX))

	# KNN
	# clf = KNeighborsClassifier()
	#clf.fit(trTrainX,trainY)
	#predicted = clf.predict(trTestX)

	#Decision Tree Classifier
	#clf = DecisionTreeClassifier()
	#clf.fit(trTrainX,trainY)
	#predicted = clf.predict(trTestX)

	#Logistic Regression
	#clf = LogisticRegression()
	#clf.fit(trTrainX,trainY)
	#predicted = clf.predict(trTestX)

	# ANN
	clf = MLPClassifier(hidden_layer_sizes= (70, 50, 20), max_iter=1000, alpha= 1e-5,
                    solver='adam', tol=1e-4, random_state=1,
                    learning_rate_init= 0.001, momentum=0.5)

 	# RF
 	# clf= RandomForestClassifier(n_estimators=100)

	# Naive Bayes
	#clf = GaussianNB()
	clf.fit(trTrainX,trainY)
	predicted = clf.predict(trTestX)

	return accuracy_score(testY,predicted)

def init_fitness() :
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv):
		fitness.append(calc_fitness(vult[i]))
		if gBest < fitness[i] :
			gBest = fitness[i]
			best = i

	bbb = max(bbb,gBest)
	#print fitness
	#print gBest

def pebble_tossing() :
	#print 'pebble tossing'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv) :
		switch = random.random()

		if switch < SWITCHPROB :
			#print 'local'
			pos = random.randrange(0,Mf);
			pebble = vult[i][pos]

			nnvult = vult[i][:pos] + vult[i][pos+1:]

			smin = random.randrange(0,Mf-1)
			smax = random.randrange(0,Mf-1)
			if smax < smin :
				smin,smax = smax,smin

			while smax - smin + 1 > TH :
				smax = random.randrange(0,Mf-1)
				if smax < smin :
					smin,smax = smax,smin

			#print smin
			#print smax
			#print nnvult
			#print len(nnvult)
			d = -1
			for j in range(smin,smax+1) :
				vvult = nnvult[:];
				#print vvult
				#print nnvult
				vvult.insert(j,pebble)
				#print j
				#print pebble
				#print vvult
				#print nnvult
				#print len(vvult)
				y = calc_fitness(vvult)
				if y > d:
					d = y
					pos = j

			nnvult.insert(pos,pebble)
			if fitness[i] < d :
				fitness[i] = d
				vult[i] = nnvult[:]

		else :
			#print 'global'
			'''for j in range(0,5):
				target = random.randrange(0,Mf)
				nextTarget = 0 if target+1 >= Mf else target+1
				A = vult[best][target]
				B = vult[best][nextTarget]
				posa ,posb = -1,-1
				nvult = vult[i][:]
				nvult[target] = A
				nvult[nextTarget] = B
				fitn = calc_fitness(nvult)
				if fitn > fitness[i] :
					fitness[i] = fitn
					vult[i] = nvult[:]
			'''
			smin = random.randrange(0,Mf-1)
			smax = random.randrange(0,Mf-1)
			if smax < smin :
				smin,smax = smax,smin

			while smax - smin + 1 > TH :
				smax = random.randrange(0,Mf-1)
				if smax < smin :
					smin,smax = smax,smin

			nvult = vult[i][:]

			for j in range(smin,smax+1) :
					nvult[j] = vult[best][j]

			fitn = calc_fitness(nvult)
			bbb = max(bbb,fitn)
			if fitn > fitness[i] :
				fitness[i] = fitn
				vult[i] = nvult[:]




def rolling_twigs() :
	#print 'rolling twig'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv) :
		lmax = random.randrange(0,Mf)
		lmin = random.randrange(0,Mf)
		if lmax < lmin :
			lmin,lmax = lmax,lmin

		window_len = lmax - lmin + 1

		ds = random.randrange(0,window_len)
		dr = random.randrange(0,2)

		aa = lmin if lmax - ds +1 > lmax else lmax - ds + 1
		bb = lmin if lmin + ds > lmax else lmin + ds

		st = aa if dr == 0 else bb
		arr = []

		for j in range(0,window_len) :
			arr.append(vult[i][st])
			st = lmin if st + 1 > lmax else st+1

		nvult = vult[i][:]
		for j in range(0,window_len) :
			nvult[j+lmin] = arr[j]

		nfit = calc_fitness(nvult)
		bbb = max(bbb,nfit)
		if fitness[i] < nfit :
			fitness[i] = nfit
			vult[i] = nvult[:]

def change_Of_Angle() :
	#print 'change of angle'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	for i in range(0,Mv) :
		lmax = random.randrange(0,Mf)
		lmin = random.randrange(0,Mf)
		if lmax < lmin :
			lmin,lmax = lmax,lmin

		nvult = vult[i]
		nvult[lmin:lmax+1] = reversed(nvult[lmin:lmax+1])
		nfit = calc_fitness(nvult)
		bbb = max(bbb,nfit)
		if fitness[i] < nfit :
			fitness[i] = nfit
			vult[i] = nvult[:]


def egyptian_vulture() :
	#print 'egyptian vulure'
	global Mf,Mv,MAXITER,SWITCHPROB,TH,trainX,trainY,testX,testY,vult,fitness,gBest,best,bbb
	readData()
	init_sequence()
	init_fitness()

	for i in range(0,MAXITER) :
		print 'iteration number : %d ' % i
		pvult = vult[:]
		pfitness = fitness[:]

		pebble_tossing()
		rolling_twigs()
		change_Of_Angle()

		for j in range(0,Mv) :
			if fitness[j] < pfitness[j] :
				fitness[j] = pfitness[j]
				vult[j] = pvult[j][:]
			if gBest < fitness[j] :
				gBest = fitness[j]
				best = j
			bbb = max(bbb,gBest)

	print 'GBEST %f ' % gBest
	print 'BBB %f ' % bbb

# main

def main():
	start_time = time.time()
	egyptian_vulture()
	printf()
	print("--- %s seconds ---" % (time.time() - start_time))
	return 0

if __name__ == '__main__':
	main()

