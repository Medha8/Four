import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


count=0
X = []; Y = []
with open("test.csv", 'r') as f:
	#line = file.readline()
	for line in f:
		if line.strip():
			line = line.strip().split(',')
			
 		
 		#print line
 			#for u in line:
 			#	u=float(u)
 			#print line[12]
			Y.append(line[0])
			X.append([float(x) for x in line[1:]])
			

 			#count=count+1
 		
X = np.asarray(X); Y = np.asarray(Y)
#print X
#print Y
#Standardize features by removing the mean and scaling to unit variance.Centering and scaling
# happen independently on each feature by computing the relevant statistics on the samples in the training set
#X = [[0, 0], [1, 1]]
#Y = [0, 1]
X = StandardScaler().fit_transform(X)
#print X

#X=preprocessing.scale(X)

clf = svm.NuSVC(kernel='rbf',nu=.5)
clf.fit(X, Y) 


Y_pred=clf.predict(X)
	#print Y_pred
train_accuracy = accuracy_score(Y_pred, Y)
print i,train_accuracy

